import numbers

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

OUT = 'out'


class GlobalOffsetArray(np.ndarray, NDArrayOperatorsMixin):
    """
    A simple VIEW CAST of a given ndarray that is addressed via global coordinates. Negative wraparound indices are NOT
    supported (i.e. used for printing out) and will IGNORE indices out of bounds

    See below link for explanations of __new__ and __array_finalize__!
    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    """# noqa
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __new__(cls, input_array, global_offset=None, *args, **kwargs):
        if not isinstance(input_array, np.ndarray):
            return input_array

        obj = np.asarray(input_array).view(cls)
        if global_offset is None:
            global_offset = tuple([0] * input_array.ndim)

        obj.global_offset = tuple(global_offset)
        if len(global_offset) != len(input_array.shape):
            raise ValueError("Global offset %s does not have same number dimensions as input_array shape %s" % (
                global_offset, input_array.shape))
        obj._bounds = None

        return obj

    def __array_finalize__(self, obj):
        """
        Called whenever the array is new-ed. Because unpickling does NOT call __array_finalize__,
        make sure they are equivalent
        """
        if obj is None:
            return
        self.global_offset = getattr(obj, 'global_offset', None)
        self._bounds = getattr(obj, '_bounds', None)

    def __reduce__(self):
        reduction = super().__reduce__()
        object_state = reduction[2]
        object_state += (self.global_offset,)
        return tuple(object_state if index is 2 else r for index, r in enumerate(reduction))

    def __setstate__(self, state):
        """
        Called when unpickling. Because unpickling does NOT call __array_finalize__, make sure they are equivalent
        """
        self.global_offset = state[-1]
        self._bounds = None
        super().__setstate__(state[:-1])

    def _to_internal_slices(self, index):
        """
        Convert given index into the index used in the internal ndarray. Does NOT support end slicing and wrap around
        negative indices. Throw an error if computed internal indices are outside the range of the data.
        """
        internal_index = ()
        new_global_offsets = ()

        if type(index) == int or type(index) == slice:
            index = (index,) + (slice(None),) * (len(self.shape) - 1)
        elif len(self.shape) > len(index):
            # Fill rest of dimensions of index that were not specified
            index = index + (slice(None),) * (len(self.shape) - len(index))

        for dimension, item in enumerate(index):
            offset = self.global_offset[dimension]
            length = self.shape[dimension]

            if item is None:
                new_item = None
                # don't need to keep track of global offset for collapsed index
            else:
                try:
                    start = item.start
                    stop = item.stop
                    if start is None:
                        start = offset
                    if stop is None:
                        stop = offset + length

                    slice_start = start - offset
                    slice_stop = stop - offset

                    new_item = slice(slice_start if slice_start > 0 else 0, slice_stop if slice_stop > 0 else 0, item.step)
                    new_global_offsets += (new_item.start + offset,)
                except AttributeError:  # Not a slice
                    new_item = item - offset
                    # don't need to keep track of global offset for collapsed index

                    if new_item < 0 or new_item > length:
                        raise IndexError('Index %s is out of bounds for axis %s with bounds [%s , %s) '
                                         'requested: %s bounds: %s' % (
                                             new_item, dimension, offset, offset + length, index, self.bounds))

            internal_index += (new_item,)
        return (internal_index, new_global_offsets)

    def __getitem__(self, index):
        """
        Access the array based on global coordinates. If we receive a tuple, it means we are slicing.
        When we slice, calculate the actual coordinates stored
        """
        internal_index, new_global_offset = self._to_internal_slices(index)

        new_from_template = super(GlobalOffsetArray, self).__getitem__(internal_index)
        if hasattr(new_from_template, 'global_offset'):
            new_from_template.global_offset = new_global_offset
            if new_from_template.shape != self.shape or self.global_offset != new_global_offset:
                new_from_template._bounds = new_from_template.calculate_bounds()

        return new_from_template

    def __setitem__(self, index, value):
        """
        Access the array based on global coordinates. If we receive a tuple, it means we are slicing.
        When we slice, calculate the actual coordinates stored
        """
        internal_index, _ = self._to_internal_slices(index)

        # use view instead of super because super will call the overriden __getitem__ function
        self.view(np.ndarray).__setitem__(internal_index, value)

    def __str__(self):
        """
        Overwrite string conversion to create a view instead of calling super. Super will call with the overridden
        __getitem__ function which will not work
        """
        if self.global_offset is not None:
            return '%s, global_offset: %s' % (self.view(np.ndarray).__str__(), self.global_offset)
        else:
            return super().__str__()

    def __repr__(self):
        """
        Overwrite string conversion to create a view instead of calling super. Super will call with the overridden
        __getitem__ function which will not work
        """
        return self.view(np.ndarray).__repr__()

    def bounds(self):
        """
        Get slices that are bounds of the available data
        """
        if self._bounds is None:
            self._bounds = self.calculate_bounds()
        return self._bounds

    def calculate_bounds(self):
        return tuple(slice(offset, offset + shape) for shape, offset in zip(self.shape, self.global_offset))

    def is_contained_within(self, other):
        """
        Check to see if this volume is contained within other
        """
        self_bounds = self.bounds()
        other_bounds = other.bounds()
        return not any(other_slice.start > self_slice.start or self_slice.start > other_slice.stop or
                       other_slice.start > self_slice.stop or self_slice.stop > other_slice.stop
                       for self_slice, other_slice in zip(self_bounds, other_bounds))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): # noqa: C901
        """
        Enable injection of customized indexing for ufunc operations
        Must defer to the implementation of the ufunc on unwrapped values to avoid infinite loop
        https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html

        Standard operators work normally when:
            * global offset and size are the same for both operands
            * one oeprand is fully encapsulated by another (returns a copy of the larger with the smaller added)

        In-place operators work normally when:
            *
        """
        in_place = OUT in kwargs
        for x in inputs + kwargs.get(OUT, ()):
            # Use GlobalOffsetArray instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle GlobalOffsetArray objects.
            if not isinstance(x, self._HANDLED_TYPES + (GlobalOffsetArray,)):
                return NotImplemented

        # global offset to use in the result
        result = global_offset = None
        if len(inputs) == 2:
            left = inputs[0]
            right = inputs[1]

            try:
                smaller = larger = None
                left_in_right = left.is_contained_within(right)
                right_in_left = right.is_contained_within(left)

                if left_in_right and right_in_left:
                    # same bounds/size
                    global_offset = left.global_offset
                else:
                    smaller = left if left_in_right else right
                    larger = right if left_in_right else left

                    sub_left = left[smaller.bounds()]
                    sub_right = right[smaller.bounds()]

                    sub_inputs = (sub_left, sub_right)
                    sub_kwargs = {}

                    if in_place:
                        # only perform op if there are values to operate on
                        if sub_left.size and sub_right.size:
                            sub_kwargs[OUT] = tuple(o[right.bounds()] for o in kwargs[OUT])
                            getattr(ufunc, method)(*sub_inputs, **sub_kwargs)

                        result = left
                        global_offset = left.global_offset
                    else:
                        if not left_in_right and not right_in_left:
                            raise ValueError("Non-in-place operations on overlapping GlobalOffsetArrays unsupported. "
                                             "Left bounds: %s, Right bounds: %s" % (left.bounds(), right.bounds()))
                        # Return a copy of the larger operand and perform in place on the sub_array of that copy
                        sample_type = type(getattr(ufunc, method)(sub_left.item(0), sub_right.item(1)))

                        result = larger.astype(sample_type)
                        sub_kwargs[OUT] = (result[smaller.bounds()])
                        sub_result = getattr(ufunc, method)(*sub_inputs, **sub_kwargs)
                        result[smaller.bounds()] = sub_result

                        global_offset = larger.global_offset
            except AttributeError:  # At least one of arguments is not a GlobalOffsetArray
                try:
                    global_offset = left.global_offset
                except AttributeError:  # Left is not a GlobalOffsetArray
                    global_offset = right.global_offset

            inputs = (left, right)

        # Must defer to the implementation of the ufunc on unwrapped values to avoid infinite loop
        inputs = tuple(i.view(np.ndarray) if isinstance(i, GlobalOffsetArray) else i for i in inputs)
        if in_place:
            kwargs[OUT] = tuple(o.view(np.ndarray) if isinstance(o, GlobalOffsetArray) else o for o in kwargs[OUT])

        if result is None:
            result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x, global_offset=global_offset) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return type(self)(result, global_offset=global_offset)
