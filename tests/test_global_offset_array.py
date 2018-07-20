import operator
import pickle
from math import factorial, floor

import numpy as np
import pytest

from chunkblocks.global_offset_array import GlobalOffsetArray

"""
create test arrays of increasing size AND dimensions i.e.
[0 1]
[[0 1 2]
 [3 4 5]]
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]
"""
TEST_ARRAYS = [np.arange(factorial(dimension + 1)).reshape(tuple(i for i in range(2, dimension + 2)))
               for dimension in range(1, 4)]
STANDARD_OPERATORS = {
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
    operator.lshift,
    operator.rshift,
    operator.and_,
    operator.or_,
    operator.xor
}
IN_PLACE_OPERATORS = {
    operator.iadd,
    operator.isub,
    operator.imul,
    operator.itruediv,
    operator.ifloordiv,
    operator.imod,
    operator.ipow,
    operator.ilshift,
    operator.irshift,
    operator.iand,
    operator.ior,
    operator.ixor
}


def no_wrap_slices(slices):
    """
    Prevent wrapping around index for array indexing (only fixes the start of the slice because negative indices wrap)
    """
    def normalize_slice(val):
        if not isinstance(val, slice):
            return 0 if val < 0 else val
        else:
            return slice(
                None if val.start is None else 0 if val.start < 0 else val.start,
                None if val.stop is None else 0 if val.stop < 0 else val.stop
            )
    return tuple(normalize_slice(s) for s in slices)


class TestGlobalOffsetArray:

    def recurse_compare(self, left_array, right_array, shape, index=(), dimension=0):
        if isinstance(left_array, GlobalOffsetArray) and isinstance(right_array, GlobalOffsetArray) and \
                left_array.global_offset != right_array.global_offset:
            raise ValueError("Comparing arrays with different global_offset %s, %s", (left_array.global_offset,
                                                                                      right_array.global_offset))
        elif isinstance(left_array, GlobalOffsetArray):
            offset = left_array.global_offset
        elif isinstance(right_array, GlobalOffsetArray):
            offset = right_array.global_offset
        else:
            offset = tuple([0] * len(shape))
        self._recurse_compare_helper(left_array, right_array, shape, index, dimension, offset)

    def _recurse_compare_helper(self, left_array, right_array, shape, index, dimension, offset):
        """
        Compare 2 ndarray-likes to make sure all values are equivalent
        :param ndarray | offset_array: array for for comparison
        :param ndarray | offset_array: array for for comparison
        :param shape: shape of the dataset we are comparing
        """
        if dimension == len(shape):
            offset_index = tuple(item + offset[dimension] for dimension, item in enumerate(index))

            if isinstance(left_array, GlobalOffsetArray):
                left_index = offset_index
            else:
                left_index = index

            if isinstance(right_array, GlobalOffsetArray):
                right_index = offset_index
            else:
                right_index = index

            assert left_array[left_index] == right_array[right_index]
        else:
            for i in range(0, shape[dimension]):
                self.recurse_compare(left_array, right_array, shape, index + (i,), dimension + 1)

    def test_no_wrap(self):
        """
        Make sure index ranges before start does not perform wrap around
        """

        def test_recurse_slices(expected, actual, index, shape, slices):
            if index == len(shape):
                normalized_slices = no_wrap_slices(slices)
                assert np.array_equal(expected[normalized_slices], actual[slices]), \
                    'Incorrect value found requested slices: %s, normalized slices: %s, actual: %s, expected: %s' % (
                        slices, normalized_slices, actual[slices], expected[normalized_slices])

            else:
                dim = shape[index]
                for start in range(-dim, dim):
                    for stop in range(start, dim * 2):
                        test_recurse_slices(expected, actual, index + 1, shape, (slices + (slice(start, stop),)))

        for test_array in TEST_ARRAYS:
            offset_array = GlobalOffsetArray(test_array)
            test_recurse_slices(test_array, offset_array, 0, offset_array.shape, ())

    def test_get_no_offset(self):
        """
        Make sure all arrays are properly equivalent when no offset is given
        """
        for test_array in TEST_ARRAYS:
            offset_array = GlobalOffsetArray(test_array)
            assert offset_array.global_offset == tuple([0] * test_array.ndim)
            assert np.array_equal(test_array, offset_array)

    def test_get_offset_origin(self):
        """
        Make sure all arrays are properly equivalent when offset of the origin is given
        """
        for test_array in TEST_ARRAYS:
            offset_array = GlobalOffsetArray(test_array, global_offset=tuple(0 for _ in range(0, test_array.ndim)))
            assert offset_array.global_offset == tuple([0] * test_array.ndim)
            assert np.array_equal(test_array, offset_array)
            self.recurse_compare(test_array, offset_array, test_array.shape)

    def test_get_with_offset(self):
        """
        Make sure all global_offset_arrays are equivalent when given offset the proper offset indices.
        """
        for test_array in TEST_ARRAYS:
            # set offset at each dimension to the dimension index + 1
            offset = tuple([index + 1 for index in range(0, len(test_array.shape))])
            shape = test_array.shape
            offset_array = GlobalOffsetArray(test_array, global_offset=offset)
            assert offset_array.global_offset == offset

            test_slices = tuple(slice(0, shape[dimension]) for dimension in range(0, test_array.ndim))
            # same as test_slices but at offset
            offset_slices = tuple(slice(test_slice.start + offset[dimension], test_slice.stop + offset[dimension])
                                  for dimension, test_slice in enumerate(test_slices))
            sliced_offset_array = offset_array[offset_slices]
            assert np.array_equal(test_array[test_slices], sliced_offset_array)
            self.recurse_compare(test_array[test_slices], sliced_offset_array, test_array.shape)

    def test_bad_offset(self):
        """
        Make sure error is thrown when trying to access out of bounds

        """
        original = np.arange(5 ** 4).reshape(tuple([5] * 4))
        global_offset = (100, 200, 300, 400)

        with pytest.raises(ValueError):
            GlobalOffsetArray(original, global_offset=global_offset + (32,))

        with pytest.raises(ValueError):
            GlobalOffsetArray(original, global_offset=global_offset[1:])

    def test_bounds(self):
        """
        Tests to make sure get index is has compatible behavior as the regular numpy array

        """
        original = np.arange(5 ** 4).reshape(tuple([5] * 4))
        global_offset = (100, 200, 300, 400)

        no_offset_array = GlobalOffsetArray(original)
        offset_array = GlobalOffsetArray(original, global_offset=global_offset)

        slices_tests = [
            # pre indexed sub slicing
            (slice(-5, 2), slice(0, 5), slice(0, 5), slice(0, 5)),
            # post indexed sub slicing
            (slice(3, 8), slice(0, 5), slice(0, 5), slice(0, 5)),
            # pre and post indexed mixed sub slicing
            (slice(-5, 2), slice(3, 8), slice(4, 7), slice(2, 8)),
            # pre to post sub slicing
            (slice(-2, 8), slice(0, 5), slice(0, 5), slice(0, 5)),
            # completely outside bounding box
            (slice(5, 6), slice(4, 5), slice(4, 7), slice(7, 8)),
            # single addressed index
            (2, slice(2, 5), slice(6, 8), slice(0, 2)),
            # single addressed index out of bounds
            (8, slice(2, 5), slice(6, 8), slice(0, 2)),
            # two addressed index but one is out of bounds
            (2, 9, slice(0, 3), slice(0, 2)),
        ]
        offset_slices_tests = [
            tuple(
                slice(s.start + o, s.stop + o) if isinstance(s, slice) else s + o
                for s, o in zip(slices, global_offset)
            ) for slices in slices_tests
        ]

        for slices, offset_slices in zip(slices_tests, offset_slices_tests):
            expected_exception = None
            expected_value = None
            try:
                expected_value = original[no_wrap_slices(slices)]
            except Exception as e:
                expected_exception = e

            if expected_exception is not None:
                with pytest.raises(type(expected_exception)):
                    no_offset_array[slices]
                with pytest.raises(type(expected_exception)):
                    offset_array[offset_slices]
            else:
                assert np.array_equal(expected_value, no_offset_array[slices])
                assert np.array_equal(expected_value, offset_array[offset_slices])

    def test_subarray(self):
        """
        Make sure subarrays of contain the correct adjusted global_offset and a copy is returned

        """

        def to_offsets(slice_or_indices):
            return tuple(
                slice(s.start + o, s.stop + o) if isinstance(s, slice) else s + o
                for s, o in zip(original_index, global_offset)
            )

        original = np.arange(5 ** 4).reshape(tuple([5] * 4))
        global_offset = (100, 200, 300, 400)
        offset_array = GlobalOffsetArray(original, global_offset=global_offset)

        # test slice with only slices
        original_index = (slice(0, 2), slice(2, 5), slice(3, 5), slice(0, 3))
        offset_index = to_offsets(original_index)
        sub_array = offset_array[offset_index]
        assert np.array_equal(sub_array, original[original_index])
        assert sub_array.global_offset == (100, 202, 303, 400)
        assert np.array_equal(sub_array[offset_index], offset_array[offset_index])

        # ensure that returned sub_array is actually a view
        sub_array[sub_array.global_offset] = 1337
        assert offset_array[sub_array.global_offset] == 1337

        # test slice with some slices some fixed
        original_index = (slice(0, 2), 3, slice(3, 5), slice(0, 3))
        offset_index = to_offsets(original_index)
        sub_array = offset_array[offset_index]
        assert np.array_equal(original[original_index], sub_array)
        assert sub_array.global_offset == (100, 303, 400)
        assert np.array_equal(sub_array[tuple(s for s in offset_index if isinstance(s, slice))],
                              offset_array[offset_index])

    def generate_data(self, ndim, length):
        """
        Generate test data
        """
        original = np.arange(1, length ** ndim + 1).reshape(tuple([length] * ndim))
        copy = original.copy()

        global_offset = tuple(dimension * 100 for dimension in range(1, ndim + 1))
        offset_array = GlobalOffsetArray(original, global_offset=global_offset)

        return (copy, offset_array)

    def generate_replacement(self, ndim, length, global_offset):
        """
        """
        # Test with regulard ndarray are properly set into the offset_array
        replacement_length = floor(length / 2)
        replacement = np.arange(1, replacement_length ** ndim + 1).reshape(
            tuple([replacement_length] * ndim))

        # replace global offset array with new replaced value
        replacement_slice = ()
        offset_replace_slice = ()
        replacement_offset = ()
        for offset in global_offset:
            replacement_slice += (slice(replacement_length, replacement_length * 2),)
            offset_replace_slice += (slice(replacement_length + offset, replacement_length * 2 + offset),)
            replacement_offset += (replacement_length + offset,)

        replacement = GlobalOffsetArray(replacement, global_offset=replacement_offset)

        return (replacement_slice, offset_replace_slice, replacement)

    def test_set(self):
        """
        Make sure slice setting modifies the correct values
        """
        ndim = 5
        length = 4
        (expected, offset_array) = self.generate_data(ndim, length)
        (replacement_slice, offset_replace_slice, replacement) = self.generate_replacement(ndim, length,
                                                                                           offset_array.global_offset)
        replacement = replacement.view(np.ndarray)

        # perform action on expected result
        expected[replacement_slice] = replacement
        offset_array[offset_replace_slice] = replacement

        # check replaced values are correctly replaced
        assert np.array_equal(offset_array[offset_replace_slice].view(np.ndarray), replacement)
        # check that the expected and the new array has same values
        assert np.array_equal(offset_array, expected)
        self.recurse_compare(expected, offset_array, offset_array.shape)
        # check replaced values are correctly replaced
        assert np.array_equal(offset_array[offset_replace_slice].view(np.ndarray), replacement)

        # ensure direct index set is correct
        offset_array[offset_array.global_offset] = 1
        assert offset_array[offset_array.global_offset] == 1

    def test_operator_same_size_ndarray(self):
        """
        Test that when using operators on one GlobalOffsetArray and one same size ndarray, it operates the same as
        with two ndarrays.
        """
        ndim = 5
        length = 4

        for op in STANDARD_OPERATORS | IN_PLACE_OPERATORS:
            for forward in [True, False]:
                (original, offset_array) = self.generate_data(ndim, length)
                operate_param = np.ones(offset_array.shape, dtype=offset_array.dtype) * 10
                # itrue div requires floats when doing true division (can't do in place conversion to float)
                if op == operator.itruediv:
                    original = original.astype(np.float64)
                    offset_array = offset_array.astype(np.float64)
                    operate_param = operate_param.astype(np.float64)
                if forward:
                    left_expected = original
                    right_expected = operate_param
                    left_offset = offset_array
                    right_offset = operate_param
                else:
                    # test operation commutativity
                    left_expected = operate_param
                    right_expected = original
                    left_offset = operate_param
                    right_offset = offset_array

                expected_result = op(left_expected, right_expected)
                actual_result = op(left_offset, right_offset)

                if op in STANDARD_OPERATORS:
                    expected = expected_result
                    actual = actual_result
                else:
                    expected = original
                    actual = offset_array

                # ensure global_offset is preserved
                assert actual.global_offset == offset_array.global_offset

                # ensure actual results match that of a regular ndarray
                assert np.array_equal(actual, expected)
                self.recurse_compare(expected, actual, offset_array.shape)

                # ensure the results that are returned are a copy of an array instead of a view just like ndarray
                expected[tuple([0] * ndim)] = 1337
                actual[actual.global_offset] = 1337

                # original arrays were not modified (or they were, compare same result from regular ndarray op)
                assert np.any(offset_array == 1337) == np.any(original == 1337)

    def test_operator_diff_size_ndarray(self):
        """
        Test to make sure operators fail when given a different size ndarray just like with 2 ndarrays
        """
        ndim = 5
        length = 4

        for op in STANDARD_OPERATORS | IN_PLACE_OPERATORS:
            for forward in [True, False]:
                (original, offset_array) = self.generate_data(ndim, length)
                half_size = tuple(floor(size/2) for size in offset_array.shape)
                operate_param = np.ones(half_size, dtype=offset_array.dtype) * 10
                # itrue div requires floats when doing true division (can't do in place conversion to float)
                if op == operator.itruediv:
                    original = original.astype(np.float64)
                    offset_array = offset_array.astype(np.float64)
                    operate_param = operate_param.astype(np.float64)

                if forward:
                    left_expected = original
                    right_expected = operate_param
                    left_offset = offset_array
                    right_offset = operate_param
                else:
                    # test operation commutativity
                    left_expected = operate_param
                    right_expected = original
                    left_offset = operate_param
                    right_offset = offset_array

                error = None
                try:
                    op(left_expected, right_expected)
                except Exception as e:
                    error = e

                with pytest.raises(error.__class__, match=str(error).replace('(', '\\(').replace(')', '\\)')):
                    op(left_offset, right_offset)

    def test_standard_operators(self):
        """
        Tests that standard operators will only work when:
            * global offset and size are same for both operands
            * one operand is fully encapsulated by another (returns a copy of the larger with the smaller added)

        All other cases should throw errors
        """
        ndim = 5
        length = 4

        all_slices = (slice(None, None),) * ndim
        # test all operators
        for op in STANDARD_OPERATORS:
            # test operation commutativity
            for forward in [True, False]:
                (original, fixed_operand_template) = self.generate_data(ndim, length)
                # test different sizes
                for shape in [fixed_operand_template.shape, tuple(s * 2 for s in fixed_operand_template.shape),
                              tuple(s // 2 + 1 for s in fixed_operand_template.shape)]:
                    # test different overlap offsets
                    for offset_of_offset in [-2, 0, 2, 100]:
                        fixed_operand = fixed_operand_template.copy()
                        # offset array used as other operand for operator
                        offsetted_operand = GlobalOffsetArray(
                            np.arange(np.prod(shape), dtype=fixed_operand.dtype).reshape(shape) * 10 + 1,
                            global_offset=tuple(offset_of_offset + offset for offset in fixed_operand.global_offset)
                        )

                        expected_error = expected_result = None
                        actual_error = actual_result = None

                        fixed_in_offsetted = fixed_operand.is_contained_within(offsetted_operand)
                        offsetted_in_fixed = offsetted_operand.is_contained_within(fixed_operand)

                        # Determine the slices when one is fully encapsulated by the other
                        fixed_slices = tuple(
                            slice(offset_of_offset, offset_of_offset + s)
                            for s in offsetted_operand.shape
                        )
                        offsetted_slices = tuple(
                            slice(-1 * offset_of_offset, -1 * offset_of_offset + s)
                            for s in fixed_operand.shape
                        )

                        if fixed_in_offsetted and not offsetted_in_fixed:
                            sub_slices = offsetted_slices
                            expected_result = offsetted_operand.view(np.ndarray).copy()
                        elif not fixed_in_offsetted and offsetted_in_fixed:
                            sub_slices = fixed_slices
                            expected_result = original.view(np.ndarray).copy()
                        else:
                            # either exactly the same or separate slices
                            fixed_slices = offsetted_slices = all_slices
                            sub_slices = all_slices

                        # swap operands around on forward flag
                        if forward:
                            left_expected = original.copy()
                            left_actual = fixed_operand
                            left_slices = fixed_slices

                            right_expected = offsetted_operand.view(np.ndarray)
                            right_actual = offsetted_operand
                            right_slices = offsetted_slices
                        else:
                            left_expected = offsetted_operand.view(np.ndarray).copy()
                            left_actual = offsetted_operand
                            left_slices = offsetted_slices

                            right_expected = original.view(np.ndarray)
                            right_actual = fixed_operand
                            right_slices = fixed_slices

                        left_slices = no_wrap_slices(left_slices)
                        right_slices = no_wrap_slices(right_slices)
                        sub_slices = no_wrap_slices(sub_slices)

                        # Condition to autofail on partial overlaps
                        if not fixed_in_offsetted and not offsetted_in_fixed:
                            with pytest.raises(ValueError):
                                op(left_actual, right_actual)
                            continue

                        try:
                            interim_result = op(left_expected[left_slices], right_expected[right_slices])
                        except Exception as e:
                            expected_error = e

                        if expected_result is not None:
                            expected_result = expected_result.astype(interim_result.dtype)
                            expected_result[sub_slices] = interim_result
                        else:
                            expected_result = interim_result

                        try:
                            actual_result = op(left_actual, right_actual)
                        except Exception as e:
                            actual_error = e

                        assert expected_error is None == actual_error is None, \
                            'Expected error: (%s) Actual error: (%s)' % (expected_error, actual_error) # noqa E711

                        # ensure global_offset is preserved
                        assert hasattr(actual_result, 'global_offset')
                        assert tuple(offset_of_offset + o for o in fixed_operand.global_offset) == actual_result.global_offset

                        # ensure actual results match that of a regular ndarray
                        assert np.array_equal(expected_result, actual_result)

                        # ensure the results behave like ndarray to return a copy of the array instead of a view by
                        # testing that the original arrays were not modified
                        actual_result[actual_result.global_offset] = 1337
                        assert np.all(fixed_operand != 1337)

    def test_in_place_operators(self):
        """
        Tests that in-place operators will only work when:
            * global offset and size are same for both operands
            * one operand is fully encapsulated by another (returns a copy of the larger with the smaller added)

        All other cases should throw errors
        """
        ndim = 5
        length = 4

        # test all operators
        for op in IN_PLACE_OPERATORS:
            # test operation commutativity
            for forward in [True, False]:
                (original, fixed_operand_template) = self.generate_data(ndim, length)

                # in place only works with floats because we can't mix int and float into the same ndarray
                if op == operator.itruediv:
                    original = original.astype(np.float64)
                    fixed_operand_template = fixed_operand_template.astype(np.float64)

                # test different sizes
                for shape in [fixed_operand_template.shape, tuple(s * 2 for s in fixed_operand_template.shape),
                              tuple(s // 2 + 1 for s in fixed_operand_template.shape)]:
                    # test different overlap offsets
                    for offset_of_offset in [-2, 0, 2, 100]:
                        fixed_operand = fixed_operand_template.copy()

                        # offset array used as other operand for operator
                        offsetted_operand = GlobalOffsetArray(
                            np.arange(np.prod(shape), dtype=fixed_operand.dtype).reshape(shape) * 10 + 1,
                            global_offset=tuple(offset_of_offset + offset for offset in fixed_operand.global_offset)
                        )

                        expected_error = expected_result = None
                        actual_error = actual_result = None

                        fixed_in_offsetted = fixed_operand.is_contained_within(offsetted_operand)
                        offsetted_in_fixed = offsetted_operand.is_contained_within(fixed_operand)

                        fixed_slices = tuple(
                            slice(offset_of_offset, offset_of_offset + s)
                            for s in offsetted_operand.shape
                        )
                        offsetted_slices = tuple(
                            slice(-1 * offset_of_offset, -1 * offset_of_offset + s)
                            for s in fixed_operand.shape
                        )

                        # swap operands around on forward flag
                        if forward:
                            left_expected = original.copy()
                            left_actual = fixed_operand
                            left_slices = fixed_slices

                            right_expected = offsetted_operand.view(np.ndarray)
                            right_actual = offsetted_operand
                            right_slices = offsetted_slices

                            sub_slices = fixed_slices
                        else:
                            left_expected = offsetted_operand.view(np.ndarray).copy()
                            left_actual = offsetted_operand
                            left_slices = offsetted_slices

                            right_expected = original.view(np.ndarray)
                            right_actual = fixed_operand
                            right_slices = fixed_slices

                            sub_slices = offsetted_slices

                        expected_result = left_expected

                        left_slices = no_wrap_slices(left_slices)
                        right_slices = no_wrap_slices(right_slices)
                        sub_slices = no_wrap_slices(sub_slices)

                        # Condition to autofail on disjoint
                        if not fixed_in_offsetted and not offsetted_in_fixed and shape[0] > length * 2:
                            with pytest.raises(ValueError):
                                op(left_actual, right_actual)
                            continue

                        interim_result = op(left_expected[left_slices], right_expected[right_slices])
                        expected_result = expected_result.astype(interim_result.dtype)
                        expected_result[sub_slices] = interim_result

                        try:
                            actual_result = op(left_actual, right_actual)
                        except Exception as e:
                            actual_error = e

                        assert expected_error is None == actual_error is None, \
                            'Expected error: (%s) Actual error: (%s)' % (expected_error, actual_error) # noqa E711

                        # ensure global_offset is preserved
                        assert hasattr(actual_result, 'global_offset')
                        assert left_actual.global_offset == actual_result.global_offset

                        # ensure actual results match that of a regular ndarray
                        assert np.array_equal(expected_result, actual_result)

                        # ensure that in place modifies the actual result
                        actual_result[actual_result.global_offset] = 1337
                        assert 1337 == left_actual[actual_result.global_offset]

    def test_aggregate_function(self):
        for test_array in TEST_ARRAYS:
            offset_array = GlobalOffsetArray(test_array)
            assert type(offset_array.sum()) == type(test_array.sum()) #noqa

    def test_slice_same_dimensions(self):
        for test_array in TEST_ARRAYS:
            sub_slices = tuple(slice(0, dim // 2) for dim in test_array.shape)

            offset_array = GlobalOffsetArray(test_array)

            sub_test_array = test_array[sub_slices]
            sub_offset_array = offset_array[sub_slices]

            assert sub_offset_array.shape == sub_test_array.shape
            assert len(sub_offset_array.global_offset) == len(sub_test_array.shape)

    def test_slice_fill_missing_dimensions(self):
        for test_array in TEST_ARRAYS[:]:
            sub_slices = tuple(slice(0, dim // 2) for dim in test_array.shape[1:])

            offset_array = 0 + GlobalOffsetArray(test_array)
            sub_offset_array = 0 + offset_array[sub_slices]
            assert sub_offset_array.global_offset == offset_array.global_offset

    def test_autofill_dimensions(self):
        dimensions = (4, 3, 2, 1)
        data = np.arange(0, np.product(dimensions)).reshape(dimensions)
        global_offset_data = GlobalOffsetArray(data, global_offset=(3, 2, 1, 0))
        assert np.array_equal(global_offset_data[5], data[2])
        assert np.array_equal(global_offset_data[5, 3], data[2, 1])

    def test_pickle(self):
        dimensions = (4, 3, 2, 1)
        data = np.arange(0, np.product(dimensions)).reshape(dimensions)
        global_offset = (3, 2, 1, 0)
        global_offset_data = GlobalOffsetArray(data, global_offset=global_offset)

        pickled = pickle.dumps(global_offset_data)
        unpickled = pickle.loads(pickled)

        assert global_offset_data.data is not unpickled
        assert global_offset == unpickled.global_offset
        assert np.array_equal(global_offset_data.data, unpickled.data)

# pylama:ignore=C901
