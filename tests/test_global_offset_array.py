import operator
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
        Make sure error is thrown when trying to access out of bounds

        """
        original = np.arange(5 ** 4).reshape(tuple([5] * 4))
        global_offset = (100, 200, 300, 400)
        offset_array = GlobalOffsetArray(original)

        with pytest.raises(IndexError):
            offset_array[5:6, 4:5, 6:7, 7:8]

        with pytest.raises(IndexError):
            offset_array[2, 2:5, 6:8, 0:2]

        with pytest.raises(IndexError):
            offset_array[2, 9, 0:3, 0:2]

        offset_array = GlobalOffsetArray(original, global_offset=global_offset)

        with pytest.raises(IndexError):
            offset_array[105:106, 204:205, 306:307, 407:408]

        with pytest.raises(IndexError):
            offset_array[102, 202:205, 306:308, 400:402]

        with pytest.raises(IndexError):
            offset_array[102, 209, 300:303, 400:402]

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

    def test_operator_same_size_global_offset_array(self):
        """
        Test that when using operators on two GlobalOffsetArray it works only when they have the same global_offset
        """
        ndim = 5
        length = 4

        for op in STANDARD_OPERATORS | IN_PLACE_OPERATORS:
            for forward in [True, False]:
                (original, offset_array) = self.generate_data(ndim, length)
                operate_param = GlobalOffsetArray(np.ones(offset_array.shape, dtype=offset_array.dtype) * 10,
                                                  global_offset=offset_array.global_offset)
                # itrue div requires floats when doing true division (can't do in place conversion to float)
                if op == operator.itruediv:
                    original = original.astype(np.float64)
                    offset_array = offset_array.astype(np.float64)
                    operate_param = operate_param.astype(np.float64)

                # Make sure to compare expected results as a ndarray because operate_param is a GlobalOffsetArray.
                if forward:
                    left_expected = original.view(np.ndarray)
                    right_expected = operate_param.view(np.ndarray)
                    left_offset = offset_array
                    right_offset = operate_param
                else:
                    # test operation commutativity
                    left_expected = operate_param.view(np.ndarray)
                    right_expected = original.view(np.ndarray)
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

                # ensure the results that are returned are a copy of an array instead of a view just like ndarray
                expected[tuple([0] * ndim)] = 1337
                actual[actual.global_offset] = 1337

                # original arrays were not modified
                assert np.any(offset_array == 1337) == np.any(original == 1337)

                # Try testing with the operate param with a different global_offset
                operate_param.global_offset = tuple([1337] * ndim)
                with pytest.raises(ValueError):
                    op(left_offset, right_offset)

                # Try testing with the operate param with a partially overlapping data
                operate_param.global_offset = tuple(floor(size/2) + offset for size, offset in
                                                    zip(offset_array.shape, offset_array.global_offset))
                with pytest.raises(ValueError):
                    op(left_offset, right_offset)

    def test_operator_diff_size_global_offset_array(self):
        """
        Test to make sure operators return a copy of the original array
        """
        ndim = 5
        length = 4

        for op in STANDARD_OPERATORS | IN_PLACE_OPERATORS:
            for forward in [True, False]:
                (original, offset_array) = self.generate_data(ndim, length)
                half_size = tuple(floor(size/2) for size in offset_array.shape)
                offset = tuple(half_size[dimension] + offset for dimension, offset in
                               enumerate(offset_array.global_offset))
                half_slice = tuple(slice(s, s + s) for s in half_size)
                operate_param = GlobalOffsetArray(np.ones(half_size, dtype=offset_array.dtype) * 255,
                                                  global_offset=offset)
                # itrue div requires floats when doing true division (can't do in place conversion to float)
                if op == operator.itruediv:
                    original = original.astype(np.float64)
                    offset_array = offset_array.astype(np.float64)
                    operate_param = operate_param.astype(np.float64)

                if forward:
                    left_expected = original[half_slice]
                    right_expected = operate_param.view(np.ndarray)
                    left_offset = offset_array
                    right_offset = operate_param
                else:
                    # test operation commutativity
                    left_expected = operate_param.view(np.ndarray)
                    right_expected = original[half_slice]
                    left_offset = operate_param
                    right_offset = offset_array

                    # in place operators should only work if the left param is larger than the right
                    if op in IN_PLACE_OPERATORS:
                        with pytest.raises(ValueError):
                            op(left_offset, right_offset)
                        continue

                actual_result = op(left_offset, right_offset)

                expected_sub_array = op(left_expected, right_expected)
                if op in STANDARD_OPERATORS:
                    expected = original.astype(type(expected_sub_array.item(0)))
                    actual = actual_result
                else:
                    # in place operations modify originals
                    expected = original
                    actual = offset_array

                # Simulate expected
                expected[half_slice] = expected_sub_array

                # ensure global_offset is preserved
                assert actual.global_offset == offset_array.global_offset

                assert np.array_equal(actual, expected)

                # ensure the results that are returned are a copy of an array instead of a view just like ndarray
                expected[tuple([0] * ndim)] = 1337
                actual[actual.global_offset] = 1337

                # original arrays were not modified
                assert np.any(original == 1337) == np.any(offset_array == 1337)

                # Fail on partial overlap
                with pytest.raises(ValueError):
                    operate_param.global_offset = tuple(floor(size/2) + floor(size/4) for size in offset_array.shape)
                    op(left_offset, right_offset)

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
