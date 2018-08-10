import itertools

import numpy as np
import pytest

from chunkblocks.global_offset_array import GlobalOffsetArray
from chunkblocks.iterators import Iterator
from chunkblocks.models import Block, Chunk


class IdentityIterator(Iterator):
    def get_all_neighbors(self, index, max=None):
        return index

    def get(self, start, dimensions):
        yield start


class TestChunk:
    def test_get_border_slices_2d(self):
        bounds = (slice(0, 50), slice(0, 50))
        chunk_shape = (30, 30)
        overlap = (10, 10)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        chunk = Chunk(block, (0, 0))

        borders = list(itertools.product(range(0, len(bounds)), [-1, 1]))

        fake_data = np.zeros(chunk.shape)
        for slices in chunk.border_slices(borders):
            fake_data[slices] += 1

        fake_data[chunk.core_slices(borders)] += 1
        assert fake_data.sum() == np.product(fake_data.shape)

    def test_get_border_slices_3d(self):
        bounds = (slice(0, 70), slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30, 30)
        overlap = (10, 10, 10)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        chunk = Chunk(block, (0, 0, 0))

        borders = list(itertools.product(range(0, len(bounds)), [-1, 1]))

        fake_data = np.zeros(chunk.shape)
        for slices in chunk.border_slices(borders):
            fake_data[slices] += 1

        fake_data[chunk.core_slices(borders)] += 1
        assert fake_data.sum() == np.product(fake_data.shape)

    def test_get_border_slices_3d_overlapping(self):
        bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        chunk = Chunk(block, (0, 0, 0))

        borders = list(itertools.product(range(0, len(bounds)), [-1, 1]))

        fake_data = np.zeros(chunk.shape)
        for slices in chunk.border_slices(borders, nonintersecting=False):
            fake_data[slices] += 1

        fake_data[chunk.core_slices(borders)] += 1
        assert np.array_equal(fake_data, [[[3, 2, 3],
                                           [2, 1, 2],
                                           [3, 2, 3]],
                                          [[2, 1, 2],
                                           [1, 1, 1],
                                           [2, 1, 2]],
                                          [[3, 2, 3],
                                           [2, 1, 2],
                                           [3, 2, 3]]])


class TestBlock:
    def test_init_wrong_size_no_overlap(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30)

        with pytest.raises(ValueError):
            Block(bounds=bounds, chunk_shape=chunk_shape)

    def test_init(self):
        bounds = (slice(0, 70), slice(0, 70))
        offset = (0, 0)
        num_chunks = (3, 3)
        overlap = (10, 10)
        chunk_shape = (30, 30)

        # test with bounds
        Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        # test with offset/num_chunks
        Block(offset=offset, num_chunks=num_chunks, chunk_shape=chunk_shape, overlap=overlap)

        # test with both offset/num_chunks
        Block(bounds=bounds, offset=offset, num_chunks=num_chunks, chunk_shape=chunk_shape, overlap=overlap)

        # test fail with neither block and offset offset/num_chunks
        with pytest.raises(ValueError):
            Block(chunk_shape=chunk_shape, overlap=overlap)

        # test fail with only offset no num_chunks
        with pytest.raises(ValueError):
            Block(offset=offset, chunk_shape=chunk_shape, overlap=overlap)

        # test fail with only num_chuks no offset
        with pytest.raises(ValueError):
            Block(num_chunks=num_chunks, chunk_shape=chunk_shape, overlap=overlap)

        # test incorrect matching bounds with offset/num_chunks
        with pytest.raises(Exception):
            Block(bounds=(slice(b.start, b.stop + 1) for b in bounds),
                  offset=offset, num_chunks=num_chunks, chunk_shape=chunk_shape, overlap=overlap)

    def test_init_wrong_size_overlap(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30)

        with pytest.raises(ValueError):
            Block(bounds=bounds, chunk_shape=chunk_shape)

    def test_index_to_slices(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30)
        overlap = (10, 10)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        assert block.unit_index_to_slices((0, 0)) == (slice(0, 30), slice(0, 30))
        assert block.unit_index_to_slices((0, 1)) == (slice(0, 30), slice(20, 50))
        assert block.unit_index_to_slices((1, 0)) == (slice(20, 50), slice(0, 30))

    def test_slices_to_index(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30)
        overlap = (10, 10)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        assert block.chunk_slices_to_unit_index((slice(0, 30), slice(0, 30))) == (0, 0)
        assert block.chunk_slices_to_unit_index((slice(0, 30), slice(20, 50))) == (0, 1)
        assert block.chunk_slices_to_unit_index((slice(20, 50), slice(0, 30))) == (1, 0)
        assert block.chunk_slices_to_unit_index((slice(20, 50), slice(20, 50))) == (1, 1)

    def test_iterator(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30)
        overlap = (10, 10)

        start = (0, 0)
        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap, base_iterator=IdentityIterator())

        chunks = list(block.chunk_iterator(start))
        assert len(chunks) == 1
        assert chunks[0].unit_index == start

    def test_get_slices_2d(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0, 0))
        assert block.num_chunks == (3, 3)

        for chunk in block.chunk_iterator((0, 0)):
            for edge_slice in block.overlap_slices(chunk):
                fake_data[edge_slice] += 1
            fake_data[block.core_slices(chunk)] += 1
        assert fake_data.sum() == np.product(fake_data.shape)

    def test_overlap_slices_3d(self):
        bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        assert block.num_chunks == (3, 3, 3)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0, 0, 0))
        for chunk in block.chunk_iterator((1, 0, 1)):
            for edge_slice in block.overlap_slices(chunk):
                fake_data[edge_slice] += 1
            fake_data[block.core_slices(chunk)] += 1
        assert fake_data.sum() == np.product(fake_data.shape)

    def test_checkpoints(self):
        bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        for chunk in block.chunk_iterator((1, 0, 1)):
            block.checkpoint(chunk)
            assert block.is_checkpointed(chunk)
            assert block.is_checkpointed(chunk, stage=0)

        for chunk in block.chunk_iterator((1, 0, 1)):
            assert not block.is_checkpointed(chunk, stage=1)
            assert not block.checkpoint(chunk, stage=1)
            assert block.all_neighbors_checkpointed(chunk, stage=0)
            block.checkpoint(chunk, stage=1)

        stage = 0
        for chunk in block.chunk_iterator((1, 0, 1)):
            print(block.checkpoints[stage][chunk.unit_index])
            for c in block.get_all_neighbors(chunk):
                print(c.unit_index, block.checkpoints[stage][c.unit_index])
            assert block.all_neighbors_checkpointed(chunk, stage=0)

    def test_slices_to_indices(self):
        bounds_1 = (slice(0, 16), slice(0, 16), slice(0, 16))
        chunk_shape_1 = (4, 4, 4)
        overlap_1 = (1, 1, 1)
        block_1 = Block(bounds=bounds_1, chunk_shape=chunk_shape_1, overlap=overlap_1)

        bounds_2 = (slice(-1, 25), slice(-1, 25), slice(-1, 25))
        chunk_shape_2 = (6, 6, 6)
        overlap_2 = (1, 1, 1)
        block_2 = Block(bounds=bounds_2, chunk_shape=chunk_shape_2, overlap=overlap_2)

        index = 1
        for unit_index in range(0, block_2.num_chunks[index]):
            chunk_2 = Chunk(block_2, (0, unit_index))
            chunk_2_coords = set(filter(lambda x: x >= block_1.bounds[index].start and x < block_1.bounds[index].stop,
                                        range(chunk_2.slices[index].start, chunk_2.slices[index].stop)))
            print('expect:', chunk_2.slices, chunk_2_coords)
            for unit_index in block_1.slices_to_unit_indices(chunk_2.slices):
                chunk_1 = Chunk(block_1, unit_index)
                chunk_1_coords = set(filter(lambda x: x >= block_1.bounds[index].start and x < block_1.bounds[index].stop,
                                            range(chunk_1.slices[index].start, chunk_1.slices[index].stop)))
                print(chunk_1.slices, chunk_1_coords)
                chunk_2_coords.difference_update(chunk_1_coords)
                assert all(tuple(u >= 0 and u <= n for u, n in zip(unit_index, block_1.num_chunks)))
            print('left', chunk_2_coords)
            assert len(chunk_2_coords) == 0

        # Test reverse direction
        block_2_temp = block_2
        block_2 = block_1
        block_1 = block_2_temp

        index = 1
        for unit_index in range(0, block_2.num_chunks[index]):
            chunk_2 = Chunk(block_2, (0, unit_index))
            chunk_2_coords = set(filter(lambda x: x >= block_1.bounds[index].start and x < block_1.bounds[index].stop,
                                        range(chunk_2.slices[index].start, chunk_2.slices[index].stop)))
            print('expect:', chunk_2.slices, chunk_2_coords)
            for unit_index in block_1.slices_to_unit_indices(chunk_2.slices):
                chunk_1 = Chunk(block_1, unit_index)
                chunk_1_coords = set(filter(lambda x: x >= block_1.bounds[index].start and x < block_1.bounds[index].stop,
                                            range(chunk_1.slices[index].start, chunk_1.slices[index].stop)))
                print(chunk_1.slices, chunk_1_coords)
                chunk_2_coords.difference_update(chunk_1_coords)
                assert all(tuple(u >= 0 and u <= n for u, n in zip(unit_index, block_1.num_chunks)))
            print('left', chunk_2_coords)
            assert len(chunk_2_coords) == 0

        # Test None
        index = 1
        for unit_index in range(0, block_2.num_chunks[index]):
            chunk_2 = Chunk(block_2, (0, unit_index))
            # use fake slices with None here!
            chunk_2_slices = (slice(None, None),) + chunk_2.slices[1:]
            chunk_2_coords = set(filter(lambda x: x >= block_1.bounds[index].start and x < block_1.bounds[index].stop,
                                        range(chunk_2_slices[index].start, chunk_2_slices[index].stop)))
            print('expect:', chunk_2_slices, chunk_2_coords)
            for unit_index in block_1.slices_to_unit_indices(chunk_2_slices):
                chunk_1 = Chunk(block_1, unit_index)
                chunk_1_coords = set(filter(lambda x: x >= block_1.bounds[index].start and x < block_1.bounds[index].stop,
                                            range(chunk_1.slices[index].start, chunk_1.slices[index].stop)))
                print(chunk_1.slices, chunk_1_coords)
                chunk_2_coords.difference_update(chunk_1_coords)
                assert all(tuple(u >= 0 and u <= n for u, n in zip(unit_index, block_1.num_chunks)))
            print('left', chunk_2_coords)
            assert len(chunk_2_coords) == 0
