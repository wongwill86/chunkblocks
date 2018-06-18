from collections import deque

from chunkblocks import iterators


class TestUnitIterator:
    def test_unit_get_all_neighbors(self):
        iterator = iterators.UnitIterator()
        expected_neighbors = set([
            (0, 1),
            (1, 0),
            (1, 1),
            (0, -1),
            (-1, 0),
            (-1, -1),
            (-1, 1),
            (1, -1),
        ])
        neighbors = iterator.get_all_neighbors((0, 0))
        assert set(neighbors) == expected_neighbors

    def test_unit_get_all_neighbors_max(self):
        iterator = iterators.UnitIterator()
        expected_neighbors = set([
            (0, 1),
            (1, 0),
            (0, 0),
        ])
        neighbors = iterator.get_all_neighbors((1, 1), max=(2, 2))
        assert set(neighbors) == expected_neighbors


class TestUnitBFSIterator:

    def test_bfs_iterator(self):
        expected_bfs_steps = [
            set([(0, 0)]),
            set([(0, 1), (1, 0), (1, 1), ]),
            set([(0, 2), (1, 2), (2, 2), (2, 1), (2, 0), ])
        ]
        iterator = iterators.UnitBFSIterator()
        bfs = deque(iterator.get((0, 0), (3, 3)))
        assert len(bfs) == 9
        for expected_bfs in expected_bfs_steps:
            while len(expected_bfs) and len(bfs):
                step = bfs.popleft()
                expected_bfs.remove(step)
            assert len(expected_bfs) == 0
        assert len(bfs) == 0
