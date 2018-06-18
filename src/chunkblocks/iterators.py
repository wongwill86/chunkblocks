import itertools
from collections import deque

all_neighbor_offsets = [-1, 0, 1]


class Iterator(object):
    def get_all_neighbors(self, index, max=None):
        raise NotImplementedError

    def iterator(self, index):
        raise NotImplementedError


class UnitIterator(Iterator):
    def __init__(self):
        pass

    def get_all_neighbors(self, index, max=None):
        """
        Return all neighbors including diagonals
        """
        neighbors = []
        origin = tuple([0]*len(index))
        for offset in itertools.product(*[all_neighbor_offsets] * len(index)):
            neighbor = tuple(offset[dim] + index[dim] for dim in range(0, len(index)))
            if offset != origin and (not max or all(idx >= 0 and idx < m for idx, m in zip(neighbor, max))):
                neighbors.append(neighbor)
        return neighbors


class UnitBFSIterator(UnitIterator):
    def get(self, start, dimensions):
        """
        Get an iterator that traverses dataset in single units using Breadth-First-Search fashion.
        Units start at 0 and end at param: dimensions
        :param start: location to start iteration from
        :param dimensions: list or tuple of units for each dimension
        """
        queue = deque()
        queue.append(start)
        visited = set((start,))

        def inside_bounds(index):
            return not any(idx < 0 or idx >= dimension for idx, dimension in zip(index, dimensions))

        while(len(queue) > 0):
            internal_index = queue.popleft()
            neighbors = [
                neighbor for neighbor in self.get_all_neighbors(internal_index)
                if inside_bounds(neighbor) and neighbor not in visited
            ]
            visited.update(neighbors)
            queue.extend(neighbors)
            yield internal_index
