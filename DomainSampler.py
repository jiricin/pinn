import numpy as np
import matplotlib.pyplot as plt


class DomainSampler:
    def __init__(self, ordered_points):
        # ordered edge points at the domain border, defining a domain,
        # stored as two arrays contained in one: [[X], [Y]]
        # in example, [[0,1,0], [0,1,2]] defines a triangle with vertices [0,0], [1,1], [0,2]
        self.points = ordered_points

        self.point_count = len(self.points[1])
        self.min_y = np.min(self.points[1])
        self.max_y = np.max(self.points[1])

    def sample_line_prod(self):
        # get a random y value for sample
        y = np.random.uniform(low=self.min_y, high=self.max_y)
        while self.points[1][-1] == y:
            np.roll(self.points, 1, axis=1)  # for algorithm purposes

        # we need to check for intersections between line Y=y and domain boundary
        ordered_intersections = []

        # indices for iterating over points and checking, if there was an intersection between them
        idx_prev = -1
        idx = 0
        while idx < self.point_count:
            # checking differences between point y-values and line Y=y
            sgn_prev = self.points[1][idx_prev] - y
            sgn = self.points[1][idx] - y

            if sgn_prev * sgn < 0:  # True = intersection somewhere between points
                # intersection point position calculation
                x_is = self.points[0][idx_prev] - sgn_prev * \
                       (self.points[0][idx] - self.points[0][idx_prev]) / \
                       (self.points[1][idx] - self.points[1][idx_prev])
                ordered_intersections.append([x_is, x_is, True])  # boolean: does crossing it change domain inclusion?

            if sgn == 0:  # True = intersection at an edge point
                x1_is = self.points[0][idx]  # store the position of the first intersection
                while sgn == 0:  # get the next point, which does not share the same y-value
                    idx += 1
                    sgn = self.points[1][idx] - y
                x2_is = self.points[0][idx - 1]  # store the position of the last intersection
                ordered_intersections.append([x1_is, x2_is, np.sign(sgn_prev) - np.sign(sgn)])

            idx_prev = idx
            idx += 1

        # sort the intersections by increasing x-values
        ordered_intersections.sort(key=lambda seg: (seg[0] + seg[1]) / 2)

        # create intervals from intersections
        ordered_intervals = []
        inclusion = False  # by going from left to right, inclusion changes
        for iseg in ordered_intersections:
            if iseg[2]:  # True = crossing changes domain inclusion
                inclusion = ~inclusion
                ordered_intervals.append(iseg[1] if inclusion else iseg[0])  # assign appropriate interval border
            elif inclusion:  # True = no inclusion change; if we are in the domain, we want to exclude the boundary
                ordered_intervals.append(iseg[0])
                ordered_intervals.append(iseg[1])

        # generate random x-value from computed y-valued intervals in domain
        r = np.random.random_sample()
        int_idx = int(r * len(ordered_intervals) / 2) * 2  # interval index
        int_pos = np.random.random_sample()  # position in interval
        x_min = ordered_intervals[int_idx]
        x_max = ordered_intervals[int_idx + 1]
        x = x_min + (x_max - x_min) * int_pos

        # return sampled point
        return [x, y]


if __name__ == '__main__':
    # Pacman
    ds = DomainSampler([[0, -1, -2, -1, 0, 1, 1, 0, -1, -2, -1],
                        [0, 1, 1, 2, 2, 1, -1, -2, -2, -1, -1]])
    plt.plot(ds.points[0], ds.points[1], 'b')
    plt.plot([ds.points[0][-1], ds.points[0][0]], [ds.points[1][-1], ds.points[1][0]], 'b')

    for i in range(500):
        sample = ds.sample_line_prod()
        plt.plot(sample[0], sample[1], 'r+')

    plt.show()

# to do
# - if points are on a horizontal line, algorithm gets stuck on line 19 (possibly 42, too)
# - sorting complexity on line 52 can be alleviated by bisection in-sorting in element adding phase
#
# - clockwise-turning-degree-sum-based check and sampling
