import math

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
        self.min_x = np.min(self.points[0])
        self.max_x = np.max(self.points[0])

        self.samples = [[], []]

    def random_line_prod(self):
        # get a random y value for sample
        y = np.random.uniform(low=self.min_y, high=self.max_y)
        idx_shift = 0
        while self.points[1][idx_shift-1] == y:
            idx_shift = idx_shift + 1  # for algorithm purposes

        # we need to check for intersections between line Y=y and domain boundary
        ordered_intersections = []

        # indices for iterating over points and checking, if there was an intersection between them
        idx_prev = idx_shift - 1
        idx = idx_shift
        point_count_shifted = self.point_count + idx_shift
        while idx < point_count_shifted:
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

    def sample_line_prod(self):
        self.samples = [[], []]

        # for a list of y-values
        n_y = 50
        for y in np.linspace(self.min_y, self.max_y, n_y):
            idx_shift = 0
            while self.points[1][idx_shift-1] == y:
                idx_shift = idx_shift + 1  # for algorithm purposes

            # we need to check for intersections between line Y=y and domain boundary
            ordered_intersections = []

            # indices for iterating over points and checking, if there was an intersection between them
            idx_prev = idx_shift - 1
            idx = idx_shift
            point_count_shifted = self.point_count + idx_shift
            while idx < point_count_shifted:
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

            # compute x-values for y-valued intervals in domain
            n_x = 50
            x = np.linspace(self.min_x, self.max_x, n_x)
            # intervals of indexes for array x
            ordered_intervals = (ordered_intervals - self.min_x) * (n_x - 1) / (self.max_x - self.min_x)
            int_idx = 0
            while int_idx < len(ordered_intervals):
                idx1 = math.ceil(ordered_intervals[int_idx])
                idx2 = math.floor(ordered_intervals[int_idx + 1])
                idx_count = idx2 - idx1 + 1
                self.samples[0].extend(x[idx1:(idx2+1)])
                self.samples[1].extend(np.full(idx_count, y))
                int_idx = int_idx + 2

            # repeat for all y

        return

    def turn_check(self, x, y):
        deg_sum = 0.0
        for idx in range(self.point_count):
            p1 = self.points[0][idx - 1] - x
            p2 = self.points[1][idx - 1] - y
            n1 = self.points[0][idx] - x
            n2 = self.points[1][idx] - y
            pn = p1*n1+p2*n2
            pp = p1*p1+p2*p2
            nn = n1*n1+n2*n2
            deg_sum = deg_sum + np.sign(p2 * (n1 - p1 * pn / pp)) * math.acos(pn / (math.sqrt(pp) * math.sqrt(nn)))

        return ~(deg_sum < 5.0)

    def plot(self):
        plt.plot(self.samples[0], self.samples[1], 'g+')


if __name__ == '__main__':
    # Pacman domain
    ds = DomainSampler([[0, -1, -2, -1, 0, 1, 1, 0, -1, -2, -1],
                        [0, 1, 1, 2, 2, 1, -1, -2, -2, -1, -1]])
    plt.plot(ds.points[0], ds.points[1], 'b')
    plt.plot([ds.points[0][-1], ds.points[0][0]], [ds.points[1][-1], ds.points[1][0]], 'b')

    ds.sample_line_prod()
    ds.plot()

    plt.show()
