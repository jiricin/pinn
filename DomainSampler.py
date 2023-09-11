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

    # SAMPLE_LINE_PROBE: Pre-samples domain using line probes
    def sample_line_probe(self, n_y=50, n_x=50):
        self.samples = [[], []]

        # uniformly generated lines
        for y in np.linspace(self.min_y, self.max_y, n_y):
            idx_shift = 0
            while self.points[1][idx_shift-1] == y:
                idx_shift = idx_shift + 1
                if idx_shift > self.point_count:
                    print('Domain is a straight horizontal line, not allowed')
                    return

            # find intersections with domain boundary
            ordered_intersections = []

            idx_prev = idx_shift - 1
            idx = idx_shift
            point_count_shifted = self.point_count + idx_shift
            while idx < point_count_shifted:
                sgn_prev = self.points[1][idx_prev] - y
                sgn = self.points[1][idx] - y

                if sgn_prev * sgn < 0:  # intersection somewhere between points
                    x_is = self.points[0][idx_prev] - sgn_prev * \
                           (self.points[0][idx] - self.points[0][idx_prev]) / \
                           (self.points[1][idx] - self.points[1][idx_prev])
                    ordered_intersections.append([x_is, x_is, True])

                if sgn == 0:  # intersection at an edge point
                    x1_is = self.points[0][idx]
                    while sgn == 0:
                        idx += 1
                        sgn = self.points[1][idx] - y
                    x2_is = self.points[0][idx - 1]
                    ordered_intersections.append([x1_is, x2_is, np.sign(sgn_prev) - np.sign(sgn)])

                idx_prev = idx
                idx += 1

            ordered_intersections.sort(key=lambda seg: (seg[0] + seg[1]) / 2)

            # create intervals from intersections
            ordered_intervals = []
            inclusion = False
            for iseg in ordered_intersections:
                if iseg[2]:
                    inclusion = ~inclusion
                    ordered_intervals.append(iseg[1] if inclusion else iseg[0])
                elif inclusion:
                    ordered_intervals.append(iseg[0])
                    ordered_intervals.append(iseg[1])

            # sample x-values from intervals
            x = np.linspace(self.min_x, self.max_x, n_x)
            ordered_intervals = (ordered_intervals - self.min_x) * (n_x - 1) / (self.max_x - self.min_x)
            int_idx = 0
            while int_idx < len(ordered_intervals):
                idx1 = math.ceil(ordered_intervals[int_idx])
                idx2 = math.floor(ordered_intervals[int_idx + 1])
                idx_count = idx2 - idx1 + 1
                self.samples[0].extend(x[idx1:(idx2+1)])
                self.samples[1].extend(np.full(idx_count, y))
                int_idx = int_idx + 2
        return

    # TURN_CHECK: Checks if chosen point lies in the domain using degree-sum
    def degree_check(self, x, y):
        deg_sum = 0.0
        for idx in range(self.point_count):
            p1 = self.points[0][idx - 1] - x
            p2 = self.points[1][idx - 1] - y
            n1 = self.points[0][idx] - x
            n2 = self.points[1][idx] - y
            pn = p1*n1+p2*n2
            pp = p1*p1+p2*p2
            nn = n1*n1+n2*n2
            if pp == 0 or nn == 0:
                return False  # points which define the domain are excluded

            product = pn / (math.sqrt(pp) * math.sqrt(nn))
            if abs(product) > 1.0:
                return False  # for rounding errors

            deg_sum = deg_sum + np.sign(p2 * (n1 - p1 * pn / pp)) * math.acos(product)
        return ~(deg_sum < 5.0)

    # SAMPLE_DEGREE_CHECK: Pre-samples domain using degree-sum check
    def sample_degree_check(self, n_x=50, n_y=50):
        self.samples = [[], []]

        for x in np.linspace(self.min_x, self.max_x, n_x):
            for y in np.linspace(self.min_y, self.max_y, n_y):
                if self.degree_check(x, y):
                    self.samples[0].append(x)
                    self.samples[1].append(y)
        return

    # (incomplete) SAMPLE_DEGREE_CHECK_HIERARCHICAL: Pre-samples domain using degree-sum check with square hierarchy
    def sample_degree_check_hierarchical(self, n_x=10, n_y=10):
        self.samples = [[], []]

        lx = np.linspace(self.min_x, self.max_x, n_x)
        ly = np.linspace(self.min_y, self.max_y, n_y)
        dw = (self.max_x - self.min_x) / n_x
        dh = (self.max_y - self.min_y) / n_y

        grid_points = [[a, b] for a in ly for b in lx]
        inclusions = [self.degree_check(grid_points[idx][0], grid_points[idx][1]) for idx in range(n_x * n_y)]
        squares = [Square() for a in range((n_x-1)*(n_y-1))]

        for y_idx in range(n_y-1):
            for x_idx in range(n_x-1):
                idx = x_idx + y_idx * n_x

                top_right = x_idx + y_idx * (n_x-1)
                top_left = top_right - 1
                bottom_right = top_right - n_x + 1
                bottom_left = top_right - n_x

                grid_points[x_idx + y_idx * n_x].append(bottom_left)
                grid_points[x_idx+1 + y_idx * n_x].append(bottom_right)
                grid_points[x_idx + (y_idx+1) * n_x].append(top_left)
                grid_points[x_idx+1 + (y_idx+1) * n_x].append(top_right)

                squares[bottom_left].top_right = idx
                squares[bottom_right].top_left = idx
                squares[top_left].bottom_right = idx
                squares[top_right].bottom_left = idx

        dw = dw / 2
        dh = dh / 2

        for square in squares:
            if grid_points[square.top_right] and grid_points[square.top_left] and grid_points[square.bottom_right] and grid_points[square.bottom_left]:
                square.included = True
            else:
                square = [Square(),
                          Square(),
                          Square(),
                          Square()]

        return

    # PLOT: Plots generated samples
    def plot(self):
        plt.plot(self.samples[0], self.samples[1], 'g+')

    # RANDOM_SAMPLE: Chooses one random sample from pre-sampled domain
    def random_sample(self):
        index = np.random.randint(0, len(self.samples[0]))
        return [self.samples[0][index], self.samples[1][index]]


class Square:
    def __init__(self):
        self.bottom_left = 0
        self.bottom_right = 0
        self.top_left = 0
        self.top_right = 0


if __name__ == '__main__':
    # Pacman domain
    ds = DomainSampler([[0, -1, -2, -1, 0, 1, 1, 0, -1, -2, -1],
                        [0, 1, 1, 2, 2, 1, -1, -2, -2, -1, -1]])
    plt.plot(ds.points[0], ds.points[1], 'b')
    plt.plot([ds.points[0][-1], ds.points[0][0]], [ds.points[1][-1], ds.points[1][0]], 'b')

    # ds.sample_line_probe()
    ds.sample_degree_check()

    # ds.plot()

    for counter in range(25):
        r = ds.random_sample()
        plt.plot(r[0], r[1], 'r+')

    plt.show()
