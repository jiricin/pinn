import math

import numpy as np
import matplotlib.pyplot as plt


class Square:
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y


class LineProbe:
    def __init__(self, y, min_x, max_x):
        self.y = y
        self.min_x = min_x
        self.max_x = max_x


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
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y

        self.line_probes = []
        self.line_probes_distribution = []

        self.squares = []
        self.squares_counts = []
        self.squares_distribution = []

        self.last_sampling = 'none'

    # SAMPLE_LINE_PROBE: Creates domain distribution using horizontal line probes
    def sample_line_probe(self, n_y=20, nudge_ratio_y=0.01):
        self.line_probes = []
        self.line_probes_distribution = []

        # uniformly generated lines
        for y in np.linspace(self.min_y + self.height * nudge_ratio_y, self.max_y - self.height * nudge_ratio_y, n_y):
            idx_shift = 0
            while self.points[1][idx_shift-1] == y:
                idx_shift = idx_shift + 1
                if idx_shift > self.point_count:
                    return

            # find intersections with domain boundary
            ordered_intersections = []

            idx = 0
            while idx < self.point_count:
                idx_shifted = (idx + idx_shift) % self.point_count
                idx_prev = idx_shifted - 1
                sgn_prev = self.points[1][idx_prev] - y
                sgn = self.points[1][idx_shifted] - y

                if sgn_prev * sgn < 0:  # intersection somewhere between points
                    x_is = self.points[0][idx_prev] - sgn_prev * \
                           (self.points[0][idx_shifted] - self.points[0][idx_prev]) / \
                           (self.points[1][idx_shifted] - self.points[1][idx_prev])
                    ordered_intersections.append([x_is, x_is, True])

                if sgn == 0:  # intersection at an edge point
                    x1_is = self.points[0][idx_shifted]
                    while sgn == 0:
                        idx += 1
                        idx_shifted = (idx + idx_shift) % self.point_count
                        sgn = self.points[1][idx_shifted] - y
                    x2_is = self.points[0][idx_shifted - 1]
                    ordered_intersections.append([x1_is, x2_is, np.sign(sgn_prev) - np.sign(sgn)])

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

            # generate line probes and their distribution
            for idx in range(0, len(ordered_intervals), 2):
                self.line_probes.append(LineProbe(y, ordered_intervals[idx], ordered_intervals[idx+1]))
                self.line_probes_distribution.append(ordered_intervals[idx+1] - ordered_intervals[idx])

        lengths_sum = np.sum(self.line_probes_distribution)
        if lengths_sum > 0:
            self.line_probes_distribution = self.line_probes_distribution / lengths_sum
        self.last_sampling = 'line_probe'
        return

    # DEGREE_CHECK: Checks if chosen point lies in the domain using degree-sum
    def degree_check(self, x, y):
        deg_sum = 0.0
        for idx in range(self.point_count):
            p1 = self.points[0][idx - 1] - x
            p2 = self.points[1][idx - 1] - y
            n1 = self.points[0][idx] - x
            n2 = self.points[1][idx] - y
            pp = p1*p1+p2*p2
            nn = n1*n1+n2*n2
            if pp == 0 or nn == 0:
                return False
            pn_dot_norm = np.clip((p1 * n1 + p2 * n2) / (math.sqrt(pp) * math.sqrt(nn)), -1.0, 1.0)
            pn_cross = p2 * n1 - p1 * n2
            deg_sum = deg_sum + np.sign(pn_cross) * math.acos(pn_dot_norm)
        return (abs(deg_sum) - 2 * np.pi)**2 < 1.0e-5

    # SAMPLE_DEGREE_CHECK_HIERARCHICAL: Creates domain distribution using degree-sum check
    def sample_degree_check_hierarchical(self, n_x=10, n_y=10, layers=4, nudge_ratio_x=0.001, nudge_ratio_y=0.001):
        self.squares = []
        self.squares_distribution = []

        min_x_nudged = self.min_x + self.width * nudge_ratio_x
        max_x_nudged = self.max_x - self.width * nudge_ratio_x
        min_y_nudged = self.min_y + self.height * nudge_ratio_y
        max_y_nudged = self.max_y - self.height * nudge_ratio_y

        dx = (max_x_nudged - min_x_nudged) / n_x
        dy = (max_y_nudged - min_y_nudged) / n_y
        squares_to_check = [Square(min_x_nudged + x_idx * dx,
                                   min_y_nudged + y_idx * dy,
                                   min_x_nudged + (x_idx+1) * dx,
                                   min_y_nudged + (y_idx+1) * dy) for x_idx in range(n_x) for y_idx in range(n_y)]
        squares_to_check_ = []

        for layer in range(layers):
            inclusion_count = 0
            dx = dx / 2
            dy = dy / 2

            for square in squares_to_check:
                deg_check = int(self.degree_check(square.min_x, square.min_y)) + \
                            int(self.degree_check(square.max_x, square.min_y)) + \
                            int(self.degree_check(square.min_x, square.max_y)) + \
                            int(self.degree_check(square.max_x, square.max_y))

                if deg_check == 4:
                    self.squares.append(square)
                    inclusion_count = inclusion_count + 1
                elif deg_check > 0:
                    squares_to_check_.append(Square(square.min_x, square.min_y, square.min_x + dx, square.min_y + dy))
                    squares_to_check_.append(Square(square.min_x + dx, square.min_y, square.max_x, square.min_y + dy))
                    squares_to_check_.append(Square(square.min_x, square.min_y + dy, square.min_x + dx, square.max_y))
                    squares_to_check_.append(Square(square.min_x + dx, square.min_y + dy, square.max_x, square.max_y))

            squares_to_check = squares_to_check_
            squares_to_check_ = []
            self.squares_counts.append(inclusion_count)

        self.squares_distribution = self.squares_counts.copy()
        for idx in range(layers):
            self.squares_distribution[idx] = self.squares_distribution[idx] * (4**(-idx+1))
        weights_sum = np.sum(self.squares_distribution)
        if weights_sum > 0:
            self.squares_distribution = self.squares_distribution / weights_sum
        self.last_sampling = 'degree_check'
        return

    # PLOT_DOMAIN: Plots specified domain
    def plot_domain(self):
        plt.plot(self.points[0], self.points[1], 'b')
        plt.plot([self.points[0][-1], self.points[0][0]], [self.points[1][-1], self.points[1][0]], 'b')
        return

    # PLOT_DISTRIBUTION: Plots generated distribution
    def plot_distribution(self, distribution='last_sampled'):
        if distribution == 'last_sampled':
            distribution = self.last_sampling

        if distribution == 'line_probe':
            for line_probe in self.line_probes:
                plt.plot([line_probe.min_x, line_probe.max_x], [line_probe.y, line_probe.y], 'g')

        elif distribution == 'degree_check':
            for square in self.squares:
                plt.plot([square.min_x, square.max_x, square.max_x, square.min_x, square.min_x],
                         [square.min_y, square.min_y, square.max_y, square.max_y, square.min_y], 'g')
        return

    # SAMPLE: Chooses one random sample from pre-sampled domain
    def sample(self, count=1, distribution='last_sampled'):
        if distribution == 'last_sampled':
            distribution = self.last_sampling

        if distribution == 'line_probe':
            line_probes = np.random.choice(self.line_probes, count, p=self.line_probes_distribution)
            x = []
            y = []
            for idx in range(count):
                x.append(np.random.uniform(line_probes[idx].min_x, line_probes[idx].max_x))
                y.append(line_probes[idx].y)
            return [x, y]

        elif distribution == 'degree_check':
            layers = np.random.choice(range(len(self.squares_counts)), count, p=self.squares_distribution)
            x = []
            y = []
            for layer in range(count):
                idx = int(np.sum(self.squares_counts[0:layers[layer]]) + np.random.randint(0, self.squares_counts[layers[layer]] - 1))
                x.append(np.random.uniform(self.squares[idx].min_x, self.squares[idx].max_x))
                y.append(np.random.uniform(self.squares[idx].min_y, self.squares[idx].max_y))
            return [x, y]

        return []


if __name__ == '__main__':

    # ds = DomainSampler([[1, 1, 0, 0], [0, 1, 1, 0]])  # Square
    ds = DomainSampler([[-0.5, 0.9, 1.7, -1, 0], [-1, -1.2, 2, 1, 0.2]])  # Ordinary shape
    # ds = DomainSampler([[0, 1, 1, 2, 2, 3, 3, 0], [0, 0, 1, 1, 0, 0, 2, 2]])  # Non-convex box horizontal shape
    # ds = DomainSampler([[0, -1, -2, -1, 0, 1, 1, 0, -1, -2, -1], [0, 1, 1, 2, 2, 1, -1, -2, -2, -1, -1]])  # Pacman
    # ds = DomainSampler([[0, 0.5, 1, 1, 0.5, 0], [0, 1, 0, 1, 0, 1]])  # Invalid shape? still works fine
    # ds = DomainSampler([[0, 1, 2], [1, 1, 1]])  # Invalid shape, not 2D
    # ds = DomainSampler([[1, 1, 1], [0, 1, 2]])  # Invalid shape, not 2D
    # ds = DomainSampler([[0, 0.5, 1, 0], [0, 0.9, 1, 1]])  # very acute corners (<= 45°)
    # ds = DomainSampler([[0, 0, 1, 0.35, 1], [0, 1, 1, 0.5, 0]])  # very acute corners (<= 45°)

    ds.sample_line_probe()
    # ds.sample_degree_check_hierarchical()
    r = ds.sample(50)

    ds.plot_domain()
    ds.plot_distribution()
    plt.plot(r[0], r[1], 'r+')

    plt.show()
