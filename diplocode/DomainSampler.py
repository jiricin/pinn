import math
import numpy as np
import matplotlib.pyplot as plt


class Boundary:
    def __init__(self, ordered_points):
        # ordered_points: [[X], [Y]]
        # defines domain border, end-points are treated as connected automatically

        self.points = ordered_points
        self.point_count = len(self.points[1])

        self.min_y = np.min(self.points[1])
        self.max_y = np.max(self.points[1])
        self.min_x = np.min(self.points[0])
        self.max_x = np.max(self.points[0])
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y


class BoundaryGroup:
    def __init__(self, boundary_idx, point_indices, label, color='blue'):
        self.boundary_idx = boundary_idx
        self.point_indices = point_indices
        self.label = label
        self.distribution = []
        self.length = 0
        self.color = color


class LineProbe:
    def __init__(self, y, min_x, max_x):
        self.y = y
        self.min_x = min_x
        self.max_x = max_x


class Square:
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y


class DomainSampler:
    def __init__(self, boundaries_array):
        self.boundaries = boundaries_array

        x_minis = [np.min(b.points[0]) for b in self.boundaries]
        x_maxes = [np.max(b.points[0]) for b in self.boundaries]
        y_minis = [np.min(b.points[1]) for b in self.boundaries]
        y_maxes = [np.max(b.points[1]) for b in self.boundaries]
        self.min_x = np.min(x_minis)
        self.max_x = np.max(x_maxes)
        self.min_y = np.min(y_minis)
        self.max_y = np.max(y_maxes)
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y

        self.boundary_groups = []

        self.line_probes = []
        self.line_probes_distribution = []

        self.squares = []
        self.squares_counts = []
        self.squares_distribution = []

        self.last_sampling = None

        self.epsilon = np.finfo(float).eps

    # BOUNDARY_GROUP_ADD: Adds a boundary group to the list and labels it for further use
    # connect_ends - can be left unused, instead end-point and start-point can be the same
    # point_indices - if left as [], which adds whole boundary with boundary_idx
    def boundary_group_add(self, label, boundary_idx, point_indices, connect_ends=False, color='blue'):
        if not point_indices:
            point_indices = list(range(0, self.boundaries[boundary_idx].point_count))
            connect_ends = True
        if connect_ends and len(point_indices) == self.boundaries[boundary_idx].point_count:
            point_indices.append(point_indices[0])
        self.boundary_groups.append(BoundaryGroup(boundary_idx, point_indices, label, color))
        return

    # BOUNDARY_GROUPS_REMOVE: Removes a boundary group from the list by label
    def boundary_groups_remove(self, label):
        self.boundary_groups = [bg for bg in self.boundary_groups if not bg.label == label]
        return

    # BOUNDARY_GROUPS_DISTRIBUTE: Creates domain boundary distribution; distributes all boundary groups if label is None
    # if a boundary group does not exist, creates one with all boundaries and distributes it
    def boundary_groups_distribute(self, label=None):
        bg_count = len(self.boundary_groups)
        if bg_count <= 0:
            for idx in range(len(self.boundaries)):
                self.boundary_group_add('default', idx, [], connect_ends=True)
            label = 'default'

        for bg in self.boundary_groups:
            if label is None or bg.label == label:
                bg.distribution = []
                boundary_point_count = len(bg.point_indices)
                idx = 0
                while idx < boundary_point_count - 1:
                    dx = self.boundaries[bg.boundary_idx].points[0][bg.point_indices[idx + 1]] \
                        - self.boundaries[bg.boundary_idx].points[0][bg.point_indices[idx]]
                    dy = self.boundaries[bg.boundary_idx].points[1][bg.point_indices[idx + 1]] \
                        - self.boundaries[bg.boundary_idx].points[1][bg.point_indices[idx]]
                    bg.distribution.append(math.sqrt(dx**2 + dy**2))
                    bg.length = bg.length + bg.distribution[-1]
                    idx = idx + 1
                if bg.length > 0:
                    bg.distribution = np.asarray(bg.distribution) / bg.length
        return

    # DISTRIBUTE_LINE_PROBE: Creates domain interior distribution using horizontal line probes
    def distribute_line_probe(self, n_y=20, nudge_ratio_y=0.01):
        self.line_probes = []
        self.line_probes_distribution = []

        # uniformly generated lines
        for y in np.linspace(self.min_y + self.height * nudge_ratio_y, self.max_y - self.height * nudge_ratio_y, n_y):

            # find intersections with domain boundary
            ordered_intersections = []

            for boundary in self.boundaries:
                idx_shift = 0
                while boundary.points[1][idx_shift-1] == y:
                    idx_shift = idx_shift + 1
                    if idx_shift > boundary.point_count:
                        return

                idx = 0
                while idx < boundary.point_count:
                    idx_shifted = (idx + idx_shift) % boundary.point_count
                    idx_prev = idx_shifted - 1
                    sgn_prev = boundary.points[1][idx_prev] - y
                    sgn = boundary.points[1][idx_shifted] - y

                    if sgn_prev * sgn < 0:  # intersection somewhere between points
                        x_is = boundary.points[0][idx_prev] - sgn_prev * \
                               (boundary.points[0][idx_shifted] - boundary.points[0][idx_prev]) / \
                               (boundary.points[1][idx_shifted] - boundary.points[1][idx_prev])
                        ordered_intersections.append([x_is, x_is, True])

                    if sgn == 0:  # intersection at an edge point
                        x1_is = boundary.points[0][idx_shifted]
                        while sgn == 0:
                            idx += 1
                            idx_shifted = (idx + idx_shift) % boundary.point_count
                            sgn = boundary.points[1][idx_shifted] - y
                        x2_is = boundary.points[0][idx_shifted - 1]
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
                x_min = ordered_intervals[idx] + self.epsilon
                x_max = ordered_intervals[idx+1] - self.epsilon
                self.line_probes.append(LineProbe(y, x_min, x_max))
                self.line_probes_distribution.append(x_max - x_min)

        # normalize distribution
        lengths_sum = np.sum(self.line_probes_distribution)
        if lengths_sum > 0:
            self.line_probes_distribution = self.line_probes_distribution / lengths_sum
        self.last_sampling = 'line_probe'
        return

    # DEGREE_CHECK: Checks if chosen point lies in the domain (interior) using degree-sum
    def degree_check(self, x, y):
        deg_sum = 0.0
        sgn = 1.0
        for boundary in self.boundaries:
            for idx in range(boundary.point_count):
                p1 = boundary.points[0][idx - 1] - x
                p2 = boundary.points[1][idx - 1] - y
                n1 = boundary.points[0][idx] - x
                n2 = boundary.points[1][idx] - y
                pp = p1*p1+p2*p2
                nn = n1*n1+n2*n2
                if pp == 0 or nn == 0:
                    return False
                pn_dot_norm = np.clip((p1 * n1 + p2 * n2) / (math.sqrt(pp) * math.sqrt(nn)), -1.0, 1.0)
                pn_cross = p2 * n1 - p1 * n2
                deg_sum = deg_sum + sgn * np.sign(pn_cross) * math.acos(pn_dot_norm)
            sgn = -sgn
        return (abs(deg_sum) - 2 * np.pi)**2 < 1.0e-5

    # DISTRIBUTE_DEGREE_CHECK_HIERARCHICAL: Creates domain interior distribution using degree-sum check
    def distribute_degree_check_hierarchical(self, n_x=10, n_y=10, layers=4, nudge_ratio_x=0.001, nudge_ratio_y=0.001):
        self.squares = []
        self.squares_distribution = []

        min_x_nudged = self.min_x + self.width * nudge_ratio_x
        max_x_nudged = self.max_x - self.width * nudge_ratio_x
        min_y_nudged = self.min_y + self.height * nudge_ratio_y
        max_y_nudged = self.max_y - self.height * nudge_ratio_y

        dx = (max_x_nudged - min_x_nudged) / n_x
        dy = (max_y_nudged - min_y_nudged) / n_y

        # grid of squares
        squares_to_check = [Square(min_x_nudged + x_idx * dx,
                                   min_y_nudged + y_idx * dy,
                                   min_x_nudged + (x_idx+1) * dx,
                                   min_y_nudged + (y_idx+1) * dy) for x_idx in range(n_x) for y_idx in range(n_y)]
        squares_to_check_ = []

        # at each layer, the squares are cut to four quarters
        for layer in range(layers):
            inclusion_count = 0
            dx = dx / 2
            dy = dy / 2

            # check for all squares: do vertices lie in domain? yes - include, no - cut and check again
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

        # normalize distribution
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
        for boundary in self.boundaries:
            plt.plot(boundary.points[0], boundary.points[1], color='gray', linestyle='dashed')
            plt.plot([boundary.points[0][-1], boundary.points[0][0]],
                     [boundary.points[1][-1], boundary.points[1][0]],
                     color='gray', linestyle='dashed')
        return

    # PLOT_DISTRIBUTION_BOUNDARY: Plots generated boundary distribution
    def plot_distribution_boundary(self, label=None):
        if label is None:
            for bg in self.boundary_groups:
                plt.plot([self.boundaries[bg.boundary_idx].points[0][idx] for idx in bg.point_indices],
                         [self.boundaries[bg.boundary_idx].points[1][idx] for idx in bg.point_indices],
                         color=bg.color, linestyle='solid')
        else:
            for bg in self.boundary_groups:
                if bg.label == label:
                    plt.plot([self.boundaries[bg.boundary_idx].points[0][idx] for idx in bg.point_indices],
                             [self.boundaries[bg.boundary_idx].points[1][idx] for idx in bg.point_indices],
                             color=bg.color, linestyle='solid')
        return

    # PLOT_DISTRIBUTION_INTERIOR: Plots generated interior distribution
    def plot_distribution_interior(self, distribution='last_sampled'):
        if distribution == 'last_sampled':
            distribution = self.last_sampling

        if distribution == 'line_probe':
            for line_probe in self.line_probes:
                plt.plot([line_probe.min_x, line_probe.max_x], [line_probe.y, line_probe.y],
                         color='green', linestyle='solid')

        elif distribution == 'degree_check':
            for square in self.squares:
                plt.plot([square.min_x, square.max_x, square.max_x, square.min_x, square.min_x],
                         [square.min_y, square.min_y, square.max_y, square.max_y, square.min_y],
                         color='green', linestyle='solid')
        return

    # SAMPLE_BOUNDARY: Generates samples from labeled boundary group; if label is None, samples from all boundary groups
    def sample_boundary(self, count=1, label=None):
        macro_bg_indices = []
        macro_boundary_distribution = []
        lengths_sum = 0
        x = []
        y = []

        # compute overall length and boundary groups' indices of distribution
        for idx, bg in enumerate(self.boundary_groups):
            if label is None or bg.label == label:
                macro_bg_indices.append(idx)
                macro_boundary_distribution.append(bg.length)
                lengths_sum = lengths_sum + bg.length

        if lengths_sum <= 0:
            return []

        # generate random vector of boundary groups (indices)
        macro_boundary_distribution = np.asarray(macro_boundary_distribution) / lengths_sum
        rand_bg_idx_vec = np.random.choice(macro_bg_indices, count, p=macro_boundary_distribution)

        # for each generated boundary group: generate random vector of boundary lines
        for idx in macro_bg_indices:
            sample_count = rand_bg_idx_vec.tolist().count(idx)
            rand_line_idx_vec = np.random.choice(range(len(self.boundary_groups[idx].distribution)),
                                                 sample_count,
                                                 p=self.boundary_groups[idx].distribution)
            # for each generated boundary line: generate random samples
            for lix in rand_line_idx_vec:
                bg = self.boundary_groups[idx]
                x1 = self.boundaries[bg.boundary_idx].points[0][bg.point_indices[lix]]
                x2 = self.boundaries[bg.boundary_idx].points[0][bg.point_indices[lix + 1]]
                y1 = self.boundaries[bg.boundary_idx].points[1][bg.point_indices[lix]]
                y2 = self.boundaries[bg.boundary_idx].points[1][bg.point_indices[lix + 1]]
                scale = np.random.uniform(0, 1)
                x.append(x1 + scale * (x2 - x1))
                y.append(y1 + scale * (y2 - y1))
        return [x, y]

    # SAMPLE_INTERIOR: Generates samples from distributed domain interior
    def sample_interior(self, count=1, distribution='last_sampled'):
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


def example():
    boundary_1 = Boundary([[1, 1, 0, 0],
                           [0, 1, 1, 0]])
    boundary_2 = Boundary([[0.5, 0.75, 0.5, 0.25],
                           [0.25, 0.5, 0.75, 0.5]])
    boundary_3 = Boundary([[0.9, 0.9, 0.65],
                           [0.1, 0.35, 0.1]])
    boundary_4 = Boundary([[0.5, 0.6, 0.5, 0.32],
                           [0.32, 0.5, 0.6, 0.5]])
    boundary_array = [boundary_2, boundary_1, boundary_4, boundary_3]
    ds = DomainSampler(boundary_array)

    ds.boundary_group_add('Dirichlet', 0, list(range(0, 4)), connect_ends=False, color='aqua')
    ds.boundary_group_add('Dirichlet', 1, list(range(0, 4)), connect_ends=True, color='aqua')
    ds.boundary_group_add('Neumann', 3, list(range(1, 3)), color='orange')
    ds.boundary_group_add('Neumann', 2, [], color='orange')
    ds.boundary_groups_distribute()
    samples_boundary_dirichlet = ds.sample_boundary(100, 'Dirichlet')
    samples_boundary_neumann = ds.sample_boundary(40, 'Neumann')

    ds.distribute_line_probe()
    # ds.distribute_degree_check_hierarchical()
    samples_interior = ds.sample_interior(200)

    ds.plot_domain()
    ds.plot_distribution_interior()
    ds.plot_distribution_boundary()

    plt.plot(samples_interior[0], samples_interior[1], 'm+')
    plt.plot(samples_boundary_dirichlet[0], samples_boundary_dirichlet[1], 'r+')
    plt.plot(samples_boundary_neumann[0], samples_boundary_neumann[1], 'r+')

    plt.show()


if __name__ == '__main__':
    example()
