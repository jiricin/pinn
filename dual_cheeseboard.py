from cheeseboard import *


class DualPINN:
    def __init__(self, pinn1, pinn2, interface_vert, interface_hor):
        self.pinn1 = pinn1
        self.pinn2 = pinn2

        self.interface_vert_x = torch.tensor(interface_vert[0].reshape(-1, 1),
                                             dtype=torch.float32,
                                             requires_grad=True)

        self.interface_vert_y = torch.tensor(interface_vert[1].reshape(-1, 1),
                                             dtype=torch.float32,
                                             requires_grad=True)

        self.interface_hor_x = torch.tensor(interface_hor[0].reshape(-1, 1),
                                            dtype=torch.float32,
                                            requires_grad=True)

        self.interface_hor_y = torch.tensor(interface_hor[1].reshape(-1, 1),
                                            dtype=torch.float32,
                                            requires_grad=True)

        self.null_vert = torch.zeros((self.interface_vert_x.shape[0], 1))
        self.null_hor = torch.zeros((self.interface_hor_x.shape[0], 1))

        self.loss_function = nn.MSELoss()
        self.loss = 0

        self.optimizer = torch.optim.Adam(list(self.pinn1.model.parameters()) + list(self.pinn2.model.parameters()),
                                          lr=0.01,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=0,
                                          amsgrad=True)

        self.iterations = 0

        self.alpha_eq_val = 1
        self.alpha_eq_diff = 1

    def evaluate_interface(self):
        output_vert_1 = pinn_1.evaluate(self.interface_vert_x, self.interface_vert_y)
        output_vert_2 = pinn_2.evaluate(self.interface_vert_x, self.interface_vert_y)
        output_hor_1 = pinn_1.evaluate(self.interface_hor_x, self.interface_hor_y)
        output_hor_2 = pinn_2.evaluate(self.interface_hor_x, self.interface_hor_y)

        d_vert_n_1 = torch.autograd.grad(
            output_vert_1, self.interface_vert_x,
            grad_outputs=torch.ones_like(output_vert_1),
            retain_graph=True,
            create_graph=True)[0]

        d_vert_n_2 = torch.autograd.grad(
            output_vert_2, self.interface_vert_x,
            grad_outputs=torch.ones_like(output_vert_2),
            retain_graph=True,
            create_graph=True)[0]

        d_hor_n_1 = torch.autograd.grad(
            output_hor_1, self.interface_hor_y,
            grad_outputs=torch.ones_like(output_hor_1),
            retain_graph=True,
            create_graph=True)[0]

        d_hor_n_2 = torch.autograd.grad(
            output_hor_2, self.interface_hor_y,
            grad_outputs=torch.ones_like(output_hor_2),
            retain_graph=True,
            create_graph=True)[0]

        return [output_vert_1 - output_vert_2, output_hor_1 - output_hor_2, d_vert_n_1 - d_vert_n_2, d_hor_n_1 - d_hor_n_2]

    def closure(self):
        self.optimizer.zero_grad()

        evaluations = self.evaluate_interface()

        eq_val1_prediction = evaluations[0]
        eq_val2_prediction = evaluations[1]
        eq_diff1_prediction = evaluations[2]
        eq_diff2_prediction = evaluations[3]

        eq_val1_loss = self.alpha_eq_val * self.loss_function(eq_val1_prediction, self.null_vert)
        eq_val2_loss = self.alpha_eq_val * self.loss_function(eq_val2_prediction, self.null_hor)
        eq_diff1_loss = self.alpha_eq_diff * self.loss_function(eq_diff1_prediction, self.null_vert)
        eq_diff2_loss = self.alpha_eq_diff * self.loss_function(eq_diff2_prediction, self.null_hor)

        self.loss = eq_val1_loss + eq_val2_loss + eq_diff1_loss + eq_diff2_loss
        self.loss.backward()

        self.iterations += 1

        if not self.iterations % 100:
            print('Iterations: {0:}, INTERFACE Loss: {1:6.10f}'.format(self.iterations, self.loss))

        return self.loss

    def train(self):
        self.pinn1.model.train()  # only sets a flag
        self.pinn2.model.train()  # only sets a flag
        self.optimizer.step(self.closure)


if __name__ == '__main__':
    # problem information
    x_square = [0, 1]  # domain of one small square
    y_square = [0, 1]  # domain of one small square
    rect_width = x_square[1] - x_square[0]
    rect_height = y_square[1] - y_square[0]

    material_rects = [4, 4]
    material_coefs = [1, 10]

    dirichlet_const_A = 1
    dirichlet_const_B = 1
    neumann_const = 0
    robin_const = 3
    robin_alpha = 2

    # domain preparation
    boundaries_omega = [[], []]
    for i in range(material_rects[0]):
        for j in range(material_rects[1]):
            x_min = x_square[0] + i * rect_width
            x_max = x_square[1] + i * rect_width
            y_min = y_square[0] + j * rect_height
            y_max = y_square[1] + j * rect_height
            boundary = Boundary([[x_min, x_max, x_max, x_min],
                                 [y_min, y_min, y_max, y_max]])
            boundaries_omega[(i + j) % 2].append(boundary)

    ds_omega = [DomainSampler(boundaries_omega[0]), DomainSampler(boundaries_omega[1])]

    ds_omega[0].boundary_group_add('dirichlet_A', 0, [3, 0], connect_ends=False, color='blue')
    ds_omega[1].boundary_group_add('dirichlet_A', 0, [3, 0], connect_ends=False, color='blue')
    ds_omega[0].boundary_group_add('dirichlet_A', 1, [3, 0], connect_ends=False, color='blue')
    ds_omega[1].boundary_group_add('dirichlet_A', 1, [3, 0], connect_ends=False, color='blue')

    ds_omega[1].boundary_group_add('dirichlet_B', 1, [2, 3], connect_ends=False, color='aqua')
    ds_omega[0].boundary_group_add('dirichlet_B', 3, [2, 3], connect_ends=False, color='aqua')
    ds_omega[1].boundary_group_add('dirichlet_B', 5, [2, 3], connect_ends=False, color='aqua')
    ds_omega[0].boundary_group_add('dirichlet_B', 7, [2, 3], connect_ends=False, color='aqua')

    ds_omega[0].boundary_group_add('robin', 7, [1, 2], connect_ends=False, color='red')
    ds_omega[1].boundary_group_add('robin', 7, [1, 2], connect_ends=False, color='red')
    ds_omega[0].boundary_group_add('robin', 6, [1, 2], connect_ends=False, color='red')
    ds_omega[1].boundary_group_add('robin', 6, [1, 2], connect_ends=False, color='red')

    ds_omega[1].boundary_group_add('neumann', 6, [0, 1], connect_ends=False, color='orange')
    ds_omega[0].boundary_group_add('neumann', 4, [0, 1], connect_ends=False, color='orange')
    ds_omega[1].boundary_group_add('neumann', 2, [0, 1], connect_ends=False, color='orange')
    ds_omega[0].boundary_group_add('neumann', 0, [0, 1], connect_ends=False, color='orange')

    for ds in ds_omega:
        ds.distribute_line_probe(n_y=40)
        ds.boundary_groups_distribute()

    x_bounds = [x_square[0], x_square[0] + material_rects[0] * rect_width]
    y_bounds = [y_square[0], y_square[0] + material_rects[1] * rect_height]
    ds_bound = DomainSampler([Boundary([[x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0]],
                                        [y_bounds[0], y_bounds[0], y_bounds[1], y_bounds[1]]])])

    ds_bound.boundary_group_add('dirichlet_A', 0, [3, 0], connect_ends=False, color='blue')
    ds_bound.boundary_group_add('dirichlet_B', 0, [2, 3], connect_ends=False, color='aqua')
    ds_bound.boundary_group_add('neumann', 0, [0, 1], connect_ends=False, color='orange')
    ds_bound.boundary_group_add('robin', 0, [1, 2], connect_ends=False, color='red')
    ds_bound.boundary_groups_distribute()

    # generate interior samples
    i_count = 200

    i_loc_O1 = np.array(ds_omega[0].sample_interior(i_count))
    i_O1 = np.ones_like(i_loc_O1[0]) * material_coefs[0]

    i_loc_O2 = np.array(ds_omega[1].sample_interior(i_count))
    i_O2 = np.ones_like(i_loc_O2[0]) * material_coefs[1]

    # generate dirichlet samples
    b_count = 80

    d_loc_A_O1 = ds_omega[0].sample_boundary(b_count, label='dirichlet_A')
    d_loc_B_O1 = ds_omega[0].sample_boundary(b_count, label='dirichlet_B')
    d_loc_O1 = np.array([d_loc_A_O1[0] + d_loc_B_O1[0], d_loc_A_O1[1] + d_loc_B_O1[1]])
    d_O1 = np.append(dirichlet_const_A * np.ones(b_count),
                     dirichlet_const_B * np.ones(b_count))

    d_loc_A_O2 = ds_omega[1].sample_boundary(b_count, label='dirichlet_A')
    d_loc_B_O2 = ds_omega[1].sample_boundary(b_count, label='dirichlet_B')
    d_loc_O2 = np.array([d_loc_A_O2[0] + d_loc_B_O2[0], d_loc_A_O2[1] + d_loc_B_O2[1]])
    d_O2 = np.append(dirichlet_const_A * np.ones(b_count),
                     dirichlet_const_B * np.ones(b_count))

    # generate neumann samples
    n_loc_O1 = np.array(ds_omega[0].sample_boundary(b_count, label='neumann'))
    n_O1 = neumann_const * np.ones(b_count)

    n_loc_O2 = np.array(ds_omega[1].sample_boundary(b_count, label='neumann'))
    n_O2 = neumann_const * np.ones(b_count)

    # generate robin samples
    r_loc_O1 = np.array(ds_omega[0].sample_boundary(b_count, label='robin'))
    r_O1 = robin_const * np.ones(b_count)
    r_material_O1 = material_coefs[0] * np.ones(b_count)

    r_loc_O2 = np.array(ds_omega[1].sample_boundary(b_count, label='robin'))
    r_O2 = robin_const * np.ones(b_count)
    r_material_O2 = material_coefs[1] * np.ones(b_count)

    # generate interface samples
    f_count = 150
    f_second_third = 100
    f_third = 50

    interface_vert = np.array(ds_bound.sample_boundary(f_count, label='dirichlet_A'))
    interface_vert[0][0:f_third] += rect_width
    interface_vert[0][f_third:f_second_third] += rect_width * 2
    interface_vert[0][f_second_third:f_count] += rect_width * 3

    interface_hor = np.array(ds_bound.sample_boundary(f_count, label='dirichlet_B'))
    interface_hor[1][0:f_third] -= rect_height
    interface_hor[1][f_third:f_second_third] -= rect_height * 2
    interface_hor[1][f_second_third:f_count] -= rect_height * 3

    # (uncomment code below to check the point sampling)

    # ds_omega[0].plot_domain()
    # ds_omega[0].plot_distribution_interior()
    # ds_omega[0].plot_distribution_boundary()
    # plt.plot(i_loc_O1[0], i_loc_O1[1], 'm+')

    # ds_omega[1].plot_domain()
    # ds_omega[1].plot_distribution_interior()
    # ds_omega[1].plot_distribution_boundary()
    # plt.plot(i_loc_O2[0], i_loc_O2[1], 'm+')

    # plt.plot(interface_vert[0], interface_vert[1], 'm+')
    # plt.plot(interface_hor[0], interface_hor[1], 'm+')

    # plt.show()

    # create PINN
    pinn_1 = PINN(i_loc_O1, i_O1, d_loc_O1, d_O1, n_loc_O1, n_O1, r_loc_O1, r_O1, r_material_O1, robin_alpha)
    pinn_2 = PINN(i_loc_O2, i_O2, d_loc_O2, d_O2, n_loc_O2, n_O2, r_loc_O2, r_O2, r_material_O2, robin_alpha)
    dual_pinn = DualPINN(pinn_1, pinn_2, interface_vert, interface_hor)

    # train PINN
    for r in range(1000):
        pinn_1.train()
        pinn_2.train()
        dual_pinn.train()

    # visualize results
    pinn_1.plot_equation_loss('Equation loss', 'pde_loss_1.png')
    pinn_1.plot_trained_function('Trained function', 'func_1.png',
                                 x_min=x_bounds[0], x_max=x_bounds[1], y_min=y_bounds[0], y_max=y_bounds[1])

    pinn_2.plot_equation_loss('Equation loss', 'pde_loss_2.png')
    pinn_2.plot_trained_function('Trained function', 'func_2.png',
                                 x_min=x_bounds[0], x_max=x_bounds[1], y_min=y_bounds[0], y_max=y_bounds[1])

    coordinates_x1 = np.array(pinn_1.loc_x.detach().numpy().transpose()[0])
    coordinates_x2 = np.array(pinn_2.loc_x.detach().numpy().transpose()[0])
    coordinates_y1 = np.array(pinn_1.loc_y.detach().numpy().transpose()[0])
    coordinates_y2 = np.array(pinn_2.loc_y.detach().numpy().transpose()[0])

    coordinates_x = np.append(coordinates_x1, coordinates_x2)
    coordinates_y = np.append(coordinates_y1, coordinates_y2)

    coordinates = np.array([coordinates_x, coordinates_y]).transpose()

    values = np.append(pinn_1.evaluate(pinn_1.loc_x, pinn_1.loc_y).detach().numpy().transpose()[0],
                       pinn_2.evaluate(pinn_2.loc_x, pinn_2.loc_y).detach().numpy().transpose()[0])

    display_heatmap(coordinates, values, 'Trained function', 'func_dual')
