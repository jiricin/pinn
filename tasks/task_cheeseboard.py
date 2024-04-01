from DomainSampler import DomainSampler, Boundary
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


class NetworkModel(nn.Module):
    def __init__(self):
        super(NetworkModel, self).__init__()

        self.linear_start = nn.Linear(2, 30)
        self.linear_middle1 = nn.Linear(60, 30)
        self.linear_middle2 = nn.Linear(30, 30)
        self.linear_middle3 = nn.Linear(30, 30)
        self.linear_middle4 = nn.Linear(30, 30)
        self.linear_middle5 = nn.Linear(30, 30)
        self.linear_middle6 = nn.Linear(30, 30)
        self.linear_middle7 = nn.Linear(30, 30)
        self.linear_end = nn.Linear(30, 1)
        self.tangent = nn.Tanh()

    def forward(self, y):
        y = self.linear_start(y)

        # fourier features
        x1 = torch.sin(y)
        x2 = torch.cos(y)
        x = torch.cat((x1, x2), dim=1)

        x = self.linear_middle1(x)
        x = self.tangent(x)
        x = self.linear_middle2(x)
        x = self.tangent(x)
        x = self.linear_middle3(x)
        x = self.tangent(x)
        x = self.linear_middle4(x)
        x = self.tangent(x)
        x = self.linear_middle5(x)
        x = self.tangent(x)
        x = self.linear_end(x)
        return x


class PINN:
    def __init__(self, i_loc, i_mat, i_mdf, d_loc, d_val, n_loc, n_val, n_dir, r_loc, r_mat, r_val, r_alf, r_dir, t_loc, t_dir, t_mat):
        self.i_x_L = torch.tensor(i_loc[0][0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.i_y_L = torch.tensor(i_loc[0][1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.i_mat_L = torch.tensor(i_mat[0].reshape(-1, 1), dtype=torch.float32)
        self.i_mdf_x_L = torch.tensor(i_mdf[0][0].reshape(-1, 1), dtype=torch.float32)
        self.i_mdf_y_L = torch.tensor(i_mdf[0][1].reshape(-1, 1), dtype=torch.float32)
        self.d_x_L = torch.tensor(d_loc[0][0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.d_y_L = torch.tensor(d_loc[0][1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.d_val_L = torch.tensor(d_val[0].reshape(-1, 1), dtype=torch.float32)
        self.n_x_L = torch.tensor(n_loc[0][0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.n_y_L = torch.tensor(n_loc[0][1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.n_val_L = torch.tensor(n_val[0].reshape(-1, 1), dtype=torch.float32)
        self.n_dir_x_L = torch.tensor(n_dir[0][0].reshape(-1, 1), dtype=torch.float32)
        self.n_dir_y_L = torch.tensor(n_dir[0][1].reshape(-1, 1), dtype=torch.float32)
        self.r_x_L = torch.tensor(r_loc[0][0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.r_y_L = torch.tensor(r_loc[0][1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.r_mat_L = torch.tensor(r_mat[0].reshape(-1, 1), dtype=torch.float32)
        self.r_val_L = torch.tensor(r_val[0].reshape(-1, 1), dtype=torch.float32)
        self.r_alf_L = torch.tensor(r_alf[0].reshape(-1, 1), dtype=torch.float32)
        self.r_dir_x_L = torch.tensor(r_dir[0][0].reshape(-1, 1), dtype=torch.float32)
        self.r_dir_y_L = torch.tensor(r_dir[0][1].reshape(-1, 1), dtype=torch.float32)

        self.i_x_R = torch.tensor(i_loc[1][0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.i_y_R = torch.tensor(i_loc[1][1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.i_mat_R = torch.tensor(i_mat[1].reshape(-1, 1), dtype=torch.float32)
        self.i_mdf_x_R = torch.tensor(i_mdf[1][0].reshape(-1, 1), dtype=torch.float32)
        self.i_mdf_y_R = torch.tensor(i_mdf[1][1].reshape(-1, 1), dtype=torch.float32)
        self.d_x_R = torch.tensor(d_loc[1][0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.d_y_R = torch.tensor(d_loc[1][1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.d_val_R = torch.tensor(d_val[1].reshape(-1, 1), dtype=torch.float32)
        self.n_x_R = torch.tensor(n_loc[1][0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.n_y_R = torch.tensor(n_loc[1][1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.n_val_R = torch.tensor(n_val[1].reshape(-1, 1), dtype=torch.float32)
        self.n_dir_x_R = torch.tensor(n_dir[1][0].reshape(-1, 1), dtype=torch.float32)
        self.n_dir_y_R = torch.tensor(n_dir[1][1].reshape(-1, 1), dtype=torch.float32)
        self.r_x_R = torch.tensor(r_loc[1][0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.r_y_R = torch.tensor(r_loc[1][1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.r_mat_R = torch.tensor(r_mat[1].reshape(-1, 1), dtype=torch.float32)
        self.r_val_R = torch.tensor(r_val[1].reshape(-1, 1), dtype=torch.float32)
        self.r_alf_R = torch.tensor(r_alf[1].reshape(-1, 1), dtype=torch.float32)
        self.r_dir_x_R = torch.tensor(r_dir[1][0].reshape(-1, 1), dtype=torch.float32)
        self.r_dir_y_R = torch.tensor(r_dir[1][1].reshape(-1, 1), dtype=torch.float32)

        self.t_x = torch.tensor(t_loc[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.t_y = torch.tensor(t_loc[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.t_dir_x = torch.tensor(t_dir[0].reshape(-1, 1), dtype=torch.float32)
        self.t_dir_y = torch.tensor(t_dir[1].reshape(-1, 1), dtype=torch.float32)
        self.t_mat_L = torch.tensor(t_mat[0].reshape(-1, 1), dtype=torch.float32)
        self.t_mat_R = torch.tensor(t_mat[1].reshape(-1, 1), dtype=torch.float32)

        self.null_eq_L = torch.zeros((self.i_x_L.shape[0], 1))
        self.null_r_L = torch.zeros((self.r_x_L.shape[0], 1))

        self.null_eq_R = torch.zeros((self.i_x_R.shape[0], 1))
        self.null_r_R = torch.zeros((self.r_x_R.shape[0], 1))

        self.null_t = torch.zeros((self.t_x.shape[0], 1))

        self.model_L = NetworkModel()
        self.model_R = NetworkModel()

        self.optimizer = torch.optim.Adam(list(self.model_L.parameters()) + list(self.model_R.parameters()),
                                          lr=0.01,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=0,
                                          amsgrad=True)

        self.loss_function = nn.MSELoss()
        self.loss = 0
        self.iterations = 0

        self.eq_loss_alf = 1
        self.d_loss_alf = 1
        self.n_loss_alf = 1
        self.r_loss_alf = 1
        self.t_loss_alf = 1

    def evaluate(self, model, *args):
        model.eval()
        output = model(torch.hstack(args))
        return output

    def evaluate_dirichlet(self, model, x, y):
        return self.evaluate(model, x, y)

    def evaluate_neumann(self, model, x, y, dir_x, dir_y):
        output = self.evaluate(model, x, y)

        dx = torch.autograd.grad(
            output, x,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        dy = torch.autograd.grad(
            output, y,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        return dir_x * dx + dir_y * dy

    def evaluate_robin(self, model, x, y, dir_x, dir_y, mat, alf, val):
        output = self.evaluate(model, x, y)

        dx = torch.autograd.grad(
            output, x,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        dy = torch.autograd.grad(
            output, y,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        return mat * (dir_x * dx + dir_y * dy) - alf * (val - output)

    def evaluate_transition(self, model, x, y):
        return self.evaluate(model, x, y)

    def evaluate_transition_dx(self, model, x, y, dir_x, dir_y, mat):
        output = self.evaluate(model, x, y)

        dx = torch.autograd.grad(
            output, x,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        dy = torch.autograd.grad(
            output, y,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        return mat * (dir_x * dx + dir_y * dy)

    def evaluate_equation(self, model, x, y, mat, matdif_x, matdif_y):
        output = self.evaluate(model, x, y)

        dx = torch.autograd.grad(
            output, x,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        dxx = torch.autograd.grad(
            dx, x,
            grad_outputs=torch.ones_like(dx),
            retain_graph=True,
            create_graph=True)[0]

        dy = torch.autograd.grad(
            output, y,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        dyy = torch.autograd.grad(
            dy, y,
            grad_outputs=torch.ones_like(dy),
            retain_graph=True,
            create_graph=True)[0]

        return mat * (dxx + dyy) + matdif_x * dx + matdif_y * dy

    def closure(self):
        self.optimizer.zero_grad()

        eq_pred_L = self.evaluate_equation(self.model_L, self.i_x_L, self.i_y_L, self.i_mat_L, self.i_mdf_x_L,
                                           self.i_mdf_y_L)
        d_pred_L = self.evaluate_dirichlet(self.model_L, self.d_x_L, self.d_y_L)
        # n_pred_L = self.evaluate_neumann(self.model_L, self.n_x_L, self.n_y_L, self.n_dir_x_L, self.n_dir_y_L)
        r_pred_L = self.evaluate_robin(self.model_L, self.r_x_L, self.r_y_L, self.r_dir_x_L, self.r_dir_y_L,
                                       self.r_mat_L, self.r_alf_L, self.r_val_L)
        t_pred_L = self.evaluate_transition(self.model_L, self.t_x, self.t_y)
        t_dx_pred_L = self.evaluate_transition_dx(self.model_L, self.t_x, self.t_y, self.t_dir_x, self.t_dir_y,
                                                  self.t_mat_L)

        eq_pred_R = self.evaluate_equation(self.model_R, self.i_x_R, self.i_y_R, self.i_mat_R, self.i_mdf_x_R,
                                           self.i_mdf_y_R)
        d_pred_R = self.evaluate_dirichlet(self.model_R, self.d_x_R, self.d_y_R)
        # n_pred_R = self.evaluate_neumann(self.model_R, self.n_x_R, self.n_y_R, self.n_dir_x_R, self.n_dir_y_R)
        r_pred_R = self.evaluate_robin(self.model_R, self.r_x_R, self.r_y_R, self.r_dir_x_R, self.r_dir_y_R,
                                       self.r_mat_R, self.r_alf_R, self.r_val_R)
        t_pred_R = self.evaluate_transition(self.model_R, self.t_x, self.t_y)
        t_dx_pred_R = self.evaluate_transition_dx(self.model_R, self.t_x, self.t_y, self.t_dir_x, self.t_dir_y,
                                                  self.t_mat_R)

        eq_loss_L = self.eq_loss_alf * self.loss_function(eq_pred_L, self.null_eq_L)
        d_loss_L = self.d_loss_alf * self.loss_function(d_pred_L, self.d_val_L)
        # n_loss_L = self.n_loss_alf * self.loss_function(n_pred_L, self.n_val_L)
        r_loss_L = self.r_loss_alf * self.loss_function(r_pred_L, self.null_r_L)

        eq_loss_R = self.eq_loss_alf * self.loss_function(eq_pred_R, self.null_eq_R)
        d_loss_R = self.d_loss_alf * self.loss_function(d_pred_R, self.d_val_R)
        # n_loss_R = self.n_loss_alf * self.loss_function(n_pred_R, self.n_val_R)
        r_loss_R = self.r_loss_alf * self.loss_function(r_pred_R, self.null_r_R)

        t_loss = self.t_loss_alf * self.loss_function(t_pred_L - t_pred_R, self.null_t)
        t_dx_loss = self.t_loss_alf * self.loss_function(t_dx_pred_L - t_dx_pred_R, self.null_t)

        self.loss = eq_loss_L + d_loss_L + r_loss_L + eq_loss_R + d_loss_R + r_loss_R + t_loss + t_dx_loss  # + n_loss_L + n_loss_R
        self.loss.backward()

        self.iterations += 1

        if not self.iterations % 100:
            print('Iterations: {0:}, Loss: {1:6.10f}'.format(self.iterations, self.loss))
            print('L-EQ: ', eq_loss_L.item())
            print('L-DI: ', d_loss_L.item())
            # print('L-NE: ', n_loss_L.item())
            print('L-RO: ', r_loss_L.item())
            print('R-EQ: ', eq_loss_R.item())
            print('R-DI: ', d_loss_R.item())
            # print('R-NE: ', n_loss_R.item())
            print('R-RO: ', r_loss_R.item())
            print('T-EQ_VAL: ', t_loss.item())
            print('T-EQ_DIF: ', t_dx_loss.item())

        return self.loss

    def train(self):
        self.model_L.train()  # only sets a flag
        self.model_R.train()  # only sets a flag
        self.optimizer.step(self.closure)

    def plot_trained_function(self, title, filename, loc_L, loc_R):
        nn_input_x_L = torch.tensor(loc_L[0].reshape(-1, 1), dtype=torch.float32)
        nn_input_y_L = torch.tensor(loc_L[1].reshape(-1, 1), dtype=torch.float32)

        z_values_L = self.evaluate(self.model_L, nn_input_x_L, nn_input_y_L)
        z_values_L = z_values_L.detach().numpy().reshape(-1)

        nn_input_x_R = torch.tensor(loc_R[0].reshape(-1, 1), dtype=torch.float32)
        nn_input_y_R = torch.tensor(loc_R[1].reshape(-1, 1), dtype=torch.float32)

        z_values_R = self.evaluate(self.model_R, nn_input_x_R, nn_input_y_R)
        z_values_R = z_values_R.detach().numpy().reshape(-1)

        X = np.append(nn_input_x_L.numpy().reshape(-1), nn_input_x_R.numpy().reshape(-1))
        Y = np.append(nn_input_y_L.numpy().reshape(-1), nn_input_y_R.numpy().reshape(-1))

        coordinates = np.array([Y, X]).transpose()

        z_values = np.append(z_values_L, z_values_R)

        display_heatmap(coordinates, z_values, title, filename)

    def plot_equation_loss(self, title, filename):
        X = np.append(self.i_x_L.detach().numpy().reshape(-1), self.i_x_R.detach().numpy().reshape(-1))
        Y = np.append(self.i_y_L.detach().numpy().reshape(-1), self.i_y_R.detach().numpy().reshape(-1))

        values_L = self.evaluate_equation(self.model_L, self.i_x_L, self.i_y_L, self.i_mat_L, self.i_mdf_x_L, self.i_mdf_y_L).detach().numpy().reshape(-1)
        values_R = self.evaluate_equation(self.model_R, self.i_x_R, self.i_y_R, self.i_mat_R, self.i_mdf_x_R, self.i_mdf_y_R).detach().numpy().reshape(-1)

        coordinates = np.array([Y, X]).transpose()
        values = np.abs(np.append(values_L, values_R))

        display_heatmap(coordinates, values, title, filename)


def plot_2d(x_mesh, y_mesh, z_values, title, filename):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    ax.set_title(title)

    h = ax.imshow(z_values,
                  interpolation='nearest',
                  cmap='rainbow',
                  extent=[y_mesh.min(), y_mesh.max(), x_mesh.min(), x_mesh.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(filename)


def display_heatmap(coordinates, values, title, filename):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.interpolate import griddata

    xmin = min(coordinates[:, 0])
    xmax = max(coordinates[:, 0])
    ymin = min(coordinates[:, 1])
    ymax = max(coordinates[:, 1])

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, 1000),
        np.linspace(ymin, ymax, 1000),
        indexing='xy'
    )
    grid_c = griddata(coordinates, values, (grid_x, grid_y), method='nearest')

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.set_title(title)

    h = ax.imshow(grid_c.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='rainbow', aspect='auto')
    # h = ax.imshow(grid_c.T, origin='lower', cmap='rainbow',)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=10)

    # ax.legend(bbox_to_anchor=(1.1, 1), loc='center', borderaxespad=2)
    plt.savefig(filename)


if __name__ == '__main__':
    print('Program started.')

    # problem information
    material_rects = [4, 4]  # TODO - neměnit, rozbije se pak samplovani na hranici(ích) (to by se taky mozna dalo zobecnit)
    material_coefs = [1e-2, 1]

    dirichlet_const_A = 293.15 / 530
    dirichlet_const_B = 1.0
    neumann_const = 0
    robin_const = 500.0 / 530
    robin_alpha = 12

    # domain preparation
    boundaries_omega = [[], []]
    for i in range(material_rects[0]):
        for j in range(material_rects[1]):
            x_min = i / material_rects[0]
            x_max = (i + 1) / material_rects[0]
            y_min = j / material_rects[1]
            y_max = (j + 1) / material_rects[1]
            boundary = Boundary([[x_min, x_max, x_max, x_min],
                                 [y_min, y_min, y_max, y_max]])
            boundaries_omega[(i + j) % 2].append(boundary)

    ds_omega = [DomainSampler(boundaries_omega[0]), DomainSampler(boundaries_omega[1])]

    ds_omega[1].boundary_group_add('neumann', 6, [0, 1], connect_ends=False, color='orange')
    ds_omega[0].boundary_group_add('neumann', 4, [0, 1], connect_ends=False, color='orange')
    ds_omega[1].boundary_group_add('neumann', 2, [0, 1], connect_ends=False, color='orange')
    ds_omega[0].boundary_group_add('neumann', 0, [0, 1], connect_ends=False, color='orange')

    ds_omega[0].boundary_group_add('robin', 7, [1, 2], connect_ends=False, color='red')
    ds_omega[1].boundary_group_add('robin', 7, [1, 2], connect_ends=False, color='red')
    ds_omega[0].boundary_group_add('robin', 6, [1, 2], connect_ends=False, color='red')
    ds_omega[1].boundary_group_add('robin', 6, [1, 2], connect_ends=False, color='red')

    ds_omega[1].boundary_group_add('dirichlet_B', 1, [2, 3], connect_ends=False, color='aqua')
    ds_omega[0].boundary_group_add('dirichlet_B', 3, [2, 3], connect_ends=False, color='aqua')
    ds_omega[1].boundary_group_add('dirichlet_B', 5, [2, 3], connect_ends=False, color='aqua')
    ds_omega[0].boundary_group_add('dirichlet_B', 7, [2, 3], connect_ends=False, color='aqua')

    ds_omega[0].boundary_group_add('dirichlet_A', 0, [3, 0], connect_ends=False, color='blue')
    ds_omega[1].boundary_group_add('dirichlet_A', 0, [3, 0], connect_ends=False, color='blue')
    ds_omega[0].boundary_group_add('dirichlet_A', 1, [3, 0], connect_ends=False, color='blue')
    ds_omega[1].boundary_group_add('dirichlet_A', 1, [3, 0], connect_ends=False, color='blue')

    for ds in ds_omega:
        ds.distribute_line_probe(n_y=40)
        ds.boundary_groups_distribute()

    ds_bound = DomainSampler([Boundary([[0, 1, 1, 0],
                                        [0, 0, 1, 1]])])
    ds_bound.boundary_group_add('horizontal', 0, [0, 1], connect_ends=False, color='green')
    ds_bound.boundary_group_add('vertical', 0, [3, 0], connect_ends=False, color='black')
    ds_bound.boundary_groups_distribute()

    i_count = 300
    d_count = 100
    n_count = 200
    r_count = 200
    t_count = 150

    i_loc = [[], []]
    i_mat = [[], []]
    i_mdf = [[], []]
    d_loc = [[], []]
    d_val = [[], []]
    n_loc = [[], []]
    n_val = [[], []]
    n_dir = [[], []]
    r_loc = [[], []]
    r_mat = [[], []]
    r_val = [[], []]
    r_alf = [[], []]
    r_dir = [[], []]
    t_loc = np.array([[], []])
    t_dir = np.array([[], []])
    # t_mat

    for k in range(2):
        # generate interior samples
        i_loc[k] = np.array(ds_omega[k].sample_interior(i_count))
        i_mat[k] = np.ones_like(i_loc[k][0]) * material_coefs[k]
        i_mdf[k] = np.array([np.zeros_like(i_mat[k]), np.zeros_like(i_mat[k])])

        # generate dirichlet samples
        d_loc_A = ds_omega[k].sample_boundary(d_count, label='dirichlet_A')
        d_loc_B = ds_omega[k].sample_boundary(d_count, label='dirichlet_B')
        d_loc[k] = np.array([d_loc_A[0] + d_loc_B[0], d_loc_A[1] + d_loc_B[1]])
        d_val[k] = np.append(dirichlet_const_A * np.ones_like(d_loc_A[0]), dirichlet_const_B * np.ones_like(d_loc_A[0]))

        # generate neumann samples
        n_loc[k] = np.array(ds_omega[k].sample_boundary(n_count, label='neumann'))
        n_val[k] = neumann_const * np.ones_like(n_loc[k][0])
        n_dir[k] = np.array([np.zeros_like(n_val[k]), np.ones_like(n_val[k])])

        # generate robin samples
        r_loc[k] = np.array(ds_omega[k].sample_boundary(r_count, label='robin'))
        r_mat[k] = np.ones_like(r_loc[k][0]) * material_coefs[k]
        r_val[k] = np.ones_like(r_mat[k]) * robin_const
        r_alf[k] = np.ones_like(r_mat[k]) * robin_alpha
        r_dir[k] = np.array([-1 * np.ones_like(r_mat[k]), np.zeros_like(r_mat[k])])

    # generate transition samples
    for i in range(1, material_rects[0]):
        t_hor = np.array(ds_bound.sample_boundary(t_count, label='horizontal'))
        t_hor[1] += i / material_rects[0]
        t_dir_hor = np.array([np.zeros_like(t_hor[0]), np.ones_like(t_hor[0])])

        t_loc = np.append(t_loc, t_hor, axis=1)
        t_dir = np.append(t_dir, t_dir_hor, axis=1)

    for i in range(1, material_rects[1]):
        t_ver = np.array(ds_bound.sample_boundary(t_count, label='vertical'))
        t_ver[0] += i / material_rects[1]
        t_dir_ver = np.array([np.ones_like(t_ver[0]), np.zeros_like(t_ver[0])])

        t_loc = np.append(t_loc, t_ver, axis=1)
        t_dir = np.append(t_dir, t_dir_ver, axis=1)

    t_mat = np.array([np.ones_like(t_loc[0]) * material_coefs[0], np.ones_like(t_loc[0]) * material_coefs[1]])

    print('Data preparation complete.')

    # uncomment to check for right sampling
    for k in range(2):
        ds_omega[k].plot_domain()
        ds_omega[k].plot_distribution_interior()
        ds_omega[k].plot_distribution_boundary()
        plt.plot(i_loc[k][0], i_loc[k][1], marker='+', color='black', linestyle='None')
        plt.plot(d_loc[k][0], d_loc[k][1], marker='+', color='blue', linestyle='None')
        plt.plot(n_loc[k][0], n_loc[k][1], marker='+', color='orange', linestyle='None')
        plt.plot(r_loc[k][0], r_loc[k][1], marker='+', color='red', linestyle='None')
        plt.show()

    ds_omega[0].plot_domain()
    ds_omega[1].plot_domain()
    plt.plot(t_loc[0], t_loc[1], marker='+', color='black', linestyle='None')
    plt.show()

    # create PINN
    pinn = PINN(i_loc, i_mat, i_mdf, d_loc, d_val, n_loc, n_val, n_dir, r_loc, r_mat, r_val, r_alf, r_dir, t_loc, t_dir, t_mat)

    print('PINN created.')

    # train PINN
    inp = ''
    while inp != 'q':
        for r in range(500):
            pinn.train()
        inp = input()

    print('PINN trained.')

    # compute locations for visualisation
    nudge = 1e-4
    loc_L = np.array([[], []])
    loc_R = np.array([[], []])
    for i in range(material_rects[0]):
        for j in range(material_rects[1]):
            x_min = i / material_rects[0]
            x_max = (i + 1) / material_rects[0]
            y_min = j / material_rects[1]
            y_max = (j + 1) / material_rects[1]

            xx = np.linspace(x_min + nudge, x_max - nudge, 15)
            yy = np.linspace(y_min + nudge, y_max - nudge, 15)

            xx_mesh, yy_mesh = np.meshgrid(xx, yy)
            xx_mesh = xx_mesh.reshape(-1)
            yy_mesh = yy_mesh.reshape(-1)

            coords = np.array([xx_mesh, yy_mesh])

            if (i + j) % 2 == 0:
                loc_L = np.append(loc_L, coords, axis=1)
            else:
                loc_R = np.append(loc_R, coords, axis=1)

    # plt.plot(loc_L[0], loc_L[1], marker='+', color='red', linestyle='None')
    # plt.plot(loc_R[0], loc_R[1], marker='+', color='green', linestyle='None')
    # plt.show()

    # visualize results
    pinn.plot_equation_loss('Equation loss', 'pde_loss.png')
    pinn.plot_trained_function('Trained function', 'func.png', loc_L, loc_R)

    print('Data plotted.')
