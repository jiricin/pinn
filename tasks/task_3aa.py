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
        x = self.linear_end(x)
        return x


class PINN:
    def __init__(self,
                 loc_L, dirichlet_loc_L, dirichlet_samples_L,
                 loc_R, dirichlet_loc_R, dirichlet_samples_R,
                 transition_loc, material):
        self.loc_x_L = torch.tensor(loc_L[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.loc_y_L = torch.tensor(loc_L[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.dirichlet_x_L = torch.tensor(dirichlet_loc_L[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.dirichlet_y_L = torch.tensor(dirichlet_loc_L[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.dirichlet_samples_L = torch.tensor(dirichlet_samples_L.reshape(-1, 1), dtype=torch.float32)

        self.loc_x_R = torch.tensor(loc_R[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.loc_y_R = torch.tensor(loc_R[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.dirichlet_x_R = torch.tensor(dirichlet_loc_R[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.dirichlet_y_R = torch.tensor(dirichlet_loc_R[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.dirichlet_samples_R = torch.tensor(dirichlet_samples_R.reshape(-1, 1), dtype=torch.float32)

        self.transition_x = torch.tensor(transition_loc[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.transition_y = torch.tensor(transition_loc[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)

        self.material_L = material[0]
        self.material_R = material[1]

        self.null_equation_L = torch.zeros((self.loc_x_L.shape[0], 1))
        self.null_equation_R = torch.zeros((self.loc_x_R.shape[0], 1))
        self.null_transition = torch.zeros((self.transition_x.shape[0], 1))

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

        self.alpha_equation = 1
        self.alpha_dirichlet = 1
        self.alpha_transition = 1

    def evaluate(self, model, *args):
        model.eval()
        output = model(torch.hstack(args))
        return output

    def evaluate_dirichlet(self, model, x, y):
        return self.evaluate(model, x, y)

    def evaluate_transition(self, model, x, y):
        return self.evaluate(model, x, y)

    def evaluate_transition_dx(self, model, x, y, material):
        output = self.evaluate(model, x, y)

        dx = torch.autograd.grad(
            output, x,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        return material * dx

    def evaluate_equation(self, model, x, y, material):
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

        return material * (dxx + dyy)

    def closure(self):
        self.optimizer.zero_grad()

        equation_prediction_L = self.evaluate_equation(self.model_L, self.loc_x_L, self.loc_y_L, self.material_L)
        dirichlet_prediction_L = self.evaluate_dirichlet(self.model_L, self.dirichlet_x_L, self.dirichlet_y_L)
        transition_prediction_L = self.evaluate_transition(self.model_L, self.transition_x, self.transition_y)
        transition_dx_prediction_L = self.evaluate_transition_dx(self.model_L, self.transition_x, self.transition_y, self.material_L)
        equation_prediction_R = self.evaluate_equation(self.model_R, self.loc_x_R, self.loc_y_R, self.material_R)
        dirichlet_prediction_R = self.evaluate_dirichlet(self.model_R, self.dirichlet_x_R, self.dirichlet_y_R)
        transition_prediction_R = self.evaluate_transition(self.model_R, self.transition_x, self.transition_y)
        transition_dx_prediction_R = self.evaluate_transition_dx(self.model_R, self.transition_x, self.transition_y, self.material_R)

        equation_loss_L = self.alpha_equation * self.loss_function(equation_prediction_L, self.null_equation_L)
        dirichlet_loss_L = self.alpha_dirichlet * self.loss_function(dirichlet_prediction_L, self.dirichlet_samples_L)
        equation_loss_R = self.alpha_equation * self.loss_function(equation_prediction_R, self.null_equation_R)
        dirichlet_loss_R = self.alpha_dirichlet * self.loss_function(dirichlet_prediction_R, self.dirichlet_samples_R)
        transition_loss = self.alpha_transition * self.loss_function(transition_prediction_L - transition_prediction_R, self.null_transition)
        transition_dx_loss = self.alpha_transition * self.loss_function(transition_dx_prediction_L - transition_dx_prediction_R, self.null_transition)

        self.loss = equation_loss_L + dirichlet_loss_L + equation_loss_R + dirichlet_loss_R + transition_loss + transition_dx_loss
        self.loss.backward()

        self.iterations += 1

        if not self.iterations % 100:
            print('Iterations: {0:}, Loss: {1:6.10f}'.format(self.iterations, self.loss))
            print('L-EQ: ', equation_loss_L.item() / self.alpha_equation)
            print('L-DI: ', dirichlet_loss_L.item() / self.alpha_dirichlet)
            print('R-EQ: ', equation_loss_R.item() / self.alpha_equation)
            print('R-DI: ', dirichlet_loss_R.item() / self.alpha_dirichlet)
            print('T-EQ_VAL: ', transition_loss.item() / self.alpha_transition)
            print('T-EQ_DIF: ', transition_dx_loss.item() / self.alpha_transition)

        return self.loss

    def train(self):
        self.model_L.train()  # only sets a flag
        self.model_R.train()  # only sets a flag
        self.optimizer.step(self.closure)

    # TODO for other domains just change it here
    def plot_trained_function(self, title, filename, steps=25):
        x_linspace_L = np.linspace(0, 1, steps)
        y_linspace_L = np.linspace(0, 1, steps)

        x_mesh_L, y_mesh_L = np.meshgrid(x_linspace_L, y_linspace_L)
        x_mesh_L = x_mesh_L.reshape(-1, 1)
        y_mesh_L = y_mesh_L.reshape(-1, 1)

        nn_input_x_L = torch.tensor(x_mesh_L.reshape(-1, 1), dtype=torch.float32)
        nn_input_y_L = torch.tensor(y_mesh_L.reshape(-1, 1), dtype=torch.float32)

        z_values_L = self.evaluate(self.model_L, nn_input_x_L, nn_input_y_L)
        z_values_L = z_values_L.detach().numpy().reshape(-1)

        x_linspace_R = np.linspace(1, 2, steps)
        y_linspace_R = np.linspace(0, 1, steps)

        x_mesh_R, y_mesh_R = np.meshgrid(x_linspace_R, y_linspace_R)
        x_mesh_R = x_mesh_R.reshape(-1, 1)
        y_mesh_R = y_mesh_R.reshape(-1, 1)

        nn_input_x_R = torch.tensor(x_mesh_R.reshape(-1, 1), dtype=torch.float32)
        nn_input_y_R = torch.tensor(y_mesh_R.reshape(-1, 1), dtype=torch.float32)

        z_values_R = self.evaluate(self.model_R, nn_input_x_R, nn_input_y_R)
        z_values_R = z_values_R.detach().numpy().reshape(-1)

        X = np.append(nn_input_x_L.numpy().reshape(-1), nn_input_x_R.numpy().reshape(-1))
        Y = np.append(nn_input_y_L.numpy().reshape(-1), nn_input_y_R.numpy().reshape(-1))

        coordinates = np.array([Y, X]).transpose()

        z_values = np.append(z_values_L, z_values_R)

        display_heatmap(coordinates, z_values, title, filename)

    def plot_equation_loss(self, title, filename):
        X = np.append(self.loc_x_L.detach().numpy().reshape(-1), self.loc_x_R.detach().numpy().reshape(-1))
        Y = np.append(self.loc_y_L.detach().numpy().reshape(-1), self.loc_y_R.detach().numpy().reshape(-1))

        values_L = self.evaluate_equation(self.model_L, self.loc_x_L, self.loc_y_L).detach().numpy().reshape(-1)
        values_R = self.evaluate_equation(self.model_R, self.loc_x_R, self.loc_y_R).detach().numpy().reshape(-1)

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
    # problem information
    x_min = 0
    x_max = 2
    y_min = 0
    y_max = 1

    material = [0.25, 1]

    # domain preparation
    ds_left = DomainSampler([Boundary([[0, 1, 1, 0],
                                        [0, 0, 1, 1]])])
    ds_right = DomainSampler([Boundary([[1, 2, 2, 1],
                                       [0, 0, 1, 1]])])

    ds_left.boundary_group_add('dirichlet', 0, [2, 3, 0, 1], connect_ends=False, color='blue')
    ds_left.boundary_group_add('transition', 0, [1, 2], connect_ends=False, color='green')
    ds_right.boundary_group_add('dirichlet', 0, [0, 1, 2, 3], connect_ends=False, color='blue')

    ds_left.distribute_line_probe(n_y=40)
    ds_left.boundary_groups_distribute()
    ds_right.distribute_line_probe(n_y=40)
    ds_right.boundary_groups_distribute()

    # generate interior samples
    i_count = 100

    i_loc_L = np.array(ds_left.sample_interior(i_count))
    i_loc_R = np.array(ds_right.sample_interior(i_count))

    # generate dirichlet samples
    d_count = 100

    d_loc_L = np.array(ds_left.sample_boundary(d_count, label='dirichlet'))
    d_loc_R = np.array(ds_right.sample_boundary(d_count, label='dirichlet'))
    d_samples_L = np.zeros_like(d_loc_L[0])
    d_samples_R = (d_loc_R[0] - 1) ** 3

    # generate transition samples
    t_count = 100

    t_loc = np.array(ds_left.sample_boundary(t_count, label='transition'))

    # uncomment to check for right sampling
    # ds_left.plot_domain()
    # ds_left.plot_distribution_interior()
    # ds_right.plot_domain()
    # ds_right.plot_distribution_interior()
    # plt.plot(i_loc_L[0], i_loc_L[1], 'm+')
    # plt.plot(i_loc_R[0], i_loc_R[1], 'm+')
    # # # plt.show()
    # ds_left.plot_domain()
    # ds_left.plot_distribution_boundary()
    # ds_right.plot_domain()
    # ds_right.plot_distribution_boundary()
    # plt.plot(d_loc_L[0], d_loc_L[1], 'm+')
    # plt.plot(d_loc_R[0], d_loc_R[1], 'm+')
    # plt.plot(t_loc[0], t_loc[1], 'm+')
    # plt.show()

    # create PINN
    pinn = PINN(i_loc_L, d_loc_L, d_samples_L, i_loc_R, d_loc_R, d_samples_R, t_loc, material)

    # train PINN
    inp = ''
    while inp != 'q':
        for r in range(500):
            pinn.train()
        inp = input()

    # visualize results
    pinn.plot_equation_loss('Equation loss', 'pde_loss.png')
    pinn.plot_trained_function('Trained function', 'func.png')
