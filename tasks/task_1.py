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
        #self.linear_middle3 = nn.Linear(30, 30)
        #self.linear_middle4 = nn.Linear(30, 30)
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
        #x = self.linear_middle3(x)
        #x = self.tangent(x)
        #x = self.linear_middle4(x)
        #x = self.tangent(x)
        x = self.linear_end(x)
        return x


class PINN:
    def __init__(self, loc, dirichlet_loc, dirichlet_samples):
        self.loc_x = torch.tensor(loc[0].reshape(-1, 1),
                                  dtype=torch.float32,
                                  requires_grad=True)
        self.loc_y = torch.tensor(loc[1].reshape(-1, 1),
                                  dtype=torch.float32,
                                  requires_grad=True)

        self.dirichlet_x = torch.tensor(dirichlet_loc[0].reshape(-1, 1),
                                        dtype=torch.float32,
                                        requires_grad=True)
        self.dirichlet_y = torch.tensor(dirichlet_loc[1].reshape(-1, 1),
                                        dtype=torch.float32,
                                        requires_grad=True)
        self.dirichlet_samples = torch.tensor(dirichlet_samples.reshape(-1, 1),
                                              dtype=torch.float32)

        self.null = torch.zeros((self.loc_x.shape[0], 1))

        self.model = NetworkModel()

        self.loss_function = nn.MSELoss()

        self.loss = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.01,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=0,
                                          amsgrad=True)

        self.iterations = 0

        self.alpha_equation = 1
        self.alpha_dirichlet = 1

    def evaluate(self, *args):
        self.model.eval()
        output = self.model(torch.hstack(args))
        return output

    def evaluate_dirichlet(self):
        return self.evaluate(self.dirichlet_x, self.dirichlet_y)

    def evaluate_equation(self):
        output = self.evaluate(self.loc_x, self.loc_y)

        dx = torch.autograd.grad(
            output, self.loc_x,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        dxx = torch.autograd.grad(
            dx, self.loc_x,
            grad_outputs=torch.ones_like(dx),
            retain_graph=True,
            create_graph=True)[0]

        dy = torch.autograd.grad(
            output, self.loc_y,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        dyy = torch.autograd.grad(
            dy, self.loc_y,
            grad_outputs=torch.ones_like(dy),
            retain_graph=True,
            create_graph=True)[0]

        return dxx + dyy

    def closure(self):
        self.optimizer.zero_grad()

        equation_prediction = self.evaluate_equation()
        dirichlet_prediction = self.evaluate_dirichlet()

        equation_loss = self.alpha_equation * self.loss_function(equation_prediction, self.null)
        dirichlet_loss = self.alpha_dirichlet * self.loss_function(dirichlet_prediction, self.dirichlet_samples)

        self.loss = equation_loss + dirichlet_loss
        self.loss.backward()

        self.iterations += 1

        if not self.iterations % 100:
            print('Iterations: {0:}, Loss: {1:6.10f}'.format(self.iterations, self.loss))
            print('EQ: ', equation_loss.item() / self.alpha_equation)
            print('DI: ', dirichlet_loss.item() / self.alpha_dirichlet)

        return self.loss

    def train(self):
        self.model.train()  # only sets a flag
        self.optimizer.step(self.closure)

    def plot_trained_function(self, title, filename, x_min=0, x_max=1, y_min=0, y_max=1, steps=50):
        x_linspace = np.linspace(x_min, x_max, steps)
        y_linspace = np.linspace(y_min, y_max, steps)

        x_mesh, y_mesh = np.meshgrid(x_linspace, y_linspace)
        x_mesh = x_mesh.reshape(-1, 1)
        y_mesh = y_mesh.reshape(-1, 1)

        nn_input_x = torch.tensor(x_mesh.reshape(-1, 1),
                                  dtype=torch.float32)
        nn_input_y = torch.tensor(y_mesh.reshape(-1, 1),
                                  dtype=torch.float32)

        z_values = self.evaluate(nn_input_x, nn_input_y)
        z_values = z_values.detach().numpy().reshape(-1)

        coordinates = np.array([nn_input_y.numpy().reshape(-1),
                                nn_input_x.numpy().reshape(-1)]).transpose()

        display_heatmap(coordinates, z_values, title, filename)

    def plot_equation_loss(self, title, filename):
        coordinates = np.array([self.loc_y.detach().numpy().reshape(-1),
                                self.loc_x.detach().numpy().reshape(-1)]).transpose()
        values = np.abs(self.evaluate_equation().detach().numpy().transpose()[0]) ** 2

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
    a = 3
    b = 2

    x_min = 0
    x_max = 2
    y_min = 0
    y_max = 1

    # domain preparation
    ds_bound = DomainSampler([Boundary([[x_min, x_max, x_max, x_min],
                                        [y_min, y_min, y_max, y_max]])])

    ds_bound.boundary_group_add('dirichlet', 0, [0, 1, 2, 3], connect_ends=True, color='blue')

    ds_bound.distribute_line_probe(n_y=40)
    ds_bound.boundary_groups_distribute()

    # generate interior samples
    i_count = 200

    i_loc = np.array(ds_bound.sample_interior(i_count))

    # generate dirichlet samples
    d_count = 200

    d_loc = np.array(ds_bound.sample_boundary(d_count, label='dirichlet'))
    # d_samples = a * d_loc[0] + b
    # d_samples = a * (d_loc[0] ** 2 - d_loc[1] ** 2)
    d_samples = a * np.exp(b * d_loc[1]) * np.sin(b * d_loc[0])

    # uncomment to check for right sampling
    # ds_bound.plot_domain()
    # ds_bound.plot_distribution_interior()
    # plt.plot(i_loc[0], i_loc[1], 'm+')
    # plt.show()
    # ds_bound.plot_domain()
    # ds_bound.plot_distribution_boundary()
    # plt.plot(d_loc[0], d_loc[1], 'm+')
    # plt.show()

    # create PINN
    pinn = PINN(i_loc, d_loc, d_samples)

    # train PINN
    inp = ''
    while inp != 'q':
        for r in range(500):
            pinn.train()
        inp = input()

    # visualize results
    pinn.plot_equation_loss('Equation loss', 'pde_loss.png')
    pinn.plot_trained_function('Trained function', 'func.png',
                               x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
