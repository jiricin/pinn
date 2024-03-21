from DomainSampler import DomainSampler, Boundary
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


class NetworkModel(nn.Module):
    def __init__(self):
        super(NetworkModel, self).__init__()

        self.linear_start = nn.Linear(2, 30)
        self.linear_middle1 = nn.Linear(30, 30)
        self.linear_middle2 = nn.Linear(30, 30)
        self.linear_middle3 = nn.Linear(30, 30)
        self.linear_end = nn.Linear(30, 1)
        self.tangent = nn.Tanh()

    def forward(self, y):
        y = self.linear_start(y)

        # fourier features
        x1 = torch.sin(y[:, 0:15])
        x2 = torch.cos(y[:, 15:30])
        x = torch.cat((x1, x2), dim=1)

        x = self.linear_middle1(x)
        x = self.tangent(x)
        x = self.linear_middle2(x)
        x = self.tangent(x)
        x = self.linear_middle3(x)
        x = self.tangent(x)
        x = self.linear_end(x)
        return x


class PINN:
    def __init__(self, loc, loc_samples, dirichlet_loc, dirichlet_samples, neumann_loc, neumann_samples, robin_loc,
                 robin_samples, robin_material, robin_alpha):
        self.loc_x = torch.tensor(loc[0].reshape(-1, 1),
                                  dtype=torch.float32,
                                  requires_grad=True)
        self.loc_y = torch.tensor(loc[1].reshape(-1, 1),
                                  dtype=torch.float32,
                                  requires_grad=True)
        self.material = torch.tensor(loc_samples.reshape(-1, 1),
                                     dtype=torch.float32)

        self.dirichlet_x = torch.tensor(dirichlet_loc[0].reshape(-1, 1),
                                        dtype=torch.float32,
                                        requires_grad=True)
        self.dirichlet_y = torch.tensor(dirichlet_loc[1].reshape(-1, 1),
                                        dtype=torch.float32,
                                        requires_grad=True)
        self.dirichlet_samples = torch.tensor(dirichlet_samples.reshape(-1, 1),
                                              dtype=torch.float32)

        self.neumann_x = torch.tensor(neumann_loc[0].reshape(-1, 1),
                                      dtype=torch.float32,
                                      requires_grad=True)
        self.neumann_y = torch.tensor(neumann_loc[1].reshape(-1, 1),
                                      dtype=torch.float32,
                                      requires_grad=True)
        self.neumann_samples = torch.tensor(neumann_samples.reshape(-1, 1),
                                            dtype=torch.float32)

        self.robin_x = torch.tensor(robin_loc[0].reshape(-1, 1),
                                    dtype=torch.float32,
                                    requires_grad=True)
        self.robin_y = torch.tensor(robin_loc[1].reshape(-1, 1),
                                    dtype=torch.float32,
                                    requires_grad=True)
        self.robin_samples = torch.tensor(robin_samples.reshape(-1, 1),
                                          dtype=torch.float32)

        self.robin_material = torch.tensor(robin_material.reshape(-1, 1),
                                           dtype=torch.float32)

        self.robin_alpha = robin_alpha

        self.null = torch.zeros((self.material.shape[0], 1))

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
        self.alpha_neumann = 1
        self.alpha_robin = 1

    def evaluate(self, *args):
        self.model.eval()
        output = self.model(torch.hstack(args))
        return output

    def evaluate_dirichlet(self):
        return self.evaluate(self.dirichlet_x, self.dirichlet_y)

    def evaluate_neumann(self):
        output = self.evaluate(self.neumann_x, self.neumann_y)

        # differential with respect to normal - in this case (0, 1)
        dy = torch.autograd.grad(
            output, self.neumann_y,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        return dy

    def evaluate_robin(self):
        output = self.evaluate(self.robin_x, self.robin_y)

        # differential with respect to normal - in this case (-1, 0)
        dx = torch.autograd.grad(
            output, self.robin_x,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        return self.robin_material * (-dx) - self.robin_alpha * (self.robin_samples - output)

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

        return self.material * (dxx + dyy)

    def closure(self):
        self.optimizer.zero_grad()

        equation_prediction = self.evaluate_equation()
        dirichlet_prediction = self.evaluate_dirichlet()
        # neumann_prediction = self.evaluate_neumann()
        robin_prediction = self.evaluate_robin()

        equation_loss = self.alpha_equation * self.loss_function(equation_prediction, self.null)
        dirichlet_loss = self.alpha_dirichlet * self.loss_function(dirichlet_prediction, self.dirichlet_samples)
        # neumann_loss = self.alpha_neumann * self.loss_function(neumann_prediction, self.neumann_samples)
        robin_loss = self.alpha_robin * self.loss_function(robin_prediction, self.robin_samples)

        self.loss = equation_loss + dirichlet_loss + robin_loss  # + neumann_loss
        self.loss.backward()

        self.iterations += 1

        if not self.iterations % 100:
            print('Iterations: {0:}, Loss: {1:6.10f}'.format(self.iterations, self.loss))
            print('EQ: ', equation_loss.item())
            print('DI: ', dirichlet_loss.item())
            # print('NE: ', neumann_loss.item())
            print('RO: ', robin_loss.item())

        return self.loss

    def train(self):
        self.model.train()  # only sets a flag
        self.optimizer.step(self.closure)

    def plot_trained_function(self, title, filename, x_min=0, x_max=1, y_min=0, y_max=1, steps=50):
        # print('Total iterations: {0:}, Final loss: {1:6.10f}'.format(self.iterations, self.loss))
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
        z_values = z_values.detach().numpy()
        z_values = z_values.reshape(steps, steps)

        plot_2d(x_mesh, y_mesh, z_values, title, filename)

    def plot_equation_loss(self, title, filename):
        coordinates = np.array([self.loc_x.detach().numpy().transpose()[0],
                                self.loc_y.detach().numpy().transpose()[0]]).transpose()
        values = (self.evaluate_equation().detach().numpy().transpose()[0]) ** 2

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
    grid_c = griddata(coordinates, values, (grid_x, grid_y), method='cubic')

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


# note: works only inside the domain; in some cases on boundary, idx_offset can help
def material(loc, x_lims, y_lims, materials, rect_counts, idx_offset=0):
    rect_width = (x_lims[1] - x_lims[0]) / rect_counts[0]
    rect_height = (y_lims[1] - y_lims[0]) / rect_counts[1]
    rect_x_idx = ((loc[0] - x_lims[0]) / rect_width).astype(int)
    rect_y_idx = ((loc[1] - y_lims[0]) / rect_height).astype(int)
    return materials[0] + (materials[1] - materials[0]) * ((rect_x_idx + rect_y_idx + idx_offset) % 2)


if __name__ == '__main__':
    # problem information
    x_domain = [0, 4]
    y_domain = [0, 4]

    material_rects = [4, 4]
    material_coefs = [1, 10]

    dirichlet_const_A = 1
    dirichlet_const_B = 1
    neumann_const = 0
    robin_const = 3
    robin_alpha = 2

    # domain preparation
    boundary = Boundary([[x_domain[0], x_domain[1], x_domain[1], x_domain[0]],
                        [y_domain[0], y_domain[0], y_domain[1], y_domain[1]]])
    boundary_array = [boundary]
    ds = DomainSampler(boundary_array)
    ds.boundary_group_add('dirichlet_A', 0, [3, 0], connect_ends=False, color='blue')
    ds.boundary_group_add('dirichlet_B', 0, [2, 3], connect_ends=False, color='aqua')
    ds.boundary_group_add('neumann', 0, [0, 1], connect_ends=False, color='orange')
    ds.boundary_group_add('robin', 0, [1, 2], connect_ends=False, color='red')

    ds.distribute_line_probe(n_y=40)
    ds.boundary_groups_distribute()

    # generate interior samples
    i_count = 500

    i_loc = np.array(ds.sample_interior(i_count))
    i_material = material(i_loc, x_domain, y_domain, material_coefs, material_rects)

    # generate boundary samples
    b_count = 125

    d_A = ds.sample_boundary(b_count, label='dirichlet_A')
    d_B = ds.sample_boundary(b_count, label='dirichlet_B')
    dirichlet_loc = np.array([d_A[0] + d_B[0], d_A[1] + d_B[1]])
    dirichlet_samples = np.append(dirichlet_const_A * np.ones(b_count),
                                  dirichlet_const_B * np.ones(b_count))

    neumann_loc = np.array(ds.sample_boundary(b_count, label='neumann'))
    neumann_samples = neumann_const * np.ones(b_count)

    robin_loc = np.array(ds.sample_boundary(b_count, label='robin'))
    robin_samples = robin_const * np.ones(b_count)
    robin_material = material(robin_loc, x_domain, y_domain, material_coefs, material_rects, idx_offset=1)
    # (situational use of 'idx offset' at line above)

    # create PINN
    pinn = PINN(i_loc, i_material, dirichlet_loc, dirichlet_samples, neumann_loc, neumann_samples, robin_loc,
                robin_samples, robin_material, robin_alpha)

    # train PINN
    for r in range(1000):
        pinn.train()

    # visualize results
    pinn.plot_equation_loss('Equation loss', 'pde_loss.png')
    pinn.plot_trained_function('Trained function', 'func.png',
                               x_min=x_domain[0], x_max=x_domain[1], y_min=y_domain[0], y_max=y_domain[1])
