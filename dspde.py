from DomainSampler import *
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


class PINN:
    # INIT args: equation - name of the equation ('laplace', 'darcyflow', 'burger')
    #            samples_loc - sampled locations: [[X],[Y]]
    #            samples - values of samples taken at samples_loc
    #            boundary_loc - sampled boundary locations: [[X],[Y]] (initial condition = [t=0] boundary condition)
    #            boundary_samples - values of samples taken at boundary_loc
    #            args - arguments specified below
    def __init__(self, equation, samples_loc, samples, boundary_loc, boundary_samples, *args):
        self.samples_x = torch.tensor(samples_loc[0].reshape(-1, 1),
                                      dtype=torch.float32,
                                      requires_grad=True)
        self.samples_y = torch.tensor(samples_loc[1].reshape(-1, 1),
                                      dtype=torch.float32,
                                      requires_grad=True)
        self.samples = torch.tensor(samples.reshape(-1, 1),
                                    dtype=torch.float32)

        self.boundary_x = torch.tensor(boundary_loc[0].reshape(-1, 1),
                                       dtype=torch.float32,
                                       requires_grad=True)
        self.boundary_y = torch.tensor(boundary_loc[1].reshape(-1, 1),
                                       dtype=torch.float32,
                                       requires_grad=True)
        self.boundary_samples = torch.tensor(boundary_samples.reshape(-1, 1),
                                             dtype=torch.float32)

        self.null = torch.zeros((self.samples.shape[0], 1))

        self.dimension = None

        # LAPLACE args: None
        if equation == 'laplace':
            self.dimension = 2
            self.evaluate_equation = self.evaluate_laplace
            self.evaluate_boundary = self.evaluate_boundary_2d

        # DARCY FLOW args: a - samples of function a(x,y) evaluated at samples_loc
        #                  dadx - differentiated a(x,y) with respect to x at samples_loc
        #                  dady - differentiated a(x,y) with respect to y at samples_loc
        if equation == 'darcy_flow':
            self.dimension = 2
            self.evaluate_equation = self.evaluate_darcyflow
            self.evaluate_boundary = self.evaluate_boundary_2d
            self.a = torch.tensor(args[0].reshape(-1, 1),
                                  dtype=torch.float32,
                                  requires_grad=True)
            self.dadx = torch.tensor(args[1].reshape(-1, 1),
                                     dtype=torch.float32,
                                     requires_grad=True)
            self.dady = torch.tensor(args[2].reshape(-1, 1),
                                     dtype=torch.float32,
                                     requires_grad=True)

        # BURGER args: samples_t - time locations for samples_loc
        #              boundary_t - time locations for boundary_loc
        #              nu - equation parameter of viscosity
        if equation == 'burger':
            self.dimension = 3
            self.evaluate_equation = self.evaluate_burger
            self.evaluate_boundary = self.evaluate_boundary_3d
            self.samples_t = torch.tensor(args[0].reshape(-1, 1),
                                          dtype=torch.float32,
                                          requires_grad=True)
            self.boundary_t = torch.tensor(args[1].reshape(-1, 1),
                                           dtype=torch.float32,
                                           requires_grad=True)
            self.nu = args[2]

        if not self.dimension:
            raise Exception("Unknown equation name")

        self.model = nn.Sequential(
            nn.Linear(self.dimension, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1))

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
        self.alpha_boundary = 100

    def evaluate(self, *args):
        self.model.eval()
        output = self.model(torch.hstack(args))
        return output

    def evaluate_boundary_2d(self):
        self.model.eval()
        output = self.model(torch.hstack((self.boundary_x, self.boundary_y)))
        return output

    def evaluate_boundary_3d(self):
        self.model.eval()
        output = self.model(torch.hstack((self.boundary_x, self.boundary_y, self.boundary_t)))
        return output

    def evaluate_laplace(self):
        f = self.samples
        u = self.evaluate(self.samples_x, self.samples_y)

        dx = torch.autograd.grad(
            u, self.samples_x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        dxx = torch.autograd.grad(
            dx, self.samples_x,
            grad_outputs=torch.ones_like(dx),
            retain_graph=True,
            create_graph=True)[0]

        dy = torch.autograd.grad(
            u, self.samples_y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        dyy = torch.autograd.grad(
            dy, self.samples_y,
            grad_outputs=torch.ones_like(dy),
            retain_graph=True,
            create_graph=True)[0]

        laplace = dxx + dyy - f
        return laplace

    def evaluate_darcyflow(self):
        x = self.samples_x
        y = self.samples_y
        f = self.samples
        a = self.a
        dadx = self.dadx
        dady = self.dady
        u = self.evaluate(x, y)

        dx = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        dxx = torch.autograd.grad(
            dx, x,
            grad_outputs=torch.ones_like(dx),
            retain_graph=True,
            create_graph=True)[0]

        dy = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        dyy = torch.autograd.grad(
            dy, y,
            grad_outputs=torch.ones_like(dy),
            retain_graph=True,
            create_graph=True)[0]

        darcy = dadx * dx + dady * dy + a * (dxx + dyy) + f
        return darcy

    def evaluate_burger(self):
        x = self.samples_x
        y = self.samples_y
        nu = self.nu
        u = self.evaluate(x, y, self.samples_t)

        dx = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        dxx = torch.autograd.grad(
            dx, x,
            grad_outputs=torch.ones_like(dx),
            retain_graph=True,
            create_graph=True)[0]

        dy = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        dyy = torch.autograd.grad(
            dy, y,
            grad_outputs=torch.ones_like(dy),
            retain_graph=True,
            create_graph=True)[0]

        dt = torch.autograd.grad(
            u, self.samples_t,
            grad_outputs=torch.ones_like(dy),
            retain_graph=True,
            create_graph=True)[0]

        burger = dt + u * dx + u * dy - nu * (dxx + dyy)
        return burger

    def closure(self):
        self.optimizer.zero_grad()

        u_prediction = self.evaluate_equation()
        boundary_prediction = self.evaluate_boundary()

        laplace_loss = self.alpha_equation * self.loss_function(u_prediction, self.null)
        boundary_loss = self.alpha_boundary * self.loss_function(boundary_prediction, self.boundary_samples)
        self.loss = laplace_loss + boundary_loss

        self.loss.backward()

        self.iterations += 1

        if not self.iterations % 100:
            print('Iterations: {0:}, Loss: {1:6.10f}'.format(self.iterations, self.loss))

        return self.loss

    def train(self):
        self.model.train()  # only sets a flag
        self.optimizer.step(self.closure)

    def plot(self, orig_form, title='', only_orig=False):
        # print('Total iterations: {0:}, Final loss: {1:6.10f}'.format(self.iterations, self.loss))
        steps = 50

        x_linspace = np.linspace(0, 1, steps)
        y_linspace = np.linspace(0, 1, steps)

        x_mesh, y_mesh = np.meshgrid(x_linspace, y_linspace)
        x_mesh = x_mesh.reshape(-1, 1)
        y_mesh = y_mesh.reshape(-1, 1)

        original_values = np.array([bilinear_form(orig_form, [x_mesh[n], y_mesh[n]]) for n in range(x_mesh.size)])
        original_values = original_values.reshape(steps, steps)

        if only_orig:
            if self.dimension == 2:
                plot_2d(x_mesh, y_mesh, original_values, title)
            else:
                print('This is not implemented yet')
            return

        nn_input_x = torch.tensor(x_mesh.reshape(-1, 1),
                                  dtype=torch.float32)
        nn_input_y = torch.tensor(y_mesh.reshape(-1, 1),
                                  dtype=torch.float32)
        if self.dimension == 2:
            z_values = self.evaluate(nn_input_x, nn_input_y)
        else:
            print('This is not implemented yet')
            return
        z_values = z_values.detach().numpy()
        z_values = z_values.reshape(steps, steps)

        error_values = z_values - original_values
        plot_2d(x_mesh, y_mesh, error_values, title)


def plot_2d(x_mesh, y_mesh, z_values, title=''):
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


def bilinear_form(form_matrix, input_values):
    rows = len(form_matrix)
    cols = len(form_matrix[0])
    output = 0
    for i in range(rows):
        for j in range(cols):
            output = output + form_matrix[i][j] * (input_values[0] ** i) * (input_values[1] ** j)
    return output


if __name__ == '__main__':
    boundary_1 = Boundary([[1, 1, 0, 0],
                           [0, 1, 1, 0]])
    ds = DomainSampler([boundary_1])

    ds.distribute_line_probe(n_y=50)
    ds.boundary_groups_distribute()

    #

    i_count = 200
    b_count = 100
    i_samples = np.array(ds.sample_interior(i_count))
    b_samples = np.array(ds.sample_boundary(b_count))

    #

    i_matrix = [[-4, 0, 6],  # [1  y   yy]
                [0, 0, 0],  # [x  xy  xyy]  <--- format
                [6, 0, 0]]  # [xx xxy xxyy]

    b_matrix = [[1, 2, -0.5],
                [-6, 0, 0],
                [-1.5, 0, 3]]

    a_matrix = [[1, 1, 0],
                [1, 0, 0],
                [0, 0, 1]]

    dadx_matrix = np.roll(np.array(a_matrix), -1, axis=1)
    dadx_matrix[-1][:] = 0
    dady_matrix = np.roll(np.array(a_matrix), -1, axis=0)
    dady_matrix[:][-1] = 0
    for k in range(len(dadx_matrix)):
        dadx_matrix[k][:] *= k
        dady_matrix[:][k] *= k

    #

    # LAPLACE variables
    i_values = np.array([bilinear_form(i_matrix, i_samples[:, k]) for k in range(i_count)])

    # DARCY FLOW variables
    # TODO get an example and try it on my code
    # i_values = ...
    # a_samples = np.array([bilinear_form(a_matrix, i_samples[:, k]) for k in range(i_count)])
    # dadx_samples = np.array([bilinear_form(dadx_matrix, i_samples[:, k]) for k in range(i_count)])
    # dady_samples = np.array([bilinear_form(dady_matrix, i_samples[:, k]) for k in range(i_count)])

    # BURGER variables
    # TODO get an example and try it on my code (edgaramo.py? but it is 1d -> set y to 0?)
    # t_samples = np.linspace(0, 1, i_count)
    # t_boundary = np.linspace(0, 1, b_count)
    # viscosity = 1e-3

    # BOUNDARY samples
    b_values = np.array([bilinear_form(b_matrix, b_samples[:, k]) for k in range(b_count)])

    #

    pinn = PINN('laplace', i_samples, i_values, b_samples, b_values)
    for r in range(2000):
        pinn.train()

    #

    # plot ABS. DIFF.
    pinn.plot(b_matrix, title='Error')

    # plot ORIG. VALS
    pinn.plot(b_matrix, only_orig=True, title='Original function')

    # plot PINN VALS
    zero_matrix = [[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]]
    pinn.plot(zero_matrix, title='PINN function')
    plt.show()
