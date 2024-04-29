from DomainSampler import DomainSampler, Boundary
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import datetime


class NetworkModel(nn.Module):
    def __init__(self):
        super(NetworkModel, self).__init__()

        self.linear_start = nn.Linear(2, 30)
        self.linear_middle1 = nn.Linear(60, 30)
        self.linear_middle2 = nn.Linear(30, 30)
        self.linear_middle3 = nn.Linear(30, 30)
        self.linear_middle4 = nn.Linear(30, 30)
        self.linear_end = nn.Linear(30, 1)
        self.activation = nn.GELU()

    def forward(self, y):
        y = self.linear_start(y)

        # fourier features
        x1 = torch.sin(y)
        x2 = torch.cos(y)
        x = torch.cat((x1, x2), dim=1)

        x = self.linear_middle1(x)
        x = self.activation(x)
        x = self.linear_middle2(x)
        x = self.activation(x)
        x = self.linear_middle3(x)
        x = self.activation(x)
        x = self.linear_middle4(x)
        x = self.activation(x)
        x = self.linear_end(x)
        return x


class PINN:
    def __init__(self, i_loc, d_loc, d_val, n_loc, n_val, n_dir):
        self.i_loc_x = torch.tensor(i_loc[0].reshape(-1, 1),
                                    dtype=torch.float32,
                                    requires_grad=True)
        self.i_loc_y = torch.tensor(i_loc[1].reshape(-1, 1),
                                    dtype=torch.float32,
                                    requires_grad=True)

        self.d_loc_x = torch.tensor(d_loc[0].reshape(-1, 1),
                                    dtype=torch.float32,
                                    requires_grad=True)
        self.d_loc_y = torch.tensor(d_loc[1].reshape(-1, 1),
                                    dtype=torch.float32,
                                    requires_grad=True)
        self.d_val = torch.tensor(d_val.reshape(-1, 1),
                                  dtype=torch.float32)

        self.n_loc_x = torch.tensor(n_loc[0].reshape(-1, 1),
                                    dtype=torch.float32,
                                    requires_grad=True)
        self.n_loc_y = torch.tensor(n_loc[1].reshape(-1, 1),
                                    dtype=torch.float32,
                                    requires_grad=True)
        self.n_val = torch.tensor(n_val.reshape(-1, 1),
                                  dtype=torch.float32)

        self.n_dir_x = torch.tensor(n_dir[0].reshape(-1, 1),
                                    dtype=torch.float32)
        self.n_dir_y = torch.tensor(n_dir[1].reshape(-1, 1),
                                    dtype=torch.float32)

        self.null = torch.zeros((self.i_loc_x.shape[0], 1))

        self.model = NetworkModel()

        self.loss_function = nn.MSELoss()

        self.loss = 0

        self.loss_history = []
        self.loss_history_eq = []
        self.loss_history_d = []
        self.loss_history_n = []

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.0025,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=0,
                                          amsgrad=True)

        self.iterations = 0

        self.alpha_eq = 1
        self.alpha_d = 1
        self.alpha_n = 1

    def evaluate(self, *args):
        self.model.eval()
        output = self.model(torch.hstack(args))
        return output

    def evaluate_dirichlet(self):
        return self.evaluate(self.d_loc_x, self.d_loc_y)

    def evaluate_neumann(self):
        output = self.evaluate(self.n_loc_x, self.n_loc_y)

        dx = torch.autograd.grad(
            output, self.n_loc_x,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        dy = torch.autograd.grad(
            output, self.n_loc_y,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        return self.n_dir_x * dx + self.n_dir_y * dy

    def evaluate_equation(self):
        output = self.evaluate(self.i_loc_x, self.i_loc_y)

        dx = torch.autograd.grad(
            output, self.i_loc_x,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        dxx = torch.autograd.grad(
            dx, self.i_loc_x,
            grad_outputs=torch.ones_like(dx),
            retain_graph=True,
            create_graph=True)[0]

        dy = torch.autograd.grad(
            output, self.i_loc_y,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True)[0]

        dyy = torch.autograd.grad(
            dy, self.i_loc_y,
            grad_outputs=torch.ones_like(dy),
            retain_graph=True,
            create_graph=True)[0]

        return dxx + dyy

    def closure(self):
        self.optimizer.zero_grad()

        pred_eq = self.evaluate_equation()
        pred_d = self.evaluate_dirichlet()
        pred_n = self.evaluate_neumann()

        loss_eq = self.loss_function(pred_eq, self.null)
        loss_d = self.loss_function(pred_d, self.d_val)
        loss_n = self.loss_function(pred_n, self.n_val)

        self.loss_history_eq += [loss_eq.item()]
        self.loss_history_d += [loss_d.item()]
        self.loss_history_n += [loss_n.item()]

        self.loss = self.alpha_eq * loss_eq + self.alpha_d * loss_d + self.alpha_n * loss_n
        self.loss.backward()

        self.iterations += 1

        self.loss_history += [self.loss.item()]

        if not self.iterations % 100:
            print('Iterations: {0:}, Loss: {1:6.10f}'.format(self.iterations, self.loss))
            print('EQ: ', loss_eq.item())
            print('DI: ', loss_d.item())
            print('NE: ', loss_n.item())

        return self.loss

    def train(self):
        self.model.train()  # only sets a flag
        self.optimizer.step(self.closure)

    def plot_trained_function(self, title, filename, xx, yy):
        nn_input_x = torch.tensor(xx.reshape(-1, 1),
                                  dtype=torch.float32)
        nn_input_y = torch.tensor(yy.reshape(-1, 1),
                                  dtype=torch.float32)

        z_values = self.evaluate(nn_input_x, nn_input_y)
        z_values = z_values.detach().numpy().reshape(-1)

        coordinates = np.array([nn_input_x.numpy().reshape(-1),
                                nn_input_y.numpy().reshape(-1)]).transpose()

        display_heatmap(coordinates, z_values, title, filename)

    def plot_error(self, title, filename, xx, yy, comp_func):
        nn_input_x = torch.tensor(xx.reshape(-1, 1),
                                  dtype=torch.float32)
        nn_input_y = torch.tensor(yy.reshape(-1, 1),
                                  dtype=torch.float32)

        z_values = self.evaluate(nn_input_x, nn_input_y)
        z_values = np.abs(z_values.detach().numpy().reshape(-1) - comp_func) / (np.abs(comp_func) + 1)

        coordinates = np.array([nn_input_x.numpy().reshape(-1),
                                nn_input_y.numpy().reshape(-1)]).transpose()

        display_heatmap(coordinates, z_values, title, filename)

    def plot_equation_loss(self, title, filename):
        coordinates = np.array([self.i_loc_x.detach().numpy().reshape(-1),
                                self.i_loc_y.detach().numpy().reshape(-1)]).transpose()
        values = np.abs(self.evaluate_equation().detach().numpy().transpose()[0]) ** 2

        display_heatmap(coordinates, values, title, filename)

    def plot_error_hist(self, title, filename, xx, yy, comp_func):
        nn_input_x = torch.tensor(xx.reshape(-1, 1),
                                  dtype=torch.float32)
        nn_input_y = torch.tensor(yy.reshape(-1, 1),
                                  dtype=torch.float32)

        z_values = self.evaluate(nn_input_x, nn_input_y)
        z_values = np.abs(z_values.detach().numpy().reshape(-1) - comp_func) / (np.abs(comp_func) + 1)

        plt.figure(figsize=(9, 4.5))
        values = np.abs(z_values.reshape(-1))
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_equation_loss_hist(self, title, filename):
        plt.figure(figsize=(9, 4.5))
        values = np.abs(self.evaluate_equation().detach().numpy().reshape(-1)) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_dirichlet_loss_hist(self, title, filename):
        plt.figure(figsize=(9, 4.5))
        values = np.abs(self.evaluate_dirichlet().detach().numpy().reshape(-1) - self.d_val.detach().numpy().reshape(-1)) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_neumann_loss_hist(self, title, filename):
        plt.figure(figsize=(9, 4.5))
        values = np.abs(self.evaluate_neumann().detach().numpy().reshape(-1) - self.n_val.detach().numpy().reshape(-1)) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_loss_history(self, title, filename):
        plt.figure(figsize=(9, 4.5))
        plt.plot(list(range(len(self.loss_history))), np.log10(self.loss_history))
        plt.title(title)
        plt.savefig(filename)

    def plot_loss_history_all(self, title, filename):
        plt.figure(figsize=(9, 4.5))
        plt.plot(list(range(len(self.loss_history_eq))), np.log10(self.loss_history_eq), color='green')
        plt.plot(list(range(len(self.loss_history_d))), np.log10(self.loss_history_d), color='blue')
        plt.plot(list(range(len(self.loss_history_n))), np.log10(self.loss_history_n), color='orange')
        plt.legend(['equation loss', 'Dirichlet loss', 'Neumann loss'])
        plt.title(title)
        plt.savefig(filename)

    def plot_loss_history_eq(self, title, filename):
        plt.figure(figsize=(9, 4.5))
        plt.plot(list(range(len(self.loss_history_eq))), np.log10(self.loss_history_eq), color='green')
        plt.title(title)
        plt.savefig(filename)

    def plot_loss_history_d(self, title, filename):
        plt.figure(figsize=(9, 4.5))
        plt.plot(list(range(len(self.loss_history_d))), np.log10(self.loss_history_d), color='blue')
        plt.title(title)
        plt.savefig(filename)

    def plot_loss_history_n(self, title, filename):
        plt.figure(figsize=(9, 4.5))
        plt.plot(list(range(len(self.loss_history_n))), np.log10(self.loss_history_n), color='orange')
        plt.title(title)
        plt.savefig(filename)


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

    h = ax.imshow(grid_c, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='rainbow', aspect='auto')
    # h = ax.imshow(grid_c.T, origin='lower', cmap='rainbow',)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=10)

    # ax.legend(bbox_to_anchor=(1.1, 1), loc='center', borderaxespad=2)
    plt.savefig(filename)


def g_func(idx, xx, yy, a, b):
    if idx == 1:
        return a * xx + b
    elif idx == 2:
        return a * (xx ** 2 - yy ** 2)
    elif idx == 3:
        return a * np.exp(b * yy) * np.sin(b * xx)
    else:
        return 0 * xx


def dg_func(idx, xx, yy, a, b, dir_x, dir_y):
    if idx == 1:
        return a * dir_x
    elif idx == 2:
        return 2 * a * xx * dir_x - 2 * a * yy * dir_y
    elif idx == 3:
        return a * b * np.exp(b * yy) * np.cos(b * xx) * dir_x + a * b * np.exp(b * yy) * np.sin(b * xx) * dir_y
    else:
        return 0 * xx


if __name__ == '__main__':
    # problem information
    start = datetime.datetime.now()

    func = 3

    a = 0.5
    b = 1.5

    x_min = 0
    x_max = 2
    y_min = 0
    y_max = 1

    # domain preparation
    ds_left = DomainSampler([Boundary([[0, 1, 1, 0],
                                       [0, 0, 1, 1]])])
    ds_right = DomainSampler([Boundary([[1, 2, 2, 1],
                                       [0, 0, 1, 1]])])

    ds_left.boundary_group_add('dirichlet', 0, [3, 0], connect_ends=False, color='blue')
    ds_left.boundary_group_add('neumann1', 0, [0, 1], connect_ends=False, color='orange')
    ds_left.boundary_group_add('neumann2', 0, [2, 3], connect_ends=False, color='orange')
    # ds_left.boundary_group_add('transition', 0, [1, 2], connect_ends=False, color='green')
    ds_right.boundary_group_add('dirichlet', 0, [1, 2], connect_ends=False, color='blue')
    ds_right.boundary_group_add('neumann1', 0, [0, 1], connect_ends=False, color='orange')
    ds_right.boundary_group_add('neumann2', 0, [2, 3], connect_ends=False, color='orange')

    ds_left.distribute_line_probe(n_y=40)
    ds_left.boundary_groups_distribute()
    ds_right.distribute_line_probe(n_y=40)
    ds_right.boundary_groups_distribute()

    # generate interior samples
    i_count = 200  # summed: 2x

    i_loc_L = np.array(ds_left.sample_interior(i_count))
    i_loc_R = np.array(ds_right.sample_interior(i_count))

    i_loc = np.append(i_loc_L, i_loc_R, axis=1)

    # generate dirichlet samples
    d_count = 200  # summed: 2x

    d_loc_L = np.array(ds_left.sample_boundary(d_count, label='dirichlet'))
    d_loc_R = np.array(ds_right.sample_boundary(d_count, label='dirichlet'))

    d_loc = np.append(d_loc_L, d_loc_R, axis=1)
    d_samples = g_func(func, d_loc[0], d_loc[1], a, b)

    # generate neumann samples
    n_count = 100  # summed: 4x

    n_loc_L_1 = np.array(ds_left.sample_boundary(n_count, label='neumann1'))
    n_loc_R_1 = np.array(ds_right.sample_boundary(n_count, label='neumann1'))
    n_loc_L_2 = np.array(ds_left.sample_boundary(n_count, label='neumann2'))
    n_loc_R_2 = np.array(ds_right.sample_boundary(n_count, label='neumann2'))

    n_loc_1 = np.append(n_loc_L_1, n_loc_R_1, axis=1)
    n_loc_2 = np.append(n_loc_L_2, n_loc_R_2, axis=1)

    n_dir_1 = np.array([np.zeros_like(n_loc_1[0]), 1 * np.ones_like(n_loc_1[0])])
    n_dir_2 = np.array([np.zeros_like(n_loc_2[0]), -1 * np.ones_like(n_loc_2[0])])

    n_samples_1 = dg_func(func, n_loc_1[0], n_loc_1[1], a, b, n_dir_1[0], n_dir_1[1])
    n_samples_2 = dg_func(func, n_loc_2[0], n_loc_2[1], a, b, n_dir_2[0], n_dir_2[1])

    n_loc = np.append(n_loc_1, n_loc_2, axis=1)
    n_dir = np.append(n_dir_1, n_dir_2, axis=1)
    n_samples = np.append(n_samples_1, n_samples_2)

    # generate transition samples
    # t_count = 100
    #
    # t_loc = np.array(ds_left.sample_boundary(t_count, label='transition'))

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
    pinn = PINN(i_loc, d_loc, d_samples, n_loc, n_samples, n_dir)

    stop = datetime.datetime.now()
    timed = stop - start
    print("Domain preparation: " + str(timed))

    # train PINN
    start = datetime.datetime.now()
    timed = start - start
    inp = '100'
    while inp != 'q':
        if inp.lower() == 'lr':
            print('Enter [c] then [N] for learning rate ([c] * 10^[N]):')
            lr_const = input()
            lr_exp = input()
            try:
                lr_const = int(lr_const)
                lr_exp = int(lr_exp)
            except ValueError:
                print('Something went wrong, learning rate unchanged.')
            else:
                lr = lr_const * (10 ** lr_exp)
                for g in pinn.optimizer.param_groups:
                    g['lr'] = lr
                print('Learning rate changed to ' + str(lr) + '.')
        else:
            try:
                iters = int(inp)
            except ValueError:
                print('Something went wrong, no iterations made.')
            else:
                start = datetime.datetime.now()
                if 0 < iters <= 10000:
                    for r in range(iters):
                        pinn.train()
                stop = datetime.datetime.now()
                timed += stop - start
                print('Iterations complete.')
        inp = input()
    print("Training time: " + str(timed))
    print("Iterations: " + str(pinn.iterations))
    print("Avg time per iteration: " + str(timed / pinn.iterations))

    # generate original function values
    x_linspace = np.linspace(x_min, x_max, 50)
    y_linspace = np.linspace(y_min, y_max, 50)
    xx, yy = np.meshgrid(x_linspace, y_linspace)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    comp_func = g_func(func, xx, yy, a, b)

    # visualize results
    pinn.plot_trained_function('Trained function', 'task_1_' + str(func) + '_function.png', xx, yy)
    pinn.plot_error('Error', 'task_1_' + str(func) + '_error.png', xx, yy, comp_func)
    pinn.plot_error_hist('Error', 'task_1_' + str(func) + '_errorh.png', xx, yy, comp_func)

    pinn.plot_equation_loss('Equation loss', 'task_1_' + str(func) + '_loss_eq.png')
    pinn.plot_equation_loss_hist('Equation loss', 'task_1_' + str(func) + '_loss_eqh.png')
    pinn.plot_dirichlet_loss_hist('Dirichlet loss', 'task_1_' + str(func) + '_loss_d.png')
    pinn.plot_neumann_loss_hist('Neumann loss', 'task_1_' + str(func) + '_loss_n.png')

    pinn.plot_loss_history('Loss sum', 'task_1_' + str(func) + '_loss_history.png')
    pinn.plot_loss_history_all('Losses', 'task_1_' + str(func) + '_loss_history_all.png')
    pinn.plot_loss_history_eq('Equation loss', 'task_1_' + str(func) + '_loss_history_eq.png')
    pinn.plot_loss_history_d('Dirichlet loss', 'task_1_' + str(func) + '_loss_history_d.png')
    pinn.plot_loss_history_n('Neumann loss', 'task_1_' + str(func) + '_loss_history_n.png')


