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
    def __init__(self,
                 loc_L, dirichlet_loc_L, dirichlet_samples_L, neumann_loc_L, neumann_samples_L, neumann_dir_L,
                 loc_R, dirichlet_loc_R, dirichlet_samples_R, neumann_loc_R, neumann_samples_R, neumann_dir_R,
                 transition_loc, transition_dir):
        self.i_loc_x_L = torch.tensor(loc_L[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.i_loc_y_L = torch.tensor(loc_L[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.d_loc_x_L = torch.tensor(dirichlet_loc_L[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.d_loc_y_L = torch.tensor(dirichlet_loc_L[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.d_val_L = torch.tensor(dirichlet_samples_L.reshape(-1, 1), dtype=torch.float32)
        self.n_loc_x_L = torch.tensor(neumann_loc_L[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.n_loc_y_L = torch.tensor(neumann_loc_L[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.n_val_L = torch.tensor(neumann_samples_L.reshape(-1, 1), dtype=torch.float32)
        self.n_dir_x_L = torch.tensor(neumann_dir_L[0].reshape(-1, 1), dtype=torch.float32)
        self.n_dir_y_L = torch.tensor(neumann_dir_L[1].reshape(-1, 1), dtype=torch.float32)

        self.i_loc_x_R = torch.tensor(loc_R[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.i_loc_y_R = torch.tensor(loc_R[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.d_loc_x_R = torch.tensor(dirichlet_loc_R[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.d_loc_y_R = torch.tensor(dirichlet_loc_R[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.d_val_R = torch.tensor(dirichlet_samples_R.reshape(-1, 1), dtype=torch.float32)
        self.n_loc_x_R = torch.tensor(neumann_loc_R[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.n_loc_y_R = torch.tensor(neumann_loc_R[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.n_val_R = torch.tensor(neumann_samples_R.reshape(-1, 1), dtype=torch.float32)
        self.n_dir_x_R = torch.tensor(neumann_dir_R[0].reshape(-1, 1), dtype=torch.float32)
        self.n_dir_y_R = torch.tensor(neumann_dir_R[1].reshape(-1, 1), dtype=torch.float32)

        self.t_loc_x = torch.tensor(transition_loc[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.t_loc_y = torch.tensor(transition_loc[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.t_dir_x = torch.tensor(transition_dir[0].reshape(-1, 1), dtype=torch.float32)
        self.t_dir_y = torch.tensor(transition_dir[1].reshape(-1, 1), dtype=torch.float32)

        self.null_eq_L = torch.zeros((self.i_loc_x_L.shape[0], 1))
        self.null_eq_R = torch.zeros((self.i_loc_x_R.shape[0], 1))
        self.null_t = torch.zeros((self.t_loc_x.shape[0], 1))

        self.model_L = NetworkModel()
        self.model_R = NetworkModel()

        self.optimizer = torch.optim.Adam(list(self.model_L.parameters()) + list(self.model_R.parameters()),
                                          lr=0.001,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=0,
                                          amsgrad=True)

        self.loss_function = nn.MSELoss()
        self.loss = 0
        self.iterations = 0

        self.loss_history = []
        self.loss_history_L = []
        self.loss_history_eq_L = []
        self.loss_history_d_L = []
        self.loss_history_n_L = []
        self.loss_history_R = []
        self.loss_history_eq_R = []
        self.loss_history_d_R = []
        self.loss_history_n_R = []
        self.loss_history_t = []
        self.loss_history_t_dx = []

        self.alpha_equation = 1
        self.alpha_dirichlet = 1
        self.alpha_neumann = 1
        self.alpha_transition = 1

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

    def evaluate_transition(self, model, x, y):
        return self.evaluate(model, x, y)

    def evaluate_transition_dx(self, model, x, y, dir_x, dir_y):
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

        return dx * dir_x + dy * dir_y

    def evaluate_equation(self, model, x, y):
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

        return dxx + dyy

    def closure(self):
        self.optimizer.zero_grad()

        pred_eq_L = self.evaluate_equation(self.model_L, self.i_loc_x_L, self.i_loc_y_L)
        pred_d_L = self.evaluate_dirichlet(self.model_L, self.d_loc_x_L, self.d_loc_y_L)
        pred_n_L = self.evaluate_neumann(self.model_L, self.n_loc_x_L, self.n_loc_y_L, self.n_dir_x_L, self.n_dir_y_L)
        pred_t_L = self.evaluate_transition(self.model_L, self.t_loc_x, self.t_loc_y)
        pred_t_dx_L = self.evaluate_transition_dx(self.model_L, self.t_loc_x, self.t_loc_y, self.t_dir_x, self.t_dir_y)
        pred_eq_R = self.evaluate_equation(self.model_R, self.i_loc_x_R, self.i_loc_y_R)
        pred_d_R = self.evaluate_dirichlet(self.model_R, self.d_loc_x_R, self.d_loc_y_R)
        pred_n_R = self.evaluate_neumann(self.model_R, self.n_loc_x_R, self.n_loc_y_R, self.n_dir_x_R, self.n_dir_y_R)
        pred_t_R = self.evaluate_transition(self.model_R, self.t_loc_x, self.t_loc_y)
        pred_t_dx_R = self.evaluate_transition_dx(self.model_R, self.t_loc_x, self.t_loc_y, self.t_dir_x, self.t_dir_y)

        loss_eq_L = self.alpha_equation * self.loss_function(pred_eq_L, self.null_eq_L)
        loss_d_L = self.alpha_dirichlet * self.loss_function(pred_d_L, self.d_val_L)
        loss_n_L = self.alpha_neumann * self.loss_function(pred_n_L, self.n_val_L)
        loss_eq_R = self.alpha_equation * self.loss_function(pred_eq_R, self.null_eq_R)
        loss_d_R = self.alpha_dirichlet * self.loss_function(pred_d_R, self.d_val_R)
        loss_n_R = self.alpha_neumann * self.loss_function(pred_n_R, self.n_val_R)
        loss_t = self.alpha_transition * self.loss_function(pred_t_L - pred_t_R, self.null_t)
        loss_t_dx = self.alpha_transition * self.loss_function(pred_t_dx_L - pred_t_dx_R, self.null_t)

        self.loss_history_eq_L += [loss_eq_L.item()]
        self.loss_history_d_L += [loss_d_L.item()]
        self.loss_history_n_L += [loss_n_L.item()]
        self.loss_history_eq_R += [loss_eq_R.item()]
        self.loss_history_d_R += [loss_d_R.item()]
        self.loss_history_n_R += [loss_n_R.item()]
        self.loss_history_t += [loss_t.item()]
        self.loss_history_t_dx += [loss_t_dx.item()]

        loss_L = loss_eq_L + loss_d_L + loss_n_L
        loss_R = loss_eq_R + loss_d_R + loss_n_R

        self.loss_history_L += [loss_L.item()]
        self.loss_history_R += [loss_R.item()]

        self.loss = loss_L + loss_R + loss_t + loss_t_dx
        self.loss.backward()

        self.iterations += 1

        self.loss_history += [self.loss.item()]

        if not self.iterations % 100:
            print('Iterations: {0:}, Loss: {1:6.10f}'.format(self.iterations, self.loss))
            print('L-EQ: ', loss_eq_L.item() / self.alpha_equation)
            print('L-DI: ', loss_d_L.item() / self.alpha_dirichlet)
            print('L-NE: ', loss_n_L.item() / self.alpha_neumann)
            print('R-EQ: ', loss_eq_R.item() / self.alpha_equation)
            print('R-DI: ', loss_d_R.item() / self.alpha_dirichlet)
            print('R-NE: ', loss_n_R.item() / self.alpha_neumann)
            print('T-EQ_VAL: ', loss_t.item() / self.alpha_transition)
            print('T-EQ_DIF: ', loss_t_dx.item() / self.alpha_transition)

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

        coordinates = np.array([X, Y]).transpose()

        z_values = np.append(z_values_L, z_values_R)

        display_heatmap(coordinates, z_values, title, filename)

    def plot_error(self, title, filename, loc_L, loc_R, orig_L, orig_R):
        nn_input_x_L = torch.tensor(loc_L[0].reshape(-1, 1), dtype=torch.float32)
        nn_input_y_L = torch.tensor(loc_L[1].reshape(-1, 1), dtype=torch.float32)

        z_values_L = self.evaluate(self.model_L, nn_input_x_L, nn_input_y_L)
        z_values_L = np.abs(z_values_L.detach().numpy().reshape(-1) - orig_L) / (np.abs(orig_L) + 1)

        nn_input_x_R = torch.tensor(loc_R[0].reshape(-1, 1), dtype=torch.float32)
        nn_input_y_R = torch.tensor(loc_R[1].reshape(-1, 1), dtype=torch.float32)

        z_values_R = self.evaluate(self.model_R, nn_input_x_R, nn_input_y_R)
        z_values_R = np.abs(z_values_R.detach().numpy().reshape(-1) - orig_R) / (np.abs(orig_R) + 1)

        X = np.append(nn_input_x_L.numpy().reshape(-1), nn_input_x_R.numpy().reshape(-1))
        Y = np.append(nn_input_y_L.numpy().reshape(-1), nn_input_y_R.numpy().reshape(-1))

        coordinates = np.array([X, Y]).transpose()

        z_values = np.append(z_values_L, z_values_R)

        display_heatmap(coordinates, z_values, title, filename)

    def plot_error_hist(self, title, filename, loc_L, loc_R, orig_L, orig_R):
        nn_input_x_L = torch.tensor(loc_L[0].reshape(-1, 1), dtype=torch.float32)
        nn_input_y_L = torch.tensor(loc_L[1].reshape(-1, 1), dtype=torch.float32)

        z_values_L = self.evaluate(self.model_L, nn_input_x_L, nn_input_y_L)
        z_values_L = np.abs(z_values_L.detach().numpy().reshape(-1) - orig_L) / (np.abs(orig_L) + 1)

        nn_input_x_R = torch.tensor(loc_R[0].reshape(-1, 1), dtype=torch.float32)
        nn_input_y_R = torch.tensor(loc_R[1].reshape(-1, 1), dtype=torch.float32)

        z_values_R = self.evaluate(self.model_R, nn_input_x_R, nn_input_y_R)
        z_values_R = np.abs(z_values_R.detach().numpy().reshape(-1) - orig_R) / (np.abs(orig_R) + 1)

        plt.figure(figsize=(9, 4.5))
        z_values = np.append(z_values_L, z_values_R)
        plt.hist(z_values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_equation_loss(self, title, filename):
        X = np.append(self.i_loc_x_L.detach().numpy().reshape(-1), self.i_loc_x_R.detach().numpy().reshape(-1))
        Y = np.append(self.i_loc_y_L.detach().numpy().reshape(-1), self.i_loc_y_R.detach().numpy().reshape(-1))

        values_L = self.evaluate_equation(self.model_L, self.i_loc_x_L, self.i_loc_y_L).detach().numpy().reshape(-1)
        values_R = self.evaluate_equation(self.model_R, self.i_loc_x_R, self.i_loc_y_R).detach().numpy().reshape(-1)

        coordinates = np.array([X, Y]).transpose()
        values = np.abs(np.append(values_L, values_R)) ** 2

        display_heatmap(coordinates, values, title, filename)

    def plot_equation_loss_hist(self, title, filename):
        values_L = self.evaluate_equation(self.model_L, self.i_loc_x_L, self.i_loc_y_L).detach().numpy().reshape(-1)
        values_R = self.evaluate_equation(self.model_R, self.i_loc_x_R, self.i_loc_y_R).detach().numpy().reshape(-1)

        plt.figure(figsize=(9, 4.5))
        values = np.abs(np.append(values_L, values_R)) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_equation_loss_hist_L(self, title, filename):
        values_L = self.evaluate_equation(self.model_L, self.i_loc_x_L, self.i_loc_y_L).detach().numpy().reshape(-1)

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_L) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_equation_loss_hist_R(self, title, filename):
        values_R = self.evaluate_equation(self.model_R, self.i_loc_x_R, self.i_loc_y_R).detach().numpy().reshape(-1)

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_R) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_dirichlet_loss_hist(self, title, filename):
        values_L = (self.evaluate_dirichlet(self.model_L, self.d_loc_x_L, self.d_loc_y_L).detach().numpy().reshape(-1) - self.d_val_L.detach().numpy().reshape(-1)) ** 2
        values_R = (self.evaluate_dirichlet(self.model_R, self.d_loc_x_R, self.d_loc_y_R).detach().numpy().reshape(-1) - self.d_val_R.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(np.append(values_L, values_R)) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_dirichlet_loss_hist_L(self, title, filename):
        values_L = (self.evaluate_dirichlet(self.model_L, self.d_loc_x_L, self.d_loc_y_L).detach().numpy().reshape(-1) - self.d_val_L.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_L) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_dirichlet_loss_hist_R(self, title, filename):
        values_R = (self.evaluate_dirichlet(self.model_R, self.d_loc_x_R, self.d_loc_y_R).detach().numpy().reshape(-1) - self.d_val_R.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_R) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_neumann_loss_hist(self, title, filename):
        values_L = (self.evaluate_neumann(self.model_L, self.n_loc_x_L, self.n_loc_y_L, self.n_dir_x_L, self.n_dir_y_L).detach().numpy().reshape(-1) - self.n_val_L.detach().numpy().reshape(-1)) ** 2
        values_R = (self.evaluate_neumann(self.model_R, self.n_loc_x_R, self.n_loc_y_R, self.n_dir_x_R, self.n_dir_y_R).detach().numpy().reshape(-1) - self.n_val_R.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(np.append(values_L, values_R)) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_neumann_loss_hist_L(self, title, filename):
        values_L = (self.evaluate_neumann(self.model_L, self.n_loc_x_L, self.n_loc_y_L, self.n_dir_x_L, self.n_dir_y_L).detach().numpy().reshape(-1) - self.n_val_L.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_L) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_neumann_loss_hist_R(self, title, filename):
        values_R = (self.evaluate_neumann(self.model_R, self.n_loc_x_R, self.n_loc_y_R, self.n_dir_x_R, self.n_dir_y_R).detach().numpy().reshape(-1) - self.n_val_R.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_R) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_transition_loss_hist(self, title, filename):
        values_L = self.evaluate_transition(self.model_L, self.t_loc_x, self.t_loc_y).detach().numpy().reshape(-1)
        values_R = self.evaluate_transition(self.model_R, self.t_loc_x, self.t_loc_y).detach().numpy().reshape(-1)

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_L - values_R) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_transition_dx_loss_hist(self, title, filename):
        values_L = self.evaluate_transition_dx(self.model_L, self.t_loc_x, self.t_loc_y, self.t_dir_x, self.t_dir_y).detach().numpy().reshape(-1)
        values_R = self.evaluate_transition_dx(self.model_R, self.t_loc_x, self.t_loc_y, self.t_dir_x, self.t_dir_y).detach().numpy().reshape(-1)

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_L - values_R) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_loss_history(self, array, filename):
        fig = plt.figure(figsize=(9, 4.5))
        plt.plot(list(range(len(array))), np.log10(array))
        plt.savefig(filename)

    def plot_loss_history_all_L(self, title, filename):
        plt.figure(figsize=(9, 4.5))
        plt.plot(list(range(len(self.loss_history_eq_L))), np.log10(self.loss_history_eq_L), color='green')
        plt.plot(list(range(len(self.loss_history_d_L))), np.log10(self.loss_history_d_L), color='blue')
        plt.plot(list(range(len(self.loss_history_n_L))), np.log10(self.loss_history_n_L), color='orange')
        plt.legend(['equation loss', 'Dirichlet loss', 'Neumann loss'])
        plt.title(title)
        plt.savefig(filename)

    def plot_loss_history_all_R(self, title, filename):
        plt.figure(figsize=(9, 4.5))
        plt.plot(list(range(len(self.loss_history_eq_R))), np.log10(self.loss_history_eq_R), color='green')
        plt.plot(list(range(len(self.loss_history_d_R))), np.log10(self.loss_history_d_R), color='blue')
        plt.plot(list(range(len(self.loss_history_n_R))), np.log10(self.loss_history_n_R), color='orange')
        plt.legend(['equation loss', 'Dirichlet loss', 'Neumann loss'])
        plt.title(title)
        plt.savefig(filename)

    def plot_loss_history_all_t(self, title, filename):
        plt.figure(figsize=(9, 4.5))
        plt.plot(list(range(len(self.loss_history_t))), np.log10(self.loss_history_t), color='aqua')
        plt.plot(list(range(len(self.loss_history_t_dx))), np.log10(self.loss_history_t_dx), color='blue')
        plt.legend(['transition loss', 'transition differential loss'])
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
    func = 3

    a = 0.5
    b = 1.5

    x_min = 0
    x_max = 2
    y_min = 0
    y_max = 1

    # domain preparation
    t_loc_count = 100
    t_boundary_x_L = np.append(np.array([0]), np.linspace(x_min, x_max, t_loc_count))
    t_boundary_x_R = np.append(np.linspace(x_min, x_max, t_loc_count), np.array([2]))
    t_boundary_y_L = (t_boundary_x_L - 2) ** 2 / 4
    t_boundary_y_R = (t_boundary_x_R - 2) ** 2 / 4
    t_boundary_y_L[0] = 0
    t_boundary_y_R[-1] = 1

    ds_left = DomainSampler([Boundary([np.asarray(t_boundary_x_L),
                                       np.asarray(t_boundary_y_L)])])
    ds_right = DomainSampler([Boundary([np.asarray(t_boundary_x_R),
                                        np.asarray(t_boundary_y_R)])])

    ds_left.boundary_group_add('dirichlet', 0, [0, 1], connect_ends=False, color='blue')
    ds_left.boundary_group_add('neumann', 0, [t_loc_count, 0], connect_ends=False, color='orange')
    ds_left.boundary_group_add('transition', 0, list(range(1, t_loc_count + 1)), connect_ends=False, color='green')
    ds_right.boundary_group_add('dirichlet', 0, [t_loc_count - 1, t_loc_count], connect_ends=False, color='blue')
    ds_right.boundary_group_add('neumann', 0, [t_loc_count, 0], connect_ends=False, color='orange')

    ds_left.distribute_line_probe(n_y=40)
    ds_left.boundary_groups_distribute()
    ds_right.distribute_line_probe(n_y=40)
    ds_right.boundary_groups_distribute()

    # generate interior samples
    i_count = 200

    i_loc_L = np.array(ds_left.sample_interior(i_count))
    i_loc_R = np.array(ds_right.sample_interior(i_count))

    # generate dirichlet samples
    d_count = 200

    d_loc_L = np.array(ds_left.sample_boundary(d_count, label='dirichlet'))
    d_loc_R = np.array(ds_right.sample_boundary(d_count, label='dirichlet'))
    d_samples_L = g_func(func, d_loc_L[0], d_loc_L[1], a, b)
    d_samples_R = g_func(func, d_loc_R[0], d_loc_R[1], a, b)

    # generate neumann samples
    n_count = 100

    n_loc_L = np.array(ds_left.sample_boundary(n_count, label='neumann'))
    n_dir_L = np.array([np.zeros_like(n_loc_L[0]), -1 * np.ones_like(n_loc_L[0])])
    n_samples_L = dg_func(func, n_loc_L[0], n_loc_L[1], a, b, n_dir_L[0], n_dir_L[1])

    n_loc_R = np.array(ds_right.sample_boundary(n_count, label='neumann'))
    n_dir_R = np.array([np.zeros_like(n_loc_R[0]), 1 * np.ones_like(n_loc_R[0])])
    n_samples_R = dg_func(func, n_loc_R[0], n_loc_R[1], a, b, n_dir_R[0], n_dir_R[1])

    # generate transition samples
    t_count = 400

    t_loc = np.array(ds_left.sample_boundary(t_count, label='transition'))
    t_dir_x = (t_loc[0] - 2) / 2  # [1, (x-2)/2] -> [(x-2)/2, -1]
    t_dir_y = - np.ones_like(t_dir_x)
    norms = np.array([(t_dir_x[i] ** 2 + t_dir_y[i] ** 2) ** 0.5 for i in range(t_count)])
    t_dir = np.array([t_dir_x / norms, t_dir_y / norms])

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
    pinn = PINN(i_loc_L, d_loc_L, d_samples_L, n_loc_L, n_samples_L, n_dir_L,
                i_loc_R, d_loc_R, d_samples_R, n_loc_R, n_samples_R, n_dir_R,
                t_loc, t_dir)

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
    coords = np.array([xx, yy]).transpose()
    loc_L = np.array([coord for coord in coords if coord[1] <= ((coord[0] - 2) ** 2) / 4])
    loc_R = np.array([coord for coord in coords if coord[1] > ((coord[0] - 2) ** 2) / 4])
    loc_L = loc_L.transpose()
    loc_R = loc_R.transpose()
    orig_L = g_func(func, loc_L[0], loc_L[1], a, b)
    orig_R = g_func(func, loc_R[0], loc_R[1], a, b)

    # visualize results
    pinn.plot_trained_function('Trained function', 'task_2B_' + str(func) + '_function.png', loc_L, loc_R)
    pinn.plot_error('Error', 'task_2B_' + str(func) + '_error.png', loc_L, loc_R, orig_L, orig_R)
    pinn.plot_error_hist('Error', 'task_2B_' + str(func) + '_errorh.png', loc_L, loc_R, orig_L, orig_R)

    pinn.plot_equation_loss('Equation loss', 'task_2B_' + str(func) + '_pde_loss.png')
    pinn.plot_equation_loss_hist('Equation loss', 'task_2B_' + str(func) + '_loss_eqh.png')
    pinn.plot_equation_loss_hist_L('Equation loss (L)', 'task_2B_' + str(func) + '_loss_eqh_L.png')
    pinn.plot_equation_loss_hist_R('Equation loss (R)', 'task_2B_' + str(func) + '_loss_eqh_R.png')
    pinn.plot_dirichlet_loss_hist('Dirichlet loss', 'task_2B_' + str(func) + '_loss_dh.png')
    pinn.plot_dirichlet_loss_hist_L('Dirichlet loss (L)', 'task_2B_' + str(func) + '_loss_dh_L.png')
    pinn.plot_dirichlet_loss_hist_R('Dirichlet loss (R)', 'task_2B_' + str(func) + '_loss_dh_R.png')
    pinn.plot_neumann_loss_hist('Neumann loss', 'task_2B_' + str(func) + '_loss_nh.png')
    pinn.plot_neumann_loss_hist_L('Neumann loss (L)', 'task_2B_' + str(func) + '_loss_nh_L.png')
    pinn.plot_neumann_loss_hist_R('Neumann loss (R)', 'task_2B_' + str(func) + '_loss_nh_R.png')
    pinn.plot_transition_loss_hist('Transition loss', 'task_2B_' + str(func) + '_loss_th.png')
    pinn.plot_transition_dx_loss_hist('Transition differential loss', 'task_2B_' + str(func) + '_loss_tdh.png')

    pinn.plot_loss_history_all_L('Loss history (L)', 'task_2B_' + str(func) + '_loss_history_L_all.png')
    pinn.plot_loss_history_all_R('Loss history (R)', 'task_2B_' + str(func) + '_loss_history_R_all.png')
    pinn.plot_loss_history_all_t('Loss history (t)', 'task_2B_' + str(func) + '_loss_history_t_all.png')
