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
        self.linear_middle5 = nn.Linear(30, 30)
        self.linear_middle6 = nn.Linear(30, 30)
        self.linear_middle7 = nn.Linear(30, 30)
        self.linear_middle8 = nn.Linear(30, 30)
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
        x = self.linear_middle5(x)
        x = self.activation(x)
        x = self.linear_middle6(x)
        x = self.activation(x)
        x = self.linear_middle7(x)
        x = self.activation(x)
        x = self.linear_middle8(x)
        x = self.activation(x)
        x = self.linear_end(x)
        return x


class PINN:
    def __init__(self, i_loc, i_mat, i_mdf, d_loc, d_val, n_loc, n_val, n_mat, n_dir, t_loc, t_dir, t_mat):
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
        self.n_mat_L = torch.tensor(n_mat[0].reshape(-1, 1), dtype=torch.float32)
        self.n_dir_x_L = torch.tensor(n_dir[0][0].reshape(-1, 1), dtype=torch.float32)
        self.n_dir_y_L = torch.tensor(n_dir[0][1].reshape(-1, 1), dtype=torch.float32)

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
        self.n_mat_R = torch.tensor(n_mat[1].reshape(-1, 1), dtype=torch.float32)
        self.n_dir_x_R = torch.tensor(n_dir[1][0].reshape(-1, 1), dtype=torch.float32)
        self.n_dir_y_R = torch.tensor(n_dir[1][1].reshape(-1, 1), dtype=torch.float32)

        self.t_x = torch.tensor(t_loc[0].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.t_y = torch.tensor(t_loc[1].reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        self.t_dir_x = torch.tensor(t_dir[0].reshape(-1, 1), dtype=torch.float32)
        self.t_dir_y = torch.tensor(t_dir[1].reshape(-1, 1), dtype=torch.float32)
        self.t_mat_L = torch.tensor(t_mat[0].reshape(-1, 1), dtype=torch.float32)
        self.t_mat_R = torch.tensor(t_mat[1].reshape(-1, 1), dtype=torch.float32)

        self.null_eq_L = torch.zeros((self.i_x_L.shape[0], 1))
        self.null_eq_R = torch.zeros((self.i_x_R.shape[0], 1))
        self.null_t = torch.zeros((self.t_x.shape[0], 1))

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

        self.loss_history_eq_L = []
        self.loss_history_d_L = []
        self.loss_history_n_L = []
        self.loss_history_eq_R = []
        self.loss_history_d_R = []
        self.loss_history_n_R = []

        self.loss_history_L = []
        self.loss_history_R = []

        self.loss_history = []
        self.loss_history_t = []
        self.loss_history_t_dx = []

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

    def evaluate_neumann(self, model, x, y, mat, dir_x, dir_y):
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

        pred_eq_L = self.evaluate_equation(self.model_L, self.i_x_L, self.i_y_L, self.i_mat_L, self.i_mdf_x_L,
                                           self.i_mdf_y_L)
        pred_d_L = self.evaluate_dirichlet(self.model_L, self.d_x_L, self.d_y_L)
        pred_n_L = self.evaluate_neumann(self.model_L, self.n_x_L, self.n_y_L, self.n_mat_L, self.n_dir_x_L,
                                         self.n_dir_y_L)
        pred_t_L = self.evaluate_transition(self.model_L, self.t_x, self.t_y)
        pred_t_dx_L = self.evaluate_transition_dx(self.model_L, self.t_x, self.t_y, self.t_dir_x, self.t_dir_y,
                                                  self.t_mat_L)

        pred_eq_R = self.evaluate_equation(self.model_R, self.i_x_R, self.i_y_R, self.i_mat_R, self.i_mdf_x_R,
                                           self.i_mdf_y_R)
        pred_d_R = self.evaluate_dirichlet(self.model_R, self.d_x_R, self.d_y_R)
        pred_n_R = self.evaluate_neumann(self.model_R, self.n_x_R, self.n_y_R, self.n_mat_R, self.n_dir_x_R,
                                         self.n_dir_y_R)
        pred_t_R = self.evaluate_transition(self.model_R, self.t_x, self.t_y)
        pred_t_dx_R = self.evaluate_transition_dx(self.model_R, self.t_x, self.t_y, self.t_dir_x, self.t_dir_y,
                                                  self.t_mat_R)

        loss_eq_L = self.eq_loss_alf * self.loss_function(pred_eq_L, self.null_eq_L)
        loss_d_L = self.d_loss_alf * self.loss_function(pred_d_L, self.d_val_L)
        loss_n_L = self.n_loss_alf * self.loss_function(pred_n_L, self.n_val_L)
        loss_eq_R = self.eq_loss_alf * self.loss_function(pred_eq_R, self.null_eq_R)
        loss_d_R = self.d_loss_alf * self.loss_function(pred_d_R, self.d_val_R)
        loss_n_R = self.n_loss_alf * self.loss_function(pred_n_R, self.n_val_R)
        loss_t = self.t_loss_alf * self.loss_function(pred_t_L - pred_t_R, self.null_t)
        loss_t_dx = self.t_loss_alf * self.loss_function(pred_t_dx_L - pred_t_dx_R, self.null_t)

        self.loss_history_eq_L.append(loss_eq_L.item())
        self.loss_history_d_L.append(loss_d_L.item())
        self.loss_history_n_L.append(loss_n_L.item())
        self.loss_history_eq_R.append(loss_eq_R.item())
        self.loss_history_d_R.append(loss_d_R.item())
        self.loss_history_n_R.append(loss_n_R.item())
        self.loss_history_t.append(loss_t.item())
        self.loss_history_t_dx.append(loss_t_dx.item())

        loss_L = loss_eq_L + loss_d_L + loss_n_L
        loss_R = loss_eq_R + loss_d_R + loss_n_R

        self.loss_history_L.append(loss_L.item())
        self.loss_history_R.append(loss_R.item())

        self.loss = loss_L + loss_R + loss_t + loss_t_dx
        self.loss.backward()

        self.iterations += 1

        self.loss_history.append(self.loss.item())

        if not self.iterations % 100:
            print('Iterations: {0:}, Loss: {1:6.10f}'.format(self.iterations, self.loss))
            print('L-EQ: ', loss_eq_L.item())
            print('L-DI: ', loss_d_L.item())
            print('L-NE: ', loss_n_L.item())
            print('R-EQ: ', loss_eq_R.item())
            print('R-DI: ', loss_d_R.item())
            print('R-NE: ', loss_n_R.item())
            print('T-EQ_VAL: ', loss_t.item())
            print('T-EQ_DIF: ', loss_t_dx.item())

        return self.loss

    def closure_d(self):
        self.optimizer.zero_grad()

        pred_eq_L = self.evaluate_equation(self.model_L, self.i_x_L, self.i_y_L, self.i_mat_L, self.i_mdf_x_L,
                                           self.i_mdf_y_L)
        pred_d_L = self.evaluate_dirichlet(self.model_L, self.d_x_L, self.d_y_L)

        pred_eq_R = self.evaluate_equation(self.model_R, self.i_x_R, self.i_y_R, self.i_mat_R, self.i_mdf_x_R,
                                           self.i_mdf_y_R)
        pred_d_R = self.evaluate_dirichlet(self.model_R, self.d_x_R, self.d_y_R)

        loss_eq_L = self.eq_loss_alf * self.loss_function(pred_eq_L, self.null_eq_L)
        loss_d_L = self.d_loss_alf * self.loss_function(pred_d_L, self.d_val_L)
        loss_eq_R = self.eq_loss_alf * self.loss_function(pred_eq_R, self.null_eq_R)
        loss_d_R = self.d_loss_alf * self.loss_function(pred_d_R, self.d_val_R)

        self.loss_history_eq_L.append(loss_eq_L.item())
        self.loss_history_d_L.append(loss_d_L.item())
        self.loss_history_n_L.append(0)
        self.loss_history_eq_R.append(loss_eq_R.item())
        self.loss_history_d_R.append(loss_d_R.item())
        self.loss_history_n_R.append(0)
        self.loss_history_t.append(0)
        self.loss_history_t_dx.append(0)

        loss_L = loss_eq_L + loss_d_L
        loss_R = loss_eq_R + loss_d_R

        self.loss_history_L.append(loss_L.item())
        self.loss_history_R.append(loss_R.item())

        self.loss = loss_L + loss_R
        self.loss.backward()

        self.iterations += 1

        self.loss_history.append(self.loss.item())

        if not self.iterations % 100:
            print('Iterations: {0:}, Loss: {1:6.10f}'.format(self.iterations, self.loss))
            print('L-EQ: ', loss_eq_L.item())
            print('L-DI: ', loss_d_L.item())
            print('L-NE: ', 0.0)
            print('R-EQ: ', loss_eq_R.item())
            print('R-DI: ', loss_d_R.item())
            print('R-NE: ', 0.0)
            print('T-EQ_VAL: ', 0.0)
            print('T-EQ_DIF: ', 0.0)

        return self.loss

    def train(self):
        self.model_L.train()  # only sets a flag
        self.model_R.train()  # only sets a flag
        self.optimizer.step(self.closure)

    def train_d(self):
        self.model_L.train()  # only sets a flag
        self.model_R.train()  # only sets a flag
        self.optimizer.step(self.closure_d)

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
        z_values_L = np.abs(z_values_L.detach().numpy().reshape(-1) - orig_L)

        nn_input_x_R = torch.tensor(loc_R[0].reshape(-1, 1), dtype=torch.float32)
        nn_input_y_R = torch.tensor(loc_R[1].reshape(-1, 1), dtype=torch.float32)

        z_values_R = self.evaluate(self.model_R, nn_input_x_R, nn_input_y_R)
        z_values_R = np.abs(z_values_R.detach().numpy().reshape(-1) - orig_R)

        X = np.append(nn_input_x_L.numpy().reshape(-1), nn_input_x_R.numpy().reshape(-1))
        Y = np.append(nn_input_y_L.numpy().reshape(-1), nn_input_y_R.numpy().reshape(-1))

        coordinates = np.array([X, Y]).transpose()

        z_values = np.append(z_values_L, z_values_R)

        display_heatmap(coordinates, z_values, title, filename)

    def plot_error_hist(self, title, filename, loc_L, loc_R, orig_L, orig_R):
        nn_input_x_L = torch.tensor(loc_L[0].reshape(-1, 1), dtype=torch.float32)
        nn_input_y_L = torch.tensor(loc_L[1].reshape(-1, 1), dtype=torch.float32)

        z_values_L = self.evaluate(self.model_L, nn_input_x_L, nn_input_y_L)
        z_values_L = np.abs(z_values_L.detach().numpy().reshape(-1) - orig_L)

        nn_input_x_R = torch.tensor(loc_R[0].reshape(-1, 1), dtype=torch.float32)
        nn_input_y_R = torch.tensor(loc_R[1].reshape(-1, 1), dtype=torch.float32)

        z_values_R = self.evaluate(self.model_R, nn_input_x_R, nn_input_y_R)
        z_values_R = np.abs(z_values_R.detach().numpy().reshape(-1) - orig_R)

        plt.figure(figsize=(9, 4.5))
        z_values = np.append(z_values_L, z_values_R)
        plt.hist(z_values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_equation_loss(self, title, filename):
        X = np.append(self.i_x_L.detach().numpy().reshape(-1), self.i_x_R.detach().numpy().reshape(-1))
        Y = np.append(self.i_y_L.detach().numpy().reshape(-1), self.i_y_R.detach().numpy().reshape(-1))

        values_L = self.evaluate_equation(self.model_L, self.i_x_L, self.i_y_L, self.i_mat_L, self.i_mdf_x_L,
                                           self.i_mdf_y_L).detach().numpy().reshape(-1)
        values_R = self.evaluate_equation(self.model_R, self.i_x_R, self.i_y_R, self.i_mat_R, self.i_mdf_x_R,
                                           self.i_mdf_y_R).detach().numpy().reshape(-1)

        coordinates = np.array([X, Y]).transpose()
        values = np.abs(np.append(values_L, values_R)) ** 2

        display_heatmap(coordinates, values, title, filename)

    def plot_equation_loss_hist(self, title, filename):
        values_L = self.evaluate_equation(self.model_L, self.i_x_L, self.i_y_L, self.i_mat_L, self.i_mdf_x_L,
                                           self.i_mdf_y_L).detach().numpy().reshape(-1)
        values_R = self.evaluate_equation(self.model_R, self.i_x_R, self.i_y_R, self.i_mat_R, self.i_mdf_x_R,
                                           self.i_mdf_y_R).detach().numpy().reshape(-1)

        plt.figure(figsize=(9, 4.5))
        values = np.abs(np.append(values_L, values_R)) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_equation_loss_hist_L(self, title, filename):
        values_L = self.evaluate_equation(self.model_L, self.i_x_L, self.i_y_L, self.i_mat_L, self.i_mdf_x_L,
                                           self.i_mdf_y_L).detach().numpy().reshape(-1)

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_L) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_equation_loss_hist_R(self, title, filename):
        values_R = self.evaluate_equation(self.model_R, self.i_x_R, self.i_y_R, self.i_mat_R, self.i_mdf_x_R,
                                           self.i_mdf_y_R).detach().numpy().reshape(-1)

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_R) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_dirichlet_loss_hist(self, title, filename):
        values_L = (self.evaluate_dirichlet(self.model_L, self.d_x_L, self.d_y_L).detach().numpy().reshape(
            -1) - self.d_val_L.detach().numpy().reshape(-1)) ** 2
        values_R = (self.evaluate_dirichlet(self.model_R, self.d_x_R, self.d_y_R).detach().numpy().reshape(
            -1) - self.d_val_R.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(np.append(values_L, values_R)) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_dirichlet_loss_hist_L(self, title, filename):
        values_L = (self.evaluate_dirichlet(self.model_L, self.d_x_L, self.d_y_L).detach().numpy().reshape(
            -1) - self.d_val_L.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_L) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_dirichlet_loss_hist_R(self, title, filename):
        values_R = (self.evaluate_dirichlet(self.model_R, self.d_x_R, self.d_y_R).detach().numpy().reshape(
            -1) - self.d_val_R.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_R) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_neumann_loss_hist(self, title, filename):
        values_L = (self.evaluate_neumann(self.model_L, self.n_x_L, self.n_y_L, self.n_mat_L, self.n_dir_x_L,
                                          self.n_dir_y_L).detach().numpy().reshape(
            -1) - self.n_val_L.detach().numpy().reshape(-1)) ** 2
        values_R = (self.evaluate_neumann(self.model_R, self.n_x_R, self.n_y_R, self.n_mat_R, self.n_dir_x_R,
                                          self.n_dir_y_R).detach().numpy().reshape(
            -1) - self.n_val_R.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(np.append(values_L, values_R)) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_neumann_loss_hist_L(self, title, filename):
        values_L = (self.evaluate_neumann(self.model_L, self.n_x_L, self.n_y_L, self.n_mat_L, self.n_dir_x_L,
                                          self.n_dir_y_L).detach().numpy().reshape(
            -1) - self.n_val_L.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_L) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_neumann_loss_hist_R(self, title, filename):
        values_R = (self.evaluate_neumann(self.model_R, self.n_x_R, self.n_y_R, self.n_mat_R, self.n_dir_x_R,
                                          self.n_dir_y_R).detach().numpy().reshape(
            -1) - self.n_val_R.detach().numpy().reshape(-1)) ** 2

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_R) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_transition_loss_hist(self, title, filename):
        values_L = self.evaluate_transition(self.model_L, self.t_x, self.t_y).detach().numpy().reshape(-1)
        values_R = self.evaluate_transition(self.model_R, self.t_x, self.t_y).detach().numpy().reshape(-1)

        plt.figure(figsize=(9, 4.5))
        values = np.abs(values_L - values_R) ** 2
        plt.hist(values, bins='auto')
        plt.title(title)
        plt.savefig(filename)

    def plot_transition_dx_loss_hist(self, title, filename):
        values_L = self.evaluate_transition_dx(self.model_L, self.t_x, self.t_y, self.t_dir_x,
                                               self.t_dir_y, self.t_mat_L).detach().numpy().reshape(-1)
        values_R = self.evaluate_transition_dx(self.model_R, self.t_x, self.t_y, self.t_dir_x,
                                               self.t_dir_y, self.t_mat_R).detach().numpy().reshape(-1)

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


if __name__ == '__main__':
    print('Program started.')

    # problem information
    material_rects = [4, 4]  # don't change this, the boundaries should be changes as well ('material_rects')
    material_coefs = [1, 0.1]  # [alpha_2, alpha_1]

    dirichlet_const_A = 0
    dirichlet_const_B = 1
    neumann_const = 0

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

    ds_omega[0].boundary_group_add('dirichlet_B', 7, [1, 2], connect_ends=False, color='aqua')
    ds_omega[1].boundary_group_add('dirichlet_B', 7, [1, 2], connect_ends=False, color='aqua')
    ds_omega[0].boundary_group_add('dirichlet_B', 6, [1, 2], connect_ends=False, color='aqua')
    ds_omega[1].boundary_group_add('dirichlet_B', 6, [1, 2], connect_ends=False, color='aqua')

    ds_omega[1].boundary_group_add('neumann', 1, [2, 3], connect_ends=False, color='orange')
    ds_omega[0].boundary_group_add('neumann', 3, [2, 3], connect_ends=False, color='orange')
    ds_omega[1].boundary_group_add('neumann', 5, [2, 3], connect_ends=False, color='orange')
    ds_omega[0].boundary_group_add('neumann', 7, [2, 3], connect_ends=False, color='orange')

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

    i_count = 500
    d_count = 125
    n_count = 250
    t_count = 200

    i_loc = [[], []]
    i_mat = [[], []]
    i_mdf = [[], []]
    d_loc = [[], []]
    d_val = [[], []]
    n_loc = [[], []]
    n_val = [[], []]
    n_mat = [[], []]
    n_dir = [[], []]
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
        n_mat[k] = material_coefs[k] * np.ones_like(n_loc[k][0])
        n_dir[k] = np.array([np.zeros_like(n_val[k]), np.ones_like(n_val[k])])

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
        plt.show()

    ds_omega[0].plot_domain()
    ds_omega[1].plot_domain()
    plt.plot(t_loc[0], t_loc[1], marker='+', color='black', linestyle='None')
    plt.show()

    # create PINN
    pinn = PINN(i_loc, i_mat, i_mdf, d_loc, d_val, n_loc, n_val, n_mat, n_dir, t_loc, t_dir, t_mat)

    print('PINN created.')

    # train PINN
    print('Mode:')
    mode = input()
    print('Iterations:')
    inp = input()

    start = datetime.datetime.now()
    timed = start - start
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
        elif inp.lower() == 'mode':
            print('Specify mode:')
            mode = input()
            print('Mode changed to ' + str(mode) + '.')
        else:
            try:
                iters = int(inp)
            except ValueError:
                print('Something went wrong, no iterations made.')
            else:
                if mode == 'd':
                    if 0 < iters <= 20000:
                        start = datetime.datetime.now()
                        for r in range(iters):
                            pinn.train_d()
                        stop = datetime.datetime.now()
                        timed += stop - start
                else:
                    if 0 < iters <= 20000:
                        start = datetime.datetime.now()
                        for r in range(iters):
                            pinn.train()
                        stop = datetime.datetime.now()
                        timed += stop - start
                print('Iterations complete.')
        inp = input()
    print('PINN trained.')
    print("Training time: " + str(timed))
    print("Iterations: " + str(pinn.iterations))
    print("Avg time per iteration: " + str(timed / pinn.iterations))

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

    plt.plot(loc_L[0], loc_L[1], marker='+', color='red', linestyle='None')
    plt.plot(loc_R[0], loc_R[1], marker='+', color='green', linestyle='None')
    plt.show()

    # visualize results
    pinn.plot_trained_function('Trained function', 'task_CH_function.png', loc_L, loc_R)
    # pinn.plot_error('Error', 'task_CH_error.png', loc_L, loc_R, orig_L, orig_R)
    # pinn.plot_error_hist('Error', 'task_CH_errorh.png', loc_L, loc_R, orig_L, orig_R)

    pinn.plot_equation_loss('Equation loss', 'task_CH_pde_loss.png')
    pinn.plot_equation_loss_hist('Equation loss', 'task_CH_loss_eqh.png')
    pinn.plot_equation_loss_hist_L('Equation loss (L)', 'task_CH_loss_eqh_L.png')
    pinn.plot_equation_loss_hist_R('Equation loss (R)', 'task_CH_loss_eqh_R.png')
    pinn.plot_dirichlet_loss_hist('Dirichlet loss', 'task_CH_loss_dh.png')
    pinn.plot_dirichlet_loss_hist_L('Dirichlet loss (L)', 'task_CH_loss_dh_L.png')
    pinn.plot_dirichlet_loss_hist_R('Dirichlet loss (R)', 'task_CH_loss_dh_R.png')
    pinn.plot_neumann_loss_hist('Neumann loss', 'task_CH_loss_nh.png')
    pinn.plot_neumann_loss_hist_L('Neumann loss (L)', 'task_CH_loss_nh_L.png')
    pinn.plot_neumann_loss_hist_R('Neumann loss (R)', 'task_CH_loss_nh_R.png')
    pinn.plot_transition_loss_hist('Transition loss', 'task_CH_loss_th.png')
    pinn.plot_transition_dx_loss_hist('Transition differential loss', 'task_CH_loss_tdh.png')

    pinn.plot_loss_history_all_L('Loss history (L)', 'task_CH_loss_history_L_all.png')
    pinn.plot_loss_history_all_R('Loss history (R)', 'task_CH_loss_history_R_all.png')
    pinn.plot_loss_history_all_t('Loss history (t)', 'task_CH_loss_history_t_all.png')

    print('Data plotted.')
