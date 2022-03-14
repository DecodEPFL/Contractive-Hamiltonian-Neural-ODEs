import torch
from torch import no_grad
import torch.nn.functional as F
from torch import nn
import numpy as np
from scipy.sparse import coo_matrix
from torch.utils import data
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import argparse
from skimage.util import random_noise 

def _data_shuffle(data2d, label):
    data_size = data2d.shape[0]
    randindex = torch.randperm(data_size)
    data2d = data2d[randindex, :, :]
    label = label[randindex, :]
    return data2d, label


def _data_extension(data2d, nf, input_ch=None):
    if nf < 2:
        print("Dimension not valid")
        return
    elif nf % 2 == 1:
        print("Using odd dimension nf")
    data_size = data2d.shape[0]
    if input_ch is not None:
        # input_ch is a list of two elements. The elements indicate where the data enters.
        idx_x = input_ch[0]
        idx_y = input_ch[1]
    else:
        idx_x = 0
        idx_y = nf-1
    data2d = torch.cat((torch.zeros(data_size, idx_x-0, 1),
                        data2d[:, 0:1, :],
                        torch.zeros(data_size, idx_y-idx_x-1, 1),
                        data2d[:, 1:2, :],
                        torch.zeros(data_size, nf-1-idx_y, 1)), 1)
    return data2d

def double_circles(data_size, shuffle=True, nf=2, noise_std=0, input_ch=None):

    data2d = torch.zeros(data_size, 2, 1)
    label = torch.zeros(data_size, 1)

    for i in range(int(data_size / 4)):
        theta = torch.tensor(i / int(data_size / 4) * 4 * 3.14)

        r = 1
        label[i, :] = 0
        data2d[i, :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

        r = 2
        label[i + int(data_size / 4), :] = 1
        data2d[i + int(data_size / 4), :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

        r = 3
        label[i + int(2 * data_size / 4), :] = 0
        data2d[i + int(2 * data_size / 4), :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

        r = 4
        label[i + int(3 * data_size / 4), :] = 1
        data2d[i + int(3 * data_size / 4), :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

    if noise_std:
        for i in range(2):
            data2d[:, i, 0] = data2d[:, i, 0] + noise_std*torch.randn(data_size)
    
    if shuffle:
        data2d, label = _data_shuffle(data2d, label)

    if nf != 2:
        data2d = _data_extension(data2d, nf, input_ch)
    
    domain = [-5, 5, -5, 5]
    return data2d, label, domain    

class Dataset(data.Dataset):

    def __len__(self):
        return len(self.list_ids)

    def __init__(self, list_ids, data_in, labels):
        self.list_ids = list_ids
        self.data = data_in
        self.labels = labels

    def __getitem__(self, index):

        idx = self.list_ids[index]

        x = self.data[idx, :, :]
        y = self.labels[idx, :]

        return x, y


def regularization(alpha, h, K, b):
    # Regularization function as introduced in [1]
    n_layers = K.shape[-1]
    loss = 0
    for j in range(n_layers - 1):
        loss = loss + alpha * h * (1 / 2 * torch.norm(K[:, :, j + 1] - K[:, :, j]) ** 2 +
                                   1 / 2 * torch.norm(b[:, :, j + 1] - b[:, :, j]) ** 2)
    return loss

def get_intermediate_states(model, Y0):
    Y0.requires_grad = True
    # Y_out N-element list containing the intermediates states. Size of each entry: n_samples * dim2 * dim1
    # Y_out[n] = torch.zeros([batch_size, nf, 1]), with n=0,1,..,
    Y_out = [Y0]
    i = 0
    for j in range(model.n_layers):
        Y = model.forward(Y_out[j], ini=j, end=j + 1)
        Y_out.append(Y)
        Y_out[j + 1].retain_grad()
    return Y_out


class Classification(nn.Module):
    def __init__(self, nf=2, nout=1):
        super().__init__()
        self.nout = nout
        self.W = nn.Parameter(torch.zeros(self.nout, nf), True)
        self.mu = nn.Parameter(torch.zeros(1, self.nout), True)

    def forward(self, Y0):
        Y = Y0.transpose(1, 2)
        NNoutput = F.linear(Y, self.W, self.mu).squeeze(1)
        return NNoutput

def viewContour2D(domain, model, model_c, input_ch=None):
    '''
    Coloured regions in domain represent the prediction of the DNN given the Hamiltonian net (model) and the output
    layer (modelc).
    input_ch indicates the indexes where the input data is plugged
    For 2d datasets.
    '''
    N = 200
    xa = np.linspace(domain[0], domain[1], N)
    ya = np.linspace(domain[2], domain[3], N)
    xv, yv = np.meshgrid(xa, ya)
    y = np.stack([xv.flatten(), yv.flatten()])
    y = np.expand_dims(y.T, axis=2)
    data2d = torch.from_numpy(y).float()
    nf = model.nf
    if nf != 2:
        data2d = _data_extension(data2d, nf, input_ch)
    
    with torch.no_grad():
        labels = torch.ge(model_c(model(data2d)), 0).int()
    plt.contourf(xa, ya, labels.view([N, N]), levels=[-0.5, 0.5, 1.5], colors=['#EAB5A0', '#99C4E2'])


def viewTestData(partition, data2d, labels, input_ch=None):
    if input_ch is not None:
        # input_ch is a list of two elements. The elements indicate where the data enters.
        idx_x = input_ch[0]
        idx_y = input_ch[1]
    else:
        nf = data2d.shape[1]
        idx_x = 0
        idx_y = nf-1
    # Plot test data for 2d datasets.
    testDataSize = len(partition['test'])
    mask0 = (labels[partition['test'], 0] == 0).view(testDataSize)
    plt.plot(data2d[partition['test'], idx_x, :].view(testDataSize).masked_select(mask0),
             data2d[partition['test'], idx_y, :].view(testDataSize).masked_select(mask0), 'r+',
             markersize=2)

    mask1 = (labels[partition['test'], 0] == 1).view(testDataSize)
    plt.plot(data2d[partition['test'], idx_x, :].view(testDataSize).masked_select(mask1),
             data2d[partition['test'], idx_y, :].view(testDataSize).masked_select(mask1), 'b+',
             markersize=2)


def viewPropagatedPoints(model, partition, data2d, labels, input_ch=None):
    if input_ch is not None:
        # input_ch is a list of two elements. The elements indicate where the data enters.
        idx_x = input_ch[0]
        idx_y = input_ch[1]
    else:
        nf = data2d.shape[1]
        idx_x = 0
        idx_y = nf-1
    test_data_size = labels[partition['test'], 0].size(0)
    mask0 = (labels[partition['test'], 0] == 0).view(test_data_size)
    YN = model(data2d[partition['test'], :, :]).detach()
    plt.plot(YN[:, idx_x, :].view(test_data_size).masked_select(mask0),
                YN[:, idx_y, :].view(test_data_size).masked_select(mask0), 'r+')

    mask1 = (labels[partition['test'], 0] == 1).view(test_data_size)
    plt.plot(YN[ :, idx_x, :].view(test_data_size).masked_select(mask1),
                YN[ :, idx_y, :].view(test_data_size).masked_select(mask1), 'b+')


def plot_grad_x_layer(gradients_matrix, colorscale=False, log=True):
    # Plot the gradient norms at each layer (different colors = different iterations)
    [tot_iters, nf, _, n_layers1] = gradients_matrix.shape
    n_layers = n_layers1 - 1

    if not colorscale:
        plt.figure()
        z = np.linspace(1, n_layers, n_layers)
        legend = []
        for ii in range(1, tot_iters, 100):
            plt.plot(z, np.linalg.norm(gradients_matrix[ii, :, :, :], axis=(0, 1), ord=2)[1:])
            legend.append("Iteration %s" % str(ii))
        for ii in range(1, tot_iters, 1):
            if np.linalg.norm(gradients_matrix[ii, :, :, :], axis=(0, 1), ord=2)[1:].sum() == 0:
                print("zero found at %s" % str(ii))
        plt.xlabel("Layers")
        plt.ylabel(r'$\left\|\frac{\partial y_N}{\partial y_\ell}\right\|$', fontsize=12)
        if log:
            plt.yscale('log')
        plt.legend(legend)
    else:
        z = np.linspace(1, n_layers, n_layers)
        fig, ax = plt.subplots()
        n = tot_iters
        # setup the normalization and the colormap
        normalize = mcolors.Normalize(vmin=1, vmax=n)
        colormap = cm.get_cmap('jet', n - 1)
        legend = ['Lower bound']
        ax.plot([1, n_layers], [1, 1], 'k--')
        plt.legend(legend)
        for ii in range(1, n, 1):
            ax.plot(z, np.linalg.norm(gradients_matrix[ii, :, :, :], axis=(0, 1), ord=2)[1:],
                    color=colormap(normalize(ii)),
                    linewidth=0.5)
        plt.xlabel("Layer $\ell$")
        plt.ylabel(r'$\left\|\frac{\partial y_N}{\partial y_\ell}\right\|$', fontsize=12)
        if log:
            plt.yscale('log')
        # setup the colorbar
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        cb = plt.colorbar(scalarmappaple)
        cb.set_label('# iteration')
        plt.tight_layout()


def plot_grad_x_iter(gradients_matrix, colorscale=False, log=True, one_line=True):
    # Plot the gradient norms at each iteration (different colors = different layers)
    [tot_iters, nf, _, n_layers1] = gradients_matrix.shape
    n_layers = n_layers1 - 1

    if not colorscale:
        plt.figure()
        z = np.linspace(1, tot_iters-1, tot_iters-1)
        legend = []
        for ii in range(1, n_layers):
            plt.plot(z, np.linalg.norm(gradients_matrix[:, :, :, ii], axis=(1, 2), ord=2)[1:])
            legend.append("Layer %s" % str(ii))
        plt.xlabel("Iteration")
        plt.ylabel(r'$\|\|\frac{\partial y_N}{\partial y_\ell}\|\|$', fontsize=12)
        if log:
            plt.yscale('log')
        plt.legend(legend)
        return legend
    else:
        x = np.linspace(0, tot_iters - 1, tot_iters)
        fig, ax = plt.subplots()
        n = n_layers
        # setup the normalization and the colormap
        normalize = mcolors.Normalize(vmin=1, vmax=n)
        colormap = cm.get_cmap('jet', n - 1)
        if one_line:
            legend = ['Upper bound']
            ax.plot([0, gradients_matrix.shape[0]], [1, 1], 'k--')
            plt.legend(legend)
        for ii in range(1, n_layers, 1):
            j = n_layers-ii
            ax.plot(x, np.linalg.norm(gradients_matrix[:, :, :, j], axis=(1, 2), ord=2), color=colormap(normalize(ii)),
                    linewidth=0.5)
        plt.xlabel("Iterations")
        plt.ylabel(r'$\|\|\frac{\partial \xi_N}{\partial \xi_{N-\ell}}\|\|$', fontsize=12)
        if log:
            plt.yscale('log')
        # setup the colorbar
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        cb = plt.colorbar(scalarmappaple)
        cb.set_label('Depth $\ell$')
        plt.tight_layout()

class R1(nn.Module):
    # ResNet
    # General ODE: \dot{y} =  \tanh( K(t) y(t) + b(t) )
    # Constraints:
    # Discretization method: Forward Euler
    def __init__(self, n_layers, t_end, nf, random=True, select_j='J1'):
        super().__init__()

        self.n_layers = n_layers  # nt: number of layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        self.nf = nf

        if random:
            K = torch.randn(self.nf, self.nf, self.n_layers)
            b = torch.randn(self.nf, 1, self.n_layers)
        else:
            K = torch.ones(self.nf, self.nf, self.n_layers)
            b = torch.zeros(self.nf, 1, self.n_layers)

        self.K = nn.Parameter(K, True)
        self.b = nn.Parameter(b, True)


    def getK(self):
        return self.K

    def getb(self):
        return self.b


    def forward(self, Y0, ini=0, end=None):

        dim = len(Y0.shape)
        Y = Y0.transpose(1, dim-1)

        if end is None:
            end = self.n_layers

        for j in range(ini, end):
            Y = Y + self.h * self.act(F.linear(
                Y, self.K[:, :, j].transpose(0, 1), self.b[:, 0, j]))

        NNoutput = Y.transpose(1, dim-1)

        return NNoutput

    def update_r(self):
        return 0        

class H1(nn.Module):
    # Hamiltonian neural network, as presented in [1,2].
    # H_1-DNN and H_2-DNN
    # General ODE: \dot{y} = J(y,t) K(t) \tanh( K^T(t) y(t) + b(t) )
    # Constraints:
    #   J(y,t) = J_1 = [ 0 I ; -I 0 ]  or  J(y,t) = J_2 = [ 0 1 .. 1 ; -1 0 .. 1 ; .. ; -1 -1 .. 0 ].
    # Discretization method: Forward Euler
    def __init__(self, n_layers, t_end, nf, random=True, select_j='J1'):
        super().__init__()

        self.n_layers = n_layers  # nt: number of layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        self.nf = nf
        self.I = torch.eye(self.nf)
        self.r = torch.tensor(0.0)
        self.eps = 1e-9

        if random: 
            K = torch.randn(self.nf, self.nf, self.n_layers)
            M = torch.eye(self.nf)
            b = torch.randn(self.nf, 1, self.n_layers)
        else:
            K = torch.ones(self.nf, self.nf, self.n_layers)
            M = torch.eye(self.nf)
            b = torch.zeros(self.nf, 1, self.n_layers)

        self.K = nn.Parameter(K, True)
        self.M = 0.2*M
        self.b = nn.Parameter(b, True)

        if select_j == 'J1':
            j_identity = torch.eye(self.nf//2)
            j_zeros = torch.zeros(self.nf//2, self.nf//2)
            self.J = torch.cat((torch.cat((j_zeros, j_identity), 0), torch.cat((- j_identity, j_zeros), 0)), 1)
        else:
            j_aux = np.hstack((np.zeros(1), np.ones(self.nf-1)))
            J = j_aux
            for j in range(self.nf-1):
                j_aux = np.hstack((-1 * np.ones(1), j_aux[:-1]))
                J = np.vstack((J, j_aux))
            self.J = torch.tensor(J, dtype=torch.float32)

    def getK(self):
        return self.K

    def getb(self):
        return self.b

    def getJ(self):
        return self.J

    def getM(self):
        return self.M

    
    def update_r(self):
        with torch.no_grad():
            
            eig_max_KtK = 0
            eig_max_MtM = 0
            eig_min_MtM = 10 
            for i in range(self.K.shape[2]):
                
                eig, _ = torch.linalg.eig(torch.matmul(self.K[:,:,i].T, self.K[:,:,i]))
                eig_M, _ = torch.linalg.eig(torch.matmul(self.M.T, self.M))

                eig_max = torch.max(torch.abs(eig))
                eig_max_M = torch.max(torch.abs(eig_M))
                eig_min_M = torch.min(torch.abs(eig_M))
                eig_max_KtK = max(eig_max, eig_max_KtK)
                eig_max_MtM = max(eig_max_M, eig_max_MtM)
                eig_min_MtM = min(eig_min_M, eig_min_MtM)
                
            c2 = eig_max_KtK + eig_max_MtM 
            c1 = eig_min_MtM 
            alpha = (c2 - c1)/(c2 + c1)
            self.r = torch.sqrt((alpha**2 - self.eps)/(1 - alpha**2 - self.eps))
        self.r = self.r
        

    def forward(self, Y0, ini=0, end=None):

        dim = len(Y0.shape)
        Y = Y0.transpose(1, dim-1)

        if end is None:
            end = self.n_layers

        for j in range(ini, end):
            
            F_NN = self.J - self.r*self.I
            MtM = torch.matmul(self.M.T, self.M)

            Y = Y + self.h * F.linear(self.act(F.linear(
                Y, self.K[:, :, j].transpose(0, 1), self.b[:, 0, j])), torch.matmul(F_NN, self.K[:, :, j])) \
                    + self.h * F.linear(F.linear(Y,MtM.transpose(0, 1)),F_NN)

        NNoutput = Y.transpose(1, dim-1)
        

        return NNoutput



def train_2d_example(dataset='double_circles', net_type='H1', nf=4, n_layers=8, t_end=1, gradient_info=False, sparse=None,
                     seed=None):

 
  
    data_gen = double_circles


    out = 1

    # Set seed
    seed = 1 
    torch.manual_seed(seed)
    np.random.seed(seed)

    # define data
    data_size = 8000
    train_data_size = 4000
    test_data_size = data_size - train_data_size
    if sparse is None:
        data2d, labels, domain = data_gen(data_size, nf=nf)

    partition = {'train': range(0, data_size, 2),
                  'test': range(1, data_size, 2)}

    # # Select training parameters
    alpha = 0.1e-4
    alphac = 0.1e-4
    learning_rate = 0.5e-1
    max_iteration = 50
    max_in_iteration = 60

    # define network structure and optimizer
    batch_size = 250
    training_set = Dataset(partition['train'], data2d, labels)
    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

    h = t_end / n_layers

    model = H1(n_layers, t_end, nf=nf, select_j='J1')


    loss_func = nn.BCEWithLogitsLoss()
    optimizer_k = torch.optim.Adam(model.parameters(), lr=learning_rate)  # , weight_decay=alpha/100)

    if gradient_info:
        loss_func2 = nn.Identity()
        gradients_matrix = np.zeros([int(train_data_size/batch_size) * max_iteration, model.nf, model.nf, n_layers + 1])
    else:
        gradients_matrix = None

    # check before correct rate
    print('%s example using a %d-layer %s-DNN with %d features. Alpha=%.1e. Final_time=%.2f'
            % (dataset, n_layers, net_type, nf, alpha, t_end))

    # Training network
    for epoch in range(max_iteration):

        training_iterator = iter(training_generator)

        for i_k in range(int(data2d[partition['train']].size(0) / training_generator.batch_size)):

            local_samples, local_labels = next(training_iterator)

            model_c = Classification(nf=nf)
            optimizer_w = torch.optim.Adam(model_c.parameters(), lr=learning_rate)
            with torch.no_grad():
                YN = model(local_samples)

            for i_w in range(max_in_iteration):  # Inner iteration

                optimizer_w.zero_grad()
                loss = loss_func(model_c(YN), local_labels)
                loss = loss + alphac * 0.5 *(torch.norm(model_c.W) ** 2 + torch.norm(model_c.mu) ** 2)
                loss.backward()
                optimizer_w.step()

            if gradient_info:
                local_samples.requires_grad = True
                matrix_aux = np.zeros([model.nf, model.nf, n_layers + 1])
                for k in range(model.nf):
                    model.update_r()
                    optimizer_k.zero_grad()
                    Y_out = get_intermediate_states(model, local_samples)
                    YN = Y_out[-1]
                    loss = loss_func2(YN[:, k, 0].sum())
                    loss.backward()
                    for j in range(n_layers + 1):
                        matrix_aux[:, k, j] = Y_out[j].grad[:, :, 0].numpy().sum(axis=0) / training_generator.batch_size
                gradients_matrix[epoch * int(train_data_size / batch_size) + i_k, :, :, :] = matrix_aux
                local_samples.requires_grad = False

            optimizer_k.zero_grad()
            K = model.getK()
            b = model.getb()
            loss = loss_func(model_c(model(local_samples)), local_labels)
            loss += regularization(alpha, h, K, b)
            loss.backward()
            li = list(optimizer_k.state)
            if not (len(li) == 0):
                for ii in range(2):
                    optimizer_k.state[li[ii]]['step'] = epoch
            optimizer_k.step()
            model.update_r()

        if epoch % 10 == 0 and out > 0:
            model_c = Classification(nf=nf)
            optimizer_w = torch.optim.Adam(model_c.parameters(), lr=learning_rate)
            with torch.no_grad():
                YN = model(local_samples)
            for i_w in range(max_in_iteration):  # Inner iteration
                optimizer_w.zero_grad()
                loss = loss_func(model_c(YN), local_labels)
                loss = loss + alphac * 0.5 * (torch.norm(model_c.W) ** 2 + torch.norm(model_c.mu) ** 2)
                loss.backward()
                optimizer_w.step()
                acc = (torch.ge(model_c(model(local_samples)), 0) == local_labels).sum().numpy() / batch_size
            print('\tTrain Epoch: {:2d} - Loss: {:.6f} - Accuracy: {:.0f}%'.format(epoch, loss, acc*100))

    # Train classification layer with all the data

    model_c = Classification(nf=nf)
    optimizer_w = torch.optim.Adam(model_c.parameters(), lr=learning_rate)

    for epoch in range(max_iteration):

        training_iterator = iter(training_generator)

        for i_w in range(int(data2d[partition['train']].size(0) / training_generator.batch_size)):

            local_samples, local_labels = next(training_iterator)
            with torch.no_grad():
                YN = model(local_samples)

            optimizer_w.zero_grad()
            loss = loss_func(model_c(YN), local_labels)
            loss = loss + alphac * 0.5 * (torch.norm(model_c.W) ** 2 + torch.norm(model_c.mu) ** 2)
            loss.backward()
            optimizer_w.step()
            
            

    # Accuracy results

    with torch.no_grad():
        train_acc = (torch.ge(model_c(model(data2d[partition['train'], :, :])), 0) == labels[partition['train'], :]
                     ).sum().numpy() / train_data_size
        test_acc = (torch.ge(model_c(model(data2d[partition['test'], :, :])), 0) == labels[partition['test'], :]
                    ).sum().numpy() / test_data_size

    return model, model_c, train_acc, test_acc, data2d, labels, partition, domain, gradients_matrix




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='double_circles')
    parser.add_argument('--nf', type=int, default= 4)
    parser.add_argument('--net_type', type=str, default='H1')
    parser.add_argument('--n_layers', type=int, default= 16)
    parser.add_argument('--t_end', type=float, default= 0.01)
    parser.add_argument('--gradient_info', type=bool, default=True)
    args = parser.parse_args()

    # # Train the network
    model, model_c, train_acc, test_acc, data2d, label, partition, domain, gradients_matrix = \
        train_2d_example(args.dataset, args.net_type, args.nf, args.n_layers, args.t_end, args.gradient_info)

    # Print classification results
    print('Train accuracy: %.2f%% - Test accuracy: %.2f%%' % (train_acc*100, test_acc*100))

    # # Plot classification results
    plt.figure(1)
    # plt.title('t_end: %.2f - %d layers' % (args.t_end, args.n_layers) + ' - Test acc %.2f%%' % (test_acc * 100))
    viewContour2D(domain, model, model_c)
    viewTestData(partition, data2d, label)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('double_circle_contractive.pdf')

    # # Plot gradients
    if args.gradient_info:
        plot_grad_x_iter(gradients_matrix, colorscale=True, log=True)
    plt.savefig('grads_double_circle_contractive.pdf')
    plt.show()
