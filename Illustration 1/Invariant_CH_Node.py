from statistics import mode
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


def swiss_roll(dataSize, nf=2, noise_std=0, input_ch=None):
    MyData = torch.zeros(dataSize, 2, 1)
    MyLabel = torch.zeros(dataSize, 1)

    for i in range(int(dataSize/2)):
        r = 2
        theta = torch.tensor(i/int(dataSize/2)*1*3.14)

        MyLabel[i, :] = 0
        MyData[i, 0, 0] = r*torch.cos(theta)
        MyData[i,1,0]   = r*torch.sin(theta)
        plt.plot(MyData[i, 0, 0], MyData[i, 1, 0], 'bx',markersize = 12)

        MyLabel[i+int(dataSize/2), :] = 1
        MyData[i+int(dataSize/2), 0, 0] = (r+2)*torch.cos(theta)
        MyData[i+int(dataSize/2), 1, 0] = (r+2)*torch.sin(theta)

        plt.plot(MyData[i+int(dataSize/2), 0, 0],
                    MyData[i+int(dataSize/2), 1, 0], 'ro',markersize = 12)


    domain = [-1.2*4.0, 1.2*4.0, -1.2*4.0, 1.2*4.0]
    return MyData, MyLabel, domain 


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
    
    with torch.no_grad():
        labels = torch.ge(model_c(model(data2d)), 0).int()
    plt.contourf(xa, ya, labels.view([N, N]), levels=[-0.5, 0.5, 1.5], colors=['#99C4E2', '#EAB5A0'])


class H1(nn.Module):
    # Hamiltonian neural network, as presented in [1,2].
    # C-HNN
    # General ODE: \dot{y} = (J- \gamma I ) (K(t) \tanh( K^T(t) y(t) + b(t) ) + (M^T(t)M(t) + \kappa I ) y(t))
    # Constraints:
    #   J(y,t) = J_1 = [ 0 I ; -I 0 ]
    # Discretization method: Forward Euler
    def __init__(self, n_layers, t_end, nf):
        super().__init__()

        self.n_layers = n_layers  # nt: number of layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        self.nf = nf
        self.I = torch.eye(self.nf)
        self.r = torch.tensor(0.0)
        self.eps = 1e-9
        self.kappa = 0.2

        K = torch.randn(self.nf, self.nf)
        M = torch.eye(self.nf)
        b = torch.randn(self.nf, 1)
   

        self.K = nn.Parameter(K, True)
        self.M = self.kappa*M
        self.b = nn.Parameter(b, True)

        j_identity = torch.eye(self.nf//2)
        j_zeros = torch.zeros(self.nf//2, self.nf//2)
        self.J = torch.cat((torch.cat((j_zeros, j_identity), 0), torch.cat((- j_identity, j_zeros), 0)), 1)
   

    def getK(self):
        return self.K

    def getb(self):
        return self.b

    def getJ(self):
        return self.J
    
    def update_r(self):
        with torch.no_grad():
            
            eig_max_KtK = 0
            eig_max_MtM = 0
            eig_min_MtM = 10 
          
            eig, _ = torch.linalg.eig(torch.matmul(self.K.T, self.K))
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
                Y, self.K.transpose(0, 1), self.b[:, 0])), torch.matmul(F_NN, self.K)) \
                    + self.h * F.linear(F.linear(Y,MtM.transpose(0, 1)),F_NN)

        NNoutput = Y.transpose(1, dim-1)
        

        return NNoutput
    def net(self,x):
        dim = len(x.shape)
        Y = x.transpose(1, dim-1)

        F_NN = self.J - self.r*self.I
        MtM = torch.matmul(self.M.T, self.M)

        Y = F.linear(self.act(F.linear(
                Y, self.K.transpose(0, 1), self.b[:, 0])), torch.matmul(F_NN, self.K)) \
                    + F.linear(F.linear(Y,MtM.transpose(0, 1)),F_NN)
        return  Y



def train_2d_example(dataset='swiss_roll', net_type='H1', nf=4, n_layers=8, t_end=1,
                     seed=None):

    
    data_gen = swiss_roll   
    out = 1
    # Set seed
    seed = 2 
    torch.manual_seed(seed)
    np.random.seed(seed)

    # define data
    data_size = 6
    train_data_size = 6
    
    data2d, labels, domain = data_gen(data_size, nf=nf)

    partition = {'train': range(0, data_size)}

    # # Select training parameters
    alpha = 0.1e-4
    alphac = 0.1e-4
    learning_rate = 0.5e-1
    max_iteration = 60
    max_in_iteration = 30

    # define network structure and optimizer
    batch_size = 6
    training_set = Dataset(partition['train'], data2d, labels)
    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

    h = t_end / n_layers

    model = H1(n_layers, t_end, nf=nf)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer_k = torch.optim.Adam(model.parameters(), lr=learning_rate)  # , weight_decay=alpha/100)

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

            optimizer_k.zero_grad()
            K = model.getK()
            b = model.getb()
            loss = loss_func(model_c(model(local_samples)), local_labels)
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

    return model, model_c, train_acc, data2d, labels, partition, domain



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='swiss_roll')
    parser.add_argument('--nf', type=int, default= 2)
    parser.add_argument('--net_type', type=str, default='H1')
    parser.add_argument('--n_layers', type=int, default= 4)
    parser.add_argument('--t_end', type=float, default= 0.03)
    parser.add_argument('--gradient_info', type=bool, default=True)
    args = parser.parse_args()

    # # Train the network
    model, model_c, train_acc, data2d, label, partition, domain = \
        train_2d_example(args.dataset, args.net_type, args.nf, args.n_layers, args.t_end)

    # Print classification results
    print('Accuracy: %.2f%%' % (train_acc*100))

    # # Plot classification results
    plt.figure(1)
    viewContour2D(domain, model, model_c)
    plt.xlim([-3, 4.8])
    plt.ylim([-2, 4.8])
    plt.xticks([])
    plt.yticks([])
    plt.savefig('swiss_roll_contractive_contour.pdf')


    plt.figure(2)
    # # # below calculate the successful prediction rate on the training data
    
    #  plot points around the train point and the propagation
    number_of_samples = 50
    radius = 0.3
    angle = 2*3.14*torch.linspace(0, 1, number_of_samples)
    for i in range(0, len(data2d), 1):
        if label[i] == 1.0:                                   
            plt.plot(data2d[i, 0, 0], data2d[i, 1, 0], 'ro', markersize = 10)
        else:
            plt.plot(data2d[i, 0, 0], data2d[i, 1, 0], 'bx', markersize = 10)
        

        sample = data2d[i].view(2) + torch.stack([radius*torch.cos(angle),
                                             radius*torch.sin(angle)]).T
        plt.plot(sample[:, 0], sample[:, 1], '--', color = 'purple')
        YN = model(sample).detach()
        plt.plot(YN[:, 0], YN[:, 1], '--', color = 'orange')
        

      # plot the classification boundary
    delta = 0.025
    xrange = torch.arange(-5.0, 5.0, delta)
    yrange = torch.arange(-5.0, 5.0, delta)
    x, y = torch.meshgrid(xrange, yrange)
    y1 = np.stack([x.flatten(), y.flatten()])
    y1 = np.expand_dims(y1.T, axis=2)
    data2d2 = torch.from_numpy(y1).float()
    nf = model.nf

    with torch.no_grad():
        labels = torch.ge(model_c(data2d2), 0).int()
    plt.contour(x, y, labels.view([len(xrange), len(xrange)]), linestyles={'dashed'},
    )


    # plot the trajectory
    Y_out23 = get_intermediate_states(model, data2d)
    traj = torch.empty(model.n_layers+1,data2d.shape[1])
    for idx2 in range(0,data2d.shape[0]):
        for idx in range(0,model.n_layers+1): 
            traj[idx,:] =  torch.tensor([Y_out23[idx][idx2].detach()[0],
                                            Y_out23[idx][idx2].detach()[1]])
        if label[idx2] == 1.0:                                   
            plt.plot(traj[:,0],traj[:,1], '--', color='black')
        else:
            plt.plot(traj[:,0],traj[:,1],'-.',color ='black')
    
    # # plotting the quiver
    xv, yv = torch.meshgrid(torch.linspace(-5, 5, 20),
                                 torch.linspace(-5, 5, 20))

    y1_quiver = torch.stack([xv.flatten(), yv.flatten()])
    
    vector_field = model.net(y1_quiver.T)
    
    u = vector_field[:, 0].detach().numpy().reshape(xv.size())
    v = vector_field[:, 1].detach().numpy().reshape(xv.size())

    plt.quiver(xv,yv,u,v,color = 'grey')
    plt.xlim([-3, 4.8])
    plt.ylim([-2, 4.8])
    plt.xticks([])
    plt.yticks([])
    plt.savefig('node_propatation_contractive_swiss_roll.pdf')
    plt.show()
