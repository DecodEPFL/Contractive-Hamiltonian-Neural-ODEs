#!/usr/bin/env python
"""
Train a Contractive Hamiltonian NN on MNIST dataset.
Author: Muhammad Zakwan (muhammad.zakwan@epfl.ch)
Usage:
python Train_MNIST.py          --net_type      [MODEL NAME]            \
                             --n_layers      [NUMBER OF LAYERS]      \
                             --gpu           [GPU ID]
Flags:
  --net_type: Network model to use. Available options are:  H1_J1, R1, H1_C
  --n_layers: Number of layers for the chosen the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from skimage.util import random_noise 

import argparse

def regularization(alpha, h, K, b):
    # Regularization function as introduced in [1]
    n_layers = K.shape[-1]
    loss = 0
    for j in range(n_layers - 1):
        loss = loss + alpha * h * (1 / 2 * torch.norm(K[:, :, j + 1] - K[:, :, j]) ** 2 +
                                   1 / 2 * torch.norm(b[:, :, j + 1] - b[:, :, j]) ** 2)
    return loss


class H1_contract(nn.Module):
    # Hamiltonian neural network, as presented in [1,2].
    # H_1-DNN and H_2-DNN
    # General ODE: \dot{y} = J(y,t) K(t) \tanh( K^T(t) y(t) + b(t) )
    # Constraints:
    #   J(y,t) = J_1 = [ 0 I ; -I 0 ] 
    # Discretization method: Forward Euler
    def __init__(self, n_layers, t_end, nf, random=True, select_j='J1'):
        super().__init__()

        self.n_layers = n_layers  # nt: number of layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        self.nf = nf
        self.I = torch.eye(self.nf)
        self.r = torch.tensor(0.0)
        self.eps = 1e-6
        self.kappa = 0.02

        if random: 
            K = torch.randn(self.nf, self.nf, self.n_layers)
            M = torch.eye(self.nf)
            b = torch.randn(self.nf, 1, self.n_layers)
        else:
            K = torch.ones(self.nf, self.nf, self.n_layers)
            M = torch.eye(self.nf)
            b = torch.zeros(self.nf, 1, self.n_layers)

        self.K = nn.Parameter(K, True)
        self.M = self.kappa*M
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
            self.r = 1.0/torch.sqrt(1 - alpha**2 - self.eps)
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
        print('nf =',self.nf)

        return NNoutput


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

        if random:
            K = torch.randn(self.nf, self.nf, self.n_layers)
            b = torch.randn(self.nf, 1, self.n_layers)
        else:
            K = torch.ones(self.nf, self.nf, self.n_layers)
            b = torch.zeros(self.nf, 1, self.n_layers)

        self.K = nn.Parameter(K, True)
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

    def forward(self, Y0, ini=0, end=None):

        dim = len(Y0.shape)
        
        Y = Y0.transpose(1, dim-1)

        if end is None:
            end = self.n_layers

        for j in range(ini, end):
            Y = Y + self.h * F.linear(self.act(F.linear(
                Y, self.K[:, :, j].transpose(0, 1), self.b[:, 0, j])), torch.matmul(self.J, self.K[:, :, j]))

        NNoutput = Y.transpose(1, dim-1)

        return NNoutput

    def update_r(self):
        return 0

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

class Net(nn.Module):
    def __init__(self, nf=8, n_layers=4, h=0.5, net_type='H1_J1'):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nf, kernel_size=3, stride=1, padding=1)
        if net_type == 'H1_J1':
            self.hamiltonian = H1(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J1')
        elif net_type == 'H1_J2':
            self.hamiltonian = H1(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J2')
        elif net_type == 'H1_C':
            self.hamiltonian = H1_contract(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J1')
        elif net_type == 'R1':
            self.hamiltonian = R1(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J1')
        else:
            raise ValueError("%s model is not yet implemented for MNIST" % net_type)
        self.fc_end = nn.Linear(nf*28*28, 10)
        self.nf = nf

    def forward(self, x):
        x = self.conv1(x)
        x = self.hamiltonian(x)
        x = x.reshape(-1, self.nf*28*28)
        x = self.fc_end(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch, alpha, out):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        K = model.hamiltonian.getK()
        b = model.hamiltonian.getb()
        for j in range(int(model.hamiltonian.n_layers) - 1):
            loss = loss + regularization(alpha, h, K, b)
        loss.backward()
        optimizer.step()
        model.hamiltonian.update_r()
        if batch_idx % 100 == 0 and out>0:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            print('\tTrain Epoch: {:2d} [{:5d}/{} ({:2.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), correct, len(data)))


def test(model, device, test_loader, out):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if out > 0:
        print('Test set:\tAverage loss: {:.4f}, Accuracy: {:5d}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct


def test_gaussian(model, device, test_loader, out, var=0.05):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # lets corrupt it
            im_data = torch.tensor(random_noise(data, mode='gaussian', mean=0, var=var, clip=True)).float()
            output = model(im_data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if out > 0:
        print('Test Gaussian:\tAverage loss: {:.4f}, Var: {:.3f}, Accuracy: {:5d}/{} ({:.2f}%)'.format(
            test_loss, var, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct


def test_salt_pepper(model, device, test_loader, out, amount = 0.05):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # lets corrupt it
            im_data = torch.tensor(random_noise(data, mode='s&p', amount=amount)).float()
            output = model(im_data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if out > 0:
        print('Test Salt and Pepper:\tAverage loss: {:.4f}, Amount: {:.3f}, Accuracy: {:5d}/{} ({:.2f}%)'.format(
            test_loss, amount, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type', type=str, default='H1_C')
    parser.add_argument('--n_layers', type=int, default= 8)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()  # not no_cuda and
    batch_size = 100
    test_batch_size = 1000
    lr = 0.04
    gamma = 0.8
    epochs = 10
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = 1

    if args.net_type == 'H1_J1':
        h = 0.5
        wd = 4e-3
        alpha = 8e-3
    elif args.net_type == 'H1_J2':
        h = 0.05
        wd = 2e-4
        alpha = 1e-3
    elif args.net_type == 'H1_C':
        h = 0.5
        wd = 2e-4
        alpha = 1e-3
    elif args.net_type == 'R1':
        h = 0.5
        wd = 2e-4
        alpha = 1e-3
    else:
        raise ValueError("%s model is not yet implemented" % args.net_type)

    # Define the net model
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    model = Net(nf=8, n_layers=args.n_layers, h=h, net_type=args.net_type).to(device)

    print("\n------------------------------------------------------------------")
    print("MNIST dataset - %s-DNN - %i layers" % (args.net_type, args.n_layers))
    print("== sgd with Adam (lr=%.1e, weight_decay=%.1e, gamma=%.1f, max_epochs=%i, alpha=%.1e, minibatch=%i)" %
          (lr, wd, gamma, epochs, alpha, batch_size))

    best_acc = 0
    best_acc_train = 0

    # Load train data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    # Load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    # Define optimization algorithm
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler for learning_rate parameter
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Training
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, alpha, out)
        test_acc = test(model, device, test_loader, out)
        # Results over training set after training
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss /= len(train_loader.dataset)
        if out > 0:
            print('Train set:\tAverage loss: {:.4f}, Accuracy: {:5d}/{} ({:.2f}%)'.format(
                train_loss, correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))
        scheduler.step()


    print("\nNetwork trained!")
    print('Test accuracy: {:.2f}%  - Train accuracy: {:.3f}% '.format(
         100. * test_acc / len(test_loader.dataset), 100. * correct / len(train_loader.dataset)))
    print("------------------------------------------------------------------\n")


    #For attack 
    No_attack = 10  # Total number of attack per noise 
    attack_accuracy_gaussian_005 = torch.zeros(1,No_attack)
    attack_accuracy_gaussian_02 = torch.zeros(1,No_attack)
    attack_accuracy_salt_005 = torch.zeros(1,No_attack)
    attack_accuracy_salt_02 = torch.zeros(1,No_attack)

    for attack in range(0,No_attack):

        attack_acc_gauss005 = test_gaussian(model, device, test_loader, out, var=0.05)
        attack_acc_gauss02 = test_gaussian(model, device, test_loader, out, var=0.2)
        attack_acc_salt005 = test_salt_pepper(model, device, test_loader, out, amount=0.05)
        attack_acc_salt02 = test_salt_pepper(model, device, test_loader, out, amount=0.2)

        attack_accuracy_gaussian_005[:,attack] = 100. * attack_acc_gauss005 / len(test_loader.dataset)
        attack_accuracy_gaussian_02[:,attack] = 100. * attack_acc_gauss02 / len(test_loader.dataset)
        attack_accuracy_salt_005[:,attack] = 100. * attack_acc_salt005 / len(test_loader.dataset)
        attack_accuracy_salt_02[:,attack] = 100. * attack_acc_salt02 / len(test_loader.dataset)

    print("\nNetwork trained!")
    print('Test accuracy: {:.2f}%  - Train accuracy: {:.3f}% '.format(
         100. * test_acc / len(test_loader.dataset), 100. * correct / len(train_loader.dataset)))
    print("------------------------------------------------------------------\n")
    print('Gaussian with 0.05: {:.2f}%'.format(attack_accuracy_gaussian_005.mean()))
    print('Gaussian with 0.2: {:.2f}%'.format(attack_accuracy_gaussian_02.mean()))
    print('Salt and pepper with 0.05: {:.2f}%'.format(attack_accuracy_salt_005.mean()))
    print('Salt and pepper with 0.05: {:.2f}%'.format(attack_accuracy_salt_02.mean()))
    print("------------------------------------------------------------------\n")


