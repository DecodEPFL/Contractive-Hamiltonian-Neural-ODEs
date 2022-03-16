import torch
import time
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch.autograd.functional as AGF
from pytorch_lightning.callbacks import EarlyStopping
import torch.linalg as linalg
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

global_max_epoches = 1000
global_train_flag = True
global_data_size = 6
global_optimizer_lr = 5e-2
torch.manual_seed(55)

class TrainDataset(Dataset):
    def __init__(self, dataSize):
        MyData = torch.zeros(dataSize, 2)
        MyLabel = torch.zeros(dataSize, 1)

        for i in range(int(dataSize/2)):
            r = 2
            theta = torch.tensor(i/int(dataSize/2)*1*3.14)

            MyLabel[i, :] = 0
            MyData[i, :] = torch.tensor(
                [r*torch.cos(theta), r*torch.sin(theta)])
            plt.plot(MyData[i, 0], MyData[i, 1], 'bx',markersize=12)

            MyLabel[i+int(dataSize/2), :] = 1
            MyData[i+int(dataSize/2), :] = torch.tensor(
                [(r+2)*torch.cos(theta), (r+2)*torch.sin(theta)])

            plt.plot(MyData[i+int(dataSize/2), 0],
                     MyData[i+int(dataSize/2), 1], 'ro', markersize=12)

        number_of_samples = 50
        radius = 0.3
        angle = 2*3.14*torch.linspace(0, 1, number_of_samples)
        for i in range(0, len(MyData), 1):

            sample = MyData[i]+torch.stack([radius*torch.cos(angle),
                                            radius*torch.sin(angle)]).T

        randindex = torch.randperm(dataSize)
        self.MyData = MyData[randindex, :]
        self.MyLabel = MyLabel[randindex, :]

    def __len__(self):
        return len(self.MyData)

    def __getitem__(self, idx):
        return self.MyData[idx, :], self.MyLabel[idx, :]


class classification_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, t, x):
        return self.net(x)

class swiss_roll_node(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.t = torch.linspace(0., 0.5, 10)
        self.func = ODEFunc()
        self.classification_layer = classification_layer()

    def node_propagation(self, x0):

        # traj_x = odeint(self.func, x0, t, method='dopri5')
        traj_x = odeint(self.func, x0, self.t, method='euler')
        return traj_x

    def forward(self, x0):
        # the final value of node propagation is input to the classification and the sigmoid function
        prediction_probability = torch.sigmoid(
            self.classification_layer(self.node_propagation(x0)[-1]))
        # output is in the raneg of (0,1)
        return prediction_probability

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(
            self.func.parameters(), lr=global_optimizer_lr)

        optimizer2 = torch.optim.Adam(
            self.classification_layer.parameters(), lr=global_optimizer_lr)

        def lambda1(epoch): return 0.99 ** epoch

        scheduler1 = torch.optim.lr_scheduler.LambdaLR(
            optimizer1, lr_lambda=lambda1)
        scheduler2 = torch.optim.lr_scheduler.LambdaLR(
            optimizer2, lr_lambda=lambda1)

        return [optimizer1, optimizer2], [scheduler1, scheduler2]

    def training_step(self, train_batch, batch_idx, optimizer_idx):

        lossFunc = nn.BCELoss()

        x0, label = train_batch
        loss = lossFunc(self.forward(x0), label)

        
        return loss

    def test_step(self, batch, batch_idx):

        data, label = batch

        # below calculate the successful prediction rate on the training data
        success_rate = (torch.ge(self.forward(data), 0.5).int()
                        == label).sum()/len(label)

        self.log("success_rate", success_rate)

        # plot points around the train point and the propatation
        number_of_samples = 50
        radius = 0.3
        angle = 2*3.14*torch.linspace(0, 1, number_of_samples)
        for i in range(0, len(data), 1):
            if label[i] == 0:
                plt.plot(data[i, 0], data[i, 1], 'bx',markersize = 5)
            if label[i] == 1:
                plt.plot(data[i, 0], data[i, 1], 'ro',markersize = 5)
            # plt.plot(data[i, 0], data[i, 1], '+')

            sample = data[i]+torch.stack([radius*torch.cos(angle),
                                          radius*torch.sin(angle)]).T
            plt.plot(sample[:, 0], sample[:, 1], '--',color = 'purple')
            traj = self.node_propagation(sample)
            plt.plot(traj[-1, :, 0], traj[-1, :, 1], '--',color = 'orange')

        # plot the classification boundary
        delta = 0.025
        xrange = torch.arange(-5.0, 5.0, delta)
        yrange = torch.arange(-5.0, 5.0, delta)
        x, y = torch.meshgrid(xrange, yrange)

        equation = self.classification_layer(
            torch.stack([x.flatten(), y.flatten()]).T)-0.5
        plt.contour(x, y, equation.reshape(x.size()),
                    [0], linestyles={'dashed'})

        # plot the trajectory

        data_propagation = self.node_propagation(data)
        for i in range(len(data)):
            ith_traj = data_propagation[:, i, :]
            if label[i] == 0:
                plt.plot(ith_traj[:, 0].cpu(),
                         ith_traj[:, 1].cpu(), '--', color='black')
            if label[i] == 1:
                plt.plot(ith_traj[:, 0].cpu(),
                         ith_traj[:, 1].cpu(), '-.', color='black')

        xv, yv = torch.meshgrid(torch.linspace(-20, 20, 30),
                                torch.linspace(-20, 20, 30))

        y1 = torch.stack([xv.flatten(), yv.flatten()])

        vector_field = self.func.net(y1.T)
        u = vector_field[:, 0].reshape(xv.size())
        v = vector_field[:, 1].reshape(xv.size())
        print('shape of u',u.shape)

        plt.quiver(xv, yv, u, v, color='grey')
        plt.xticks([])
        plt.yticks([])
        plt.xlim([-12.5, 13])
        plt.ylim([-5, 7])

        plt.savefig('node_propatation_neural_ODE.pdf')

        return success_rate

# data

if __name__ == '__main__':

    plt.figure
    training_data = TrainDataset(dataSize=global_data_size)
    train_dataloader = DataLoader(
        training_data, batch_size=global_data_size)

    model = swiss_roll_node()

    trainer = pl.Trainer(gpus=None,  num_nodes=1,
                         max_epochs=global_max_epoches)

    if global_train_flag:
        trainer.fit(model, train_dataloader)
        trainer.save_checkpoint("example_pendulum.ckpt")

    time.sleep(5)

    new_model = swiss_roll_node.load_from_checkpoint(
        checkpoint_path="example_pendulum.ckpt")

    domain = [-10, 10, -10, 10]
    N = 200
    xa = np.linspace(domain[0], domain[1], N)
    ya = np.linspace(domain[2], domain[3], N)
    xv, yv = np.meshgrid(xa, ya)
    y = np.stack([xv.flatten(), yv.flatten()])
    y = np.expand_dims(y.T, axis=1)
    data2d = torch.from_numpy(y).float()

    with torch.no_grad():
        labels = torch.ge(new_model(data2d), 0.5).int()
    plt.contourf(xa, ya, labels.view(
        [N, N]), levels=[-0.5, 0.5, 1.5], colors=['#99C4E2', '#EAB5A0'])
    plt.xticks([])
    plt.yticks([]) 
    plt.xlim([-3, 4.8])
    plt.ylim([-2, 4.8])
    plt.savefig('swiss_roll_neural_ode.pdf')
    print("classification boundry ploted")

    plt.figure(2)
    trainer = pl.Trainer(gpus=None,  num_nodes=1,
                         max_epochs=global_max_epoches)
    success_rate = trainer.test(new_model, train_dataloader)
    print("after training, the successful predition rate on train set is", success_rate)
    plt.show()