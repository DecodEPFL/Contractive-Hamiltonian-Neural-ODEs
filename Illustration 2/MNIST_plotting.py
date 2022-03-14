import torch
from torch.utils.data import DataLoader

# torchvision loaded...!!!
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import datasets
import torchvision.transforms as transforms

# other  libraries...!!!
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

torch.manual_seed(5)

dataset = 'mnist'
BATCH_SIZE = 4

dataset = 'mnist'
transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
trainset = datasets.MNIST('../data',train=True, download=True, transform=transform)
test = datasets.MNIST('../data',train=False, download=True, transform=transform)



trainLoader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)

def save_noisy_image(img, name):
    
    img = img.view(img.size(0), 1, 28, 28)
    save_image(make_grid(img, nrow=1), name)

def gaussian_noise005():
    for data in trainLoader:
        img, _ = data[0], data[1]
        gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.05, clip=True))
        save_noisy_image(gauss_img, r"gaussian005.png")        
        break

def salt_paper_noise005():
    for data in trainLoader:
        img, _ = data[0], data[1]
        salt_img = torch.tensor(random_noise(img, mode='s&p', amount=0.05))
        save_noisy_image(salt_img, r"salt005.png")
        break

def gaussian_noise02():
    for data in trainLoader:
        img, _ = data[0], data[1]
        gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.2, clip=True))
        save_noisy_image(gauss_img, r"gaussian02.png")        
        break

def salt_paper_noise02():
    for data in trainLoader:
        img, _ = data[0], data[1]
        salt_img = torch.tensor(random_noise(img, mode='s&p', amount=0.2))
        save_noisy_image(salt_img, r"salt02.png")
        break

def normal():
    for data in trainLoader:
        img, _ = data[0], data[1]
        save_noisy_image(img, r"normal.png")
        break

gaussian_noise005()
salt_paper_noise005()
gaussian_noise02()
salt_paper_noise02()
normal()

import cv2
plt.figure(figsize=(5,4))
plt.subplot(1,5,1)
plt.imshow(cv2.imread(r"normal.png"))
# plt.axis('off')
plt.yticks([], [])
plt.xticks([], [])
plt.xlabel('Nominal')

plt.subplot(1,5,2)
plt.imshow(cv2.imread(r"gaussian005.png"))
# plt.axis('off')
plt.yticks([], [])
plt.xticks([], [])
plt.xlabel('$\mathcal{N}(0,0.05)$')

plt.subplot(1,5,3)
plt.imshow(cv2.imread(r"salt005.png"))
# plt.axis('off')
plt.yticks([], [])
plt.xticks([], [])
plt.xlabel('S&P(0.05)')

plt.subplot(1,5,4)
plt.imshow(cv2.imread(r"gaussian02.png"))
# plt.axis('off')
plt.yticks([], [])
plt.xticks([], [])
plt.xlabel('$\mathcal{N}(0,0.2)$')

plt.subplot(1,5,5)
plt.imshow(cv2.imread(r"salt02.png"))
# plt.axis('off')
plt.yticks([], [])
plt.xticks([], [])
plt.xlabel('S&P(0.2)')

plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig('MNIST.pdf')  
plt.savefig('MNIST.eps',bbox_inches='tight',pad_inches = 0) 
plt.show()



# ax = [plt.subplot(2,2,i+1) for i in range(4)]

# for a in ax:
#     a.set_xticklabels([])
#     plt.imshow(cv2.imread(r"\noisy\gaussian.png"))
#     plt.axis('off')
#     plt.title('Gaussian Noise', fontsize=10)
#     a.set_yticklabels([])

# plt.subplots_adjust(wspace=0, hspace=0)

# plt.show()