# Contractive-Hamiltonian-Neural-ODEs
PyTorch implementation of Contractive Hamiltonian Neural ODEs


## Installation 



## Illustration 1
To show how contractivity promotes the robustness of a neual ODE, a comparison between a vanilla neural ODE  and a contractive Hamiltonian Neural ODE (CH-NODE) is provided.

<p align="center">
<img src="./Figures/boundary_vanilla.png" alt="Class_vanilla" width="400"/>
<img src="./Figures/Flow_contractive.png" alt="flow_vanilla" width="400"/>
</p>



<p align="center">
<img src="./Figures/boundary_contractive.png" alt="class_CHNODE" width="400"/>
<img src="./Figures/Flow_contractive.png" alt="FLOW_CHNODE" width="400"/>
</p>




Contractive_neural_ODE_flow
## Illustration 2
We provide the comparison of CH-NODE with ResNets and H-DNNs. We use the complete MNIST dataset (60,000 training samples and 10,000 test samples), a mini-batch size of 100, and 10 epochs for the training. For the optimization algorithm, we use SGD with Adam \cite{kingma2015adam} and the cross-entropy loss. The learning rate, or optimization step size, is initialized to be 0.04 with a decay rate of 0.8 at each epoch.  


<p align="center">
<img src="./Figures/MNIST.png" alt="MNIST" width="400"/>
</p>


## Illustration 3
We demonstrate that our proposed CH-NODE enjoys non-exploding gradients properties by design. 

<p align="center">
  <img src="./Figures/Double_circles.png" alt="circles" width="400"/>
<img src="./Figures/Grads_CHNODE.png" alt="GRADS" width="400"/>
</p>

