# Contractive-Neural-ODEs
PyTorch implementation of Contractive Hamiltonian Neural ODEs


## Installation 



## Illustration 1
To show how contractivity promotes the robustness of a neual ODE, a comparison between a vanilla neural ODE  and a contractive Hamiltonian Neural ODE (CH-NODE) is provided.

<p align="center">
<img src="./Figures/Vanilla_neural_ode.png" alt="Class_vanilla" width="400"/>
<img src="./Figures/vanilla_neural_ode_phaseplot.png" alt="flow_vanilla" width="400"/>
</p>



<p align="center">
<img src="./Figures/Classification_CHNNs.png" alt="class_CHNODE" width="400"/>
<img src="./Figures/Contractive_neural_ODE_flow.png" alt="FLOW_CHNODE" width="400"/>
</p>




Contractive_neural_ODE_flow
## Illustration 2
We provide the comparison of CH-NODE with ResNets and H-DNNs. 

<p align="center">
<img src="./Figures/MNIST.png" alt="MNIST" width="400"/>
</p>


## Illustration 3
We demonstrate that our proposed CH-NODE enjoys non-exploding gradients properties by design. 

<p align="center">
  <img src="./Figures/Double_circles.png" alt="circles" width="400"/>
<img src="./Figures/Grads_CHNODE.png" alt="GRADS" width="400"/>
</p>

