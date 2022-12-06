Code for manuscript
**Role of interneuron subtypes in controlling trial-by-trial output variability in the neocortex**
*Lihao Guo, Arvind Kumar*


1. Requirements

To run all simulations, the nest simulator needs to be installed with version 2.*

To just generate figures from processed data, only figurefirst is needed for figure formatting [https://flyranch.github.io/figurefirst/]

2. Simulations and Figures

runsim.sh:
This runs the process for simulating steady-state response of the EPSV network to generate neuron transfer-function and to estimate variability transfer from the neuron tranfer-function.
To run simulations, set the flag sim=1
To generate figure of neuron tranfer-function (Fig1.D-F) and estimated variability transfer(Fig.3), set the flag fig=1

runsamp.sh:
The runs the process for simulating with sampled inputs from a given covariance matrix
To run simulations, set the flag sim=1
To generate figure of sampled simulations for variability control with variance ratio (Fig.4) and covariance (Fig.5), set the flag fig=1

By default:
    all simulations are stored in ./data folder
    all processed data are stored in ./experiments/J25.0-m0.5-r0.0/frs
    all figures are stored in ./experiments/J25.0-m0.5-r0.0/fig
