# AEflow3D

This repository is the 3-channel implementation of the paper:

Glaws, A., King, R., and Sprague, M., *Deep learning for in-situ data compression of large turbulent flow simulations*.  [https://doi.org/10.1103/PhysRevFluids.5.114602](https://doi.org/10.1103/PhysRevFluids.5.114602).

___
### Requirements
- Python v3.7+
- TensorFlow v2.8+
- matplotlib v3.0+

### Running the Model
The `main3D.py` file provides the functionality to run the AEflow3D model for data compression and reconstruction. The `utils.py` file contains functionality for generating and parsing the TFRecord data files as well as plotting the results. Additionally, a framework for training the network on new data is provided in the `train3D.py` file; however, training data would need to be provided by the user.

### Data and Model Weights
Sample test data is provided in the `data/` directory. This data includes a snapshot of homogeneous isotropic turbulence generated using the [spectralDNS package](https://github.com/spectralDNS/spectralDNS "spectralDNS"). 

Pretrained models of AEflow1D (implementation of the original version in TFv2) and AEflow3D can be found under `models/`. For the original weights of the AEflow model please refer to the `model/` folder. The AEflow3D model was trained on snapshots of 3-channel homogeneous isotropic turbulence flows and evaluated with several metrics in `utils.py` and `train3D.py`.

#### Acknowledgments
This work was authored by the National Renewable Energy Laboratory (NREL), operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by the Laboratory Directed Research and Development (LDRD) Program at NREL. The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
