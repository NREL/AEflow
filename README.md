# AEflow
## Deep learning for in-situ data compression of large turbulent flow simulations

This repository contains the code associated with the paper

Glaws, A., King, R., and Sprague, M., *Deep learning for in-situ data compression of large turbulent flow simulations*.  [https://doi.org/10.1103/PhysRevFluids.5.114602](https://doi.org/10.1103/PhysRevFluids.5.114602).

___
### Requirements
- Python v3.7
- TensorFlow v1.12+
- numpy v1.16+
- matplotlib v3.0+

### Running the Model
The `main.py` file provides the functionality to run the model for data compression and reconstruction. The `utils.py` file contains functionality for generating and parsing the TFRecord data files as well as plotting the results. Additionally, a framework for training the network on new data is provided in the `train.py` file; however, training data would need to be provided by the user.

### Data and Model Weights
Sample test data is provided in the `data/` directory. This data includes a snapshot of homogeneous isotropic turbulence generated using the [spectralDNS package](https://github.com/spectralDNS/spectralDNS "spectralDNS"). 

Pretrained model weights can be found in `models/`. The model was trained on snapshots of homogeneous isotropic turbulence as discussed in the paper. It has been shown to generalize well to other canonical flow problems. However, the model may be retrained on user-provided data using the `train.py` script.

#### Acknowledgments
This work was authored by the National Renewable Energy Laboratory (NREL), operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the Exascale Computing Project (No. 17-SC-20-SC), a collaborative effort of two DOE organizations (Office of Science and the National Nuclear Security Administration) responsible for the planning and preparation of a capable exascale ecosystem, including software, applications, hardware, advanced system engineering, and early test-bed platforms, in support of the nation’s exascale computing imperative. The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
