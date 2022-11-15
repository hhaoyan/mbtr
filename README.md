## Gradient domain many-body tensor representation (MBTR)

This code extends the many-body tensor representation (MBTR) to include derivatives so that one could perform gradient-domain machine-learning.

### Computing MBTR and its derivative

Computing MBTR and its derivative requires iterating all k-body terms of an atomistic system and applying the corresponding geometry/weighting functions.
The derivatives of the geometry/weighting functions are easily obtained by the chain rule. We vectorize the calculation and automate the differentiation using
software packages such as Jax and PyTorch.

The following example demonstrates an example of computing 2-body MBTR and the derivative for aspirin.

```python
import numpy as np

from mbtr_grad.mbtr_python_torch import (
    mbtr_python,
    WeightFunc2BodyInvDist,
    GeomFunc2BodyInvDist,
    DistFuncGaussian,
)

aspirin = {
    'z': np.asarray([6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]),
    'r': np.asarray([
        [+1.967773, -1.138310, -0.543556],
        [+0.645441, +0.992465, -1.702570],
        [+2.234543, -0.777031, -1.830163],
        [+1.581020, +0.289498, -2.446978],
        [-3.016591, +1.741695, +0.612453],
        [+0.878228, -0.578833, +0.189787],
        [+0.263598, +0.519752, -0.384926],
        [+1.503426, -1.267264, +2.463581],
        [-2.236631, +0.279797, -1.121948],
        [-0.715723, -1.175512, +1.929332],
        [+0.633549, -0.964428, +1.667591],
        [-2.016979, +0.998229, -0.174489],
        [-0.753226, +1.137627, +0.363041],
        [-0.693787, -1.606394, +2.839317],
        [+2.522198, -1.999940, -0.132890],
        [+0.118692, +1.850180, -2.156963],
        [+3.099397, -1.279973, -2.334072],
        [+1.852263, +0.669541, -3.409804],
        [-4.006237, +1.796931, +0.063728],
        [-3.090694, +1.158956, +1.570798],
        [-2.624328, +2.785921, +0.842982] 
    ])
}

aspirin_range = np.linspace(0, 1.1, 500)
aspirin_rep, aspirin_rep_div = mbtr_python(
    z=aspirin['z'], r=aspirin['r'][None],
    order=2, grid=aspirin_range,
    weightf=WeightFunc2BodyInvDist(), 
    geomf=GeomFunc2BodyInvDist(),
    distf=DistFuncGaussian(2**-4.5)
)

print(aspirin_rep.shape, aspirin_rep_div.shape)
# Should display:
# (1, 3, 3, 500) (1, 3, 3, 500, 21, 3)
```

### sGDML

Training kernel models for forces requires a twice differentiable kernel function, which gives the potential function when integrated. 
This is the basic idea behind symmetric gradient domain machine learning (sGDML). 
You may read more information the papers by Chmiela et al. at this [link](http://quantum-machine.org/gdml/).

We here use MBTR for training sGDML models using the MD17 dataset (http://quantum-machine.org/gdml/). 

#### Prepare datasets

Download MD17 dataset:

```bash
wget -nc http://quantum-machine.org/gdml/data/npz/benzene2017_dft.npz -O notebooks/datasets/md17/benzene2017_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/uracil_dft.npz -O notebooks/datasets/md17/uracil_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/naphthalene_dft.npz -O notebooks/datasets/md17/naphthalene_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/aspirin_dft.npz -O notebooks/datasets/md17/aspirin_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/salicylic_dft.npz -O notebooks/datasets/md17/salicylic_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/malonaldehyde_dft.npz -O notebooks/datasets/md17/malonaldehyde_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/ethanol_dft.npz -O notebooks/datasets/md17/ethanol_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/toluene_dft.npz -O notebooks/datasets/md17/toluene_dft.npz
```

#### Running CM experiments

To get started, we train sGDML models using the Coulomb matrix (CM) representation:

```bash
./script_sgdml_cm_lr.py --output ./tmp/sgdml/cm
```

#### Running MBTR experiments

The following script trains sGDML models using MBTR.

```bash
./script_sgdml_mbtr2_lr.py --use_gpu --output ./tmp/sgdml/cm
```

#### Precomputed results

The trained models and their performance of each of the eight MD17 datasets are also pre-computed and uploaded to [figshare](https://figshare.com/ndownloader/files/34766953). 
```bash
mkdir -p notebooks/results
wget -O notebooks/results/sgdml.zip https://figshare.com/ndownloader/files/34766962 -o
unzip -l notebooks/results/
# Extract the files:
# unzip notebooks/results/sgdml.zip -d notebooks/results/
```

### MBTR energy models

The following sections are pertain to the ML models that do not use derivatives in the original MBTR paper. Training and 
validation of these models can be found in [this Jupyter notebook file](notebooks/mbtr.ipynb). Dataset and result files can also 
be found at [this link](https://figshare.com/articles/dataset/Unified_Representation_of_Molecules_and_Crystals_for_Machine_Learning_-_Data_and_Models/19567324).

### Citation

If you use this code in your project, please consider citing:

```latex
@article{huo2017unified,
  title={Unified representation of molecules and crystals for machine learning},
  author={Huo, Haoyan and Rupp, Matthias},
  journal={arXiv preprint arXiv:1704.06439},
  year={2017}
}
```