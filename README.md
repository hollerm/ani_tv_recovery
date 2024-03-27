# Source code to reproduce the results of "Exact reconstruction and reconstruction from noisy data with anisotropic total variation"
This repository provides the source code for the paper "Exact reconstruction and reconstruction from noisy data with anisotropic total variation" as cited below.

## Requirements and Installation


The code was written and tested with Python 3.10.13 under Linux. We recomend to use pyenv (tested with version 2.3.35) to choose the correct python version, in which case, after typing

```bash
pyenv install 3.10.13
```

the correct python version will be automatically set by the .python_version file in this repository.


All required python packages can be found in the requirement.txt file. We recoment to install then with venv via


```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Note that the code in particular depends on [pyopencl](https://pypi.org/project/pyopencl/), for which platform dependent, suitable drivers must be installed. Please see [PyOpenCL's documentation](https://documen.tician.de/pyopencl/) for details. For performance reasons, we recommend to use GPU support to run the code.

## Examples

* To run a quick demo example, call 
```bash
python demo_tv_recon.py
```
in a terminal. This should compute results for a simple experiment and store them in the "results" folder and resulting images in the "images" folder.

* To re-compute experiments of the paper, call 
```bash
reproduce_experiments.py
```
This will compute all results of the paper and the corresponding images and plots (this might take a while). To select only specific results, see the source code of "reproduce_experiments.py" and select only specific experiments.

## Important modules

* "tv_recon_ocl.py" implements the TV-regularized recovery algorithm
* "ani_tv_supp.py" implements various context-specific helper functions, in particular for computing and evaluation experiments
* "matpy.py" implements some general purpose functions.

## Authors of the code


* **Martin Holler** martin.holler@uni-graz.at 
* **Benedikt Wirth** benedikt.wirth@uni-muenster.de

MH is currently affiliated with the Department of Mathematics and Scientific Computing, University of Graz, Graz, Austria. BW is currently affiliated with the Institute for Applied Mathematics: Analysis and Numerics, University of Münster, Münster, Germany.

## Publications
If you use this code, please cite the following associated publication.

* M. Holler and B. Wirth. Exact reconstruction and reconstruction from noisy data with anisotropic total variation. To appear in SIAM Journal on Mathematical Analysis, 2023. [arXiv](https://arxiv.org/abs/2207.04757)

## License
The code in this project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details.
