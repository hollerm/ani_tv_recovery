# Source code to reproduce the results of "Exact reconstruction and reconstruction from noisy data with anisotropic total variation"
This repository provides the source code for the paper "Exact reconstruction and reconstruction from noisy data with anisotropic total variation" as cited below.

## Requirements and Installation


The code was written and tested with Python 3.10.13 under Linux. We recomend to use pyenv (tested with version 2.3.35) to chose the correct python version, in which case, after typing

```bash
pyenv install 3.10.13
```

the correct python version will be automatically set by the .python_version file in this repository.


All required python packages can be found in the requirement.txt file. We recoment to install then with venv via


```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Note that the code in particular depends on [pyopencl](https://pypi.org/project/pyopencl/), for which platform dependent, suitable drivers must be installed. Please see [PyOpenCL's documentation](https://documen.tician.de/pyopencl/) for details. For performance reasons, we recommend to use GPU support to run the code.

## Examples

* To run a quick demo example, call "python demo_tv_recon.py" in a terminal. This should compute results for a simple experiment and store the result in the "results" folder and resultin images in the "images" folder.

* To re-compute experiments of the paper, call "reproduce_experiments.py". This will compute all results of the paper that were computed with the PyTorch code (it might take a while). To select only specific results, see the source code of "reproduce_experiments.py" and select only specific experiments.

## Authors of the code

* **Christian Aarset** c.aarset@math.uni-goettingen.de
* **Martin Holler** martin.holler@uni-graz.at 
* **Tram Thi Ngoc Nguyen** nguyen@mps.mpg.de

CA is currently affiliated with the Institute for Numerical and Applied Mathematics, University of Göttingen, Göttingen, Germany. MH is currently affiliated with the Department of Mathematics and Scientific Computing, University of Graz, Graz, Austria. TTNN is currently affiliated with the Max Planck Institute for Solar System Research, Göttingen, Germany.

## Publications
If you use this code, please cite the following associated publication.

* Aarset, C., Holler, M., Nguyen, T.T.N. Learning-Informed Parameter Identification in Nonlinear Time-Dependent PDEs. Applied Mathematics & Optimization, 2023. https://doi.org/10.1007/s00245-023-10044-y

## License
The code in this project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details.
