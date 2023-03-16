# OptSchedule

Flexible parameter scheduler that can be implemented with proprietary and open source optimizers and algorithms.

* Free software: MIT license
* Documentation: <https://optschedule.readthedocs.io/en/latest/>

## Installation

`optschedule` can be installed through Python's package installer pip. To install, run

```shell
pip install optschedule
```

in your terminal. Alternatively, install the package directly from GitHub

```shell
git clone -b development https://github.com/draktr/optschedule.git
cd monte
python setup.py install
```

## Features

* Exponential decay (gradual and staircase)
* Cosine decay
* Inverse time decay (gradual and staircase)
* Polynomial decay
* Piecewise constant decay
* Constant schedule

## Advantages

* **FLEXIBLE** - the package is designed to be simple and compatible with existing implementations and custom algorithms

* **COMPREHENSIVE** - the package contains the largest collection of schedules of any Python package. For more, feel free to raise a feature request in Issues.

* **NUMBA FRIENDLY** - schedule produced by the package is compatible with Numba and will not cause any issues if the rest of the algorithm is Numba compatible. This can drastically speed up the algorithm.
