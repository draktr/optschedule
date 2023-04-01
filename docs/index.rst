OptSchedule
===========

Flexible parameter scheduler that can be implemented with proprietary
and open source optimizers and algorithms.

-  Free software: MIT license
-  Documentation: https://optschedule.readthedocs.io/en/latest/

Installation
------------

``optschedule`` can be installed through Python’s package installer pip.
To install, run

.. code:: shell

   pip install optschedule

in your terminal. Alternatively, install the package directly from
GitHub

.. code:: shell

   git clone -b development https://github.com/draktr/optschedule.git
   cd monte
   python setup.py install

Features
--------

-  Exponential decay (gradual and staircase)
-  Cosine decay
-  Inverse time decay (gradual and staircase)
-  Polynomial decay
-  Piecewise constant decay
-  Constant schedule
-  Geometric decay
-  Arithmetic decay
-  Time decay
-  Step decay

Advantages
----------

-  **FLEXIBLE** - the package is designed to be simple and compatible
   with existing implementations and custom algorithms

-  **COMPREHENSIVE** - the package contains the largest collection of
   schedules of any Python package. For more, feel free to raise a
   feature request in Issues.

-  **NUMBA FRIENDLY** - schedule produced by the package is compatible
   with Numba and will not cause any issues if the rest of the algorithm
   is Numba compatible. This can drastically speed up the algorithm.

Usage
-----

Package contains ``Schedule`` class with methods that return an array of
elements that is useful as a pre-defined parameter schedule
(e.g. learning rate). The package can also be used for manually
assigning varying weights to abstract particles. Overall, due to the
general nature of the package a user might finds its own particular
application.

Example: Variable Learning Rate in Gradient Descent Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from optschedule import Schedule

   # Function to be minimized (objective function) $ f(x) = (x+2)^2 $
   def foo(params):
       return (params[0] + 2) ** 2

   # Creating learning rate schedule
   schedule = Schedule(1000)
   learning_rate = schedule.exponential_decay(initial_value=0.1, decay_rate=0.5)

   # Array with objective value
   objective = np.zeros(1000)
   # Initial parameter value
   param = [10]
   # Difference
   d = 0.01

   # Gradient Descent Algorithm
   for epoch, l in enumerate(learning_rate):
       objective[epoch] = foo(param)
       difference_objective = foo([param[0]+d])
       param[0] = param[0] - l*(difference_objective - objective[epoch])/d

   print(f"Solution: {param[0]}")

Maintaining and Contributing
----------------------------

Feel free to reach out through Issues forum to add or request features.
Any issues, bugs and improvement recommendations are very welcome.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   optschedule


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
