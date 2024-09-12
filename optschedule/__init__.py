"""
OptSchedule

Flexible parameter scheduler that can be implemented with proprietary and open source optimizers and algorithms.
"""

from optschedule.schedule import *

__all__ = [s for s in dir() if not s.startswith("_")]

__version__ = "0.1.0"
__author__ = "draktr"
