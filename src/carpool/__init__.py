# carpool/__init__.py
"""
CARPOOL: A simulation and control framework for [brief description, e.g., multi-agent car systems].
"""

__version__ = "0.1.0"  # Update as needed for versioning

# Import subpackages for convenience (optional; users can still import them directly)
from . import config
from . import controllers
from . import environments
from . import experiments
from . import optimization
from . import planners
from . import utils

# Expose key functions/classes if commonly used (e.g., from your scripts)
# Adjust based on what's in experiment_runner.py or similar
from .experiments.two_robots_in_mujoco import run_carpool_simulation  # Or whatever the actual function/module is

# Define __all__ to control 'from carpool import *' (optional, list public names)
__all__ = [
    "config",
    "controllers",
    "environments",
    "experiments",
    "optimization",
    "planners",
    "utils",
    "run_carpool_simulation",
]