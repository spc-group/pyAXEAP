__version__ = "0.0.1"
__author__ = 'Vikram Kashyap'
__credits__ = 'Argonne National Laboratory'
__doc_url__ = r'https://aps.anl.gov'

import sys
assert sys.version_info >= (3, 7)

import logging
logger = logging.getLogger('axeap')
logger.setLevel(logging.DEBUG)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

import importlib as _importlib

submodules = [
        'core',
        'utils',
        'experiment',
        'inspect',
    ]

# Load core along with top-level name
from .core import *

# Load submodules on-demand
def __getattr__(subname):
    if subname in submodules:
        return _importlib.import_module(f'{name}.{subname}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module '{name}' has no attribute '{subname}'"
            )
