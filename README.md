# pyAXEAP: Argonne X-Ray Emission Analysis Package in Python

AXEAP is a software package being developed at Argonne National Lab for use in
extracting x-ray emission spectra from sensors in beamline application and
analyzing the spectra. `axeap` is a python module being developed as part of
AXEAP that aims to provide a framework for the extraction of XES from images
from pixel array detectors. The `axeap` includes features like automatically
detecting regions of interest (ROIs), automatically monitoring directories
for new images, calibrating pixel-to-energy maps from calibration images, and
calculating spectra from experimental images. The goal for `axeap` is for it to
be usable in standalone scripts, in conjunction with
[bluesky](https://blueskyproject.io/), or the basis for a graphical application.

## Installation

To install `axeap`, download this repository and run the following command
inside of the "pyAXEAP" directory:

```
pip install .
```

## Examples

An example jupyter notebook can be viewed [here](examples/pyAXEAP_example.ipynb).

## Documentation

The API can be found at [https://spc-group.github.io/pyAXEAP/](https://spc-group.github.io/pyAXEAP/).
