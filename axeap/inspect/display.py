from ..core import (Scan, Spectra, ImageSpec, ROI, HROI, VROI, RectangleROI)
from ..core.conventions import X, Y
from ..utils import getCoordsFromImage

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatches


def displayScan(scan, rois=None, ax=None, plotargs=None):
    """Display a `.core.Scan` using pyplot.

    Parameters
    ----------
    scan : :obj:`.core.scan.Scan`)
        Scan to be displayed.
    ax : :obj:`matplotlib.pyplot.Axes`
        PyPlot Axes to display on.
    plotargs : :obj:`dict`
        Key word arguments to pass to :obj:`matplotlib.pyplot.Axes.imshow`.
    """
    if ax==None:
        fig, ax = plt.subplots()
    if plotargs==None:
        plotargs = {}
        origin, cmap = 'lower', 'binary'
    else:
        plotargs = plotargs.copy()
        origin = d.pop('origin') if 'origin' in plotargs else 'lower'
        cmap = d.pop('cmap') if 'cmap' in plotargs else 'gray'
    img = scan.getImg()
    #points = np.array(getCoordsFromImage(img))
    ax.set_xlim(0, scan.dims[X])
    ax.set_ylim(0, scan.dims[Y])
    ax.imshow(img.swapaxes(0,1), origin=origin, cmap=cmap, **plotargs)
    #ax.scatter(points[:,0], points[:,1], s=3)
    if rois is not None:
        #print('got ROIs', len(rois))
        for roi in rois:
            if isinstance(roi, HROI):
                ax.add_patch(pltpatches.Rectangle( \
                    (roi.lo,0), roi.hi-roi.lo, scan.dims[Y], color='r', alpha=0.2))
            elif isinstance(roi, VROI):
                ax.add_patch(pltpatches.Rectangle( \
                    (0,roi.lo), scan.dims[X], roi.hi-roi.lo, color='r', alpha=0.2))
            elif isinstance(roi, RectangleROI):
                #print(roi)
                ax.add_patch(pltpatches.Rectangle( \
                    (roi.lox, roi.loy), roi.hix-roi.lox, roi.hiy-roi.loy, color='r', alpha=0.2))
    #plt.show()

def displaySpectra(spectra, ax=None, plotargs=None):
    """Display a `.core.Spectra` using pyplot.

    Parameters
    ----------
    spectra : :obj:`.core.spectra.Spectra`
        Spectra to display.
    ax : :obj:`matplotlib.pyplot.Axes`
        Pyplot axes to display on.
    plotargs : :obj:`dict`
        Key word arguments to pass to `matplotlib.pyplot.Axes.plot`.
    """
    if ax==None:
        _, ax = plt.subplots()
    if plotargs==None:
        plotargs = {}
    line = ax.plot(spectra.energies, spectra.intensities, **plotargs)[0] # Should be only one line
    return line
