"""Energy Map

Energy maps are 2D arrays the same size as a XES image where each pixel in value
in the energymap corresponds to an energy. Thus when a pixel is excited in an
image, the corresponding energy is added to the spectra with an intensity
corresponding to the energy of the pixel.
"""

import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
import random
from pathlib import Path

from .scan import Scan, ScanSet
from .item import DataItem, Saveable, Loadable, PathType
from .spectra import Spectra, calcSpectra
from .roi import HROI
from ..utils import *
from .conventions import X, Y

class EnergyMap(DataItem, Saveable, Loadable):
    """
    Class representing a mapping of pixels onto energies.
    """

    TYPE_NAME = "EnergyMap"
    LOAD_FILE_EXTENTIONS = ['npy']
    LOAD_PATH_TYPE = PathType.FILE
    SAVE_FILE_EXTENTIONS = ['npy']
    SAVE_PATH_TYPE = PathType.FILE

    def __init__(self, values, name=None, eres=None):
        """
        Parameters
        ----------
        values : :obj:`numpy.ndarray`
            2D array containing energy corresponding to each pixel
        eres : float
            Energy resolution corresponding to mapping, units of eV/pixel
        """
        self.values = values
        self._eres = eres
        self.name = name if name is not None else "energymap"

    @property
    def dims(self):
        """
        Get dimensions of energy map.

        Returns
        -------
        tuple
            Shape of energy map as (x, y).
        """
        return self.values.shape

    @property
    def eres(self):
        """
        Get the energy resolution (eV/pixel) for the energy map.

        Returns
        -------
        float
            Stored value if set in constructor, else estimates resolution from
            map values.
        """
        if not self._eres:
            self._eres = self._inferRes()*1.1
        return self._eres

    def _inferRes(self):
        """
        Estimate energy resolution (eV/pixel) for the energy map.

        Works by sampling multiple columns of pixels and averaging change in
        energy over change in vertical pixel position for each column. Invalid
        regions of energy map are excluded from calculation.

        Returns
        -------
        float
            Calculated estimate of resolution of energy map.
        """
        testedsegments = 0
        numtestsegments = 5
        eres = 0.0
        while testedsegments < numtestsegments:
            col = self.values[random.randint(0, self.values.shape[X]-1)]
            validyvals = np.where(col>0)[0] # np.where returns list in tuple
            if len(validyvals) == 0:
                continue
            else:
                segmentstart = validyvals[0]
                i = segmentstart+1
                while i!=self.dims[Y] and col[i]>0:
                    i+=1
                segmentend = i-1
                segmentlen = segmentend-segmentstart
                energychange = col[segmentend]-col[segmentstart]
                eres += (energychange/segmentlen)
                testedsegments+=1
        eres /= numtestsegments
        return eres

    def calcSpectra(self, scan):
        """Calculate spectra from scan.

        Parameters
        ----------
        scan : :obj:`.scan.Scan`
            Scan to calculate spectra from.

        Returns
        -------
        Spectra
            Spectra object.
        """

        spectravals = calcSpectra(self.values, scan.getImg(), self.eres)
        return Spectra(spectravals[:,0],spectravals[:,1])

    def saveToPath(self, fpath):
        """
        Save energy map to file.

        Energy resolution is not saved. This will be added in the future.

        Parameters
        ----------
        fpath : Path
            Path of file to which to save energy map
        """
        np.save(fpath, self.values)

    def loadFromPath(fpath):
        """
        Load energy map from file.

        Parameters
        ----------
        fpath : Path
            Path of file from which to load energy map.

        Returns
        -------
        EnergyMap
            Energy map loaded from file.
        """
        fpath = Path(fpath)
        emapvals = np.load(fpath)
        return EnergyMap(emapvals, fpath.stem)

    # def __add__(self, other:EnergyMap) -> EnergyMap:
    #     return EnergyMap(self.values+other.values)


def calcEMap(scanset, hrois):
    """Function that calculates an pixel-to-energy mapping (an energy map) from a
    set of scans and the horizontal pixel ranges (the groups) that have been
    determined to correspond to different crystals in the scans.

    Parameters
    ----------
    scanset : ScanSet
        Set of calibration scans.
    hrois : list
        List of :obj:`.core.roi.HROI` corresponding to each crystal.

    Returns
    -------
    np.ndarray
        2D array same size as scans of energy values
        corresponding to each pixel.
    """
    emap = np.full(scanset.dims, float(-1)) #Energy map same size as image, default -1 (invalid)
    # Find regression for each set of image points within each region
    # Each region should correspond to one crystal, or perhaps a pair of crystals
    for hroi in hrois:
        # Generate models for the line in the group in each image
        # The line corresponds to a single energy
        lox, hix = hroi
        linemodels = {}
        energies = [s.meta['IncidentEnergy'] for s in scanset]
        for scan in scanset:
            energy = scan.meta['IncidentEnergy']
            image = scan.getImg(cuts=(6,10))
            imgheight = scanset.dims[Y]
            grouparea = [[lox,round(imgheight/10)],[hix,round(imgheight*9/10)]]
            points = np.array(getCoordsFromImage(image, area=grouparea, subsample=None))
            linemodels[energy]=np.poly1d(np.polyfit(points[:,0], points[:,1], 4, w=points[:,2]))
        # Generate the energymap values for the pixels within the group
        for xval in range(int(np.ceil(lox)), int(np.ceil(hix))):
            # Fit function to column with energy as a function of pixel height y
            # Based on Bragg's Angle formula, E=a/y for some a value
            known_yvals = [linemodels[energy](xval) for energy in linemodels]
            known_evals = energies
            #def efunc(y, a): return a/y
            #fitparams, cov = curve_fit(efunc, known_yvals, known_evals)[0]
            efunc = interpolate.interp1d(known_yvals, known_evals, kind='cubic')
            for yval in range(int(np.ceil(min(known_yvals))), int(max(known_yvals))):
                #emap[xval][yval] = efunc(yval, a)
                emap[xval][yval] = efunc(yval)
    return EnergyMap(emap)
