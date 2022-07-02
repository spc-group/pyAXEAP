import numpy as np

from .item import PathType, Saveable, Loadable, DataItemSet

class Spectra(Saveable, Loadable):
    """
    Class representing a spectra, indicating intensity over a range of energies.
    """

    TYPE_NAME = 'Spectra'
    LOAD_PATH_TYPE = PathType.FILE
    LOAD_FILE_EXTENTIONS = ['npy']
    SAVE_PATH_TYPE = PathType.FILE
    SAVE_PATH_TYPE = ['npy']

    def __init__(self, energies, intensities):
        """
        Parameters
        ----------
        energies : :obj:`numpy.ndarray`
            List of energies.
        intensities : :obj:`numpy.ndarray`
            List of intensities corresponding to each energy.
        """
        self.energies = energies
        self.intensities = intensities
        self.name = repr(self)

    # def getIntensity(self, energy):
    #    # Interpolated intensity
    #    pass

    # @property
    # def name(self):
    #     return repr(self)

    def saveToFile(self, fpath):
        """
        Save spectra to file.

        Parameters
        ----------
        fpath : path
            Path of file to which to save spectra.
        """
        np.save(fpath, np.stack((self.energies, self.intensities)).T)

    def loadFromFile(fpath:PathType):
        """
        Load spectra from file.

        Parameters
        ----------
        fpath : path
            Path of file from which to load spectra.

        Returns
        -------
        :obj:`Spectra`
            Spectra with data loaded from file.
        """
        vals = np.load(fpath)
        return Spectra(vals[:,0], vals[:,1])


class SpectraSet(DataItemSet):
    ''':obj:`.core.item.DataItemSet` for :obj:`Spectra`.
    '''
    def __init__(self, spectra=None, selection_default=True):
        ItemSet.__init__(self, spectra, selection_default=selection_default)


def calcSpectra(emap, image, evres=0.5, normalize=False):
    """
    Generates spectra from image and energy map.

    Parameters
    ----------
    emap : :obj:`numpy.ndarray`
        2D array of energy values.
    image : :obj:`numpy.ndarray`
        2D array of intensity values, same size as emap.
    evres : float
        Resolution of spectra in number of eVs.
    normalize : bool
        Normalize the spectra so peak has intensity 1.

    Returns
    -------
    :obj:`numpy.ndarray`
        2D array structured as list of energy-intensity pairs.
    """
    minenergy = np.min(emap, initial=1000000, where=emap>0)
    maxenergy = np.max(emap, initial=0, where=emap>0)
    energies = np.arange(minenergy, maxenergy+evres, evres)
    emapenergies = []
    emapenergyweights = []
    intensities = np.zeros(energies.shape)
    for x in range(emap.shape[0]):
        for y in range(emap.shape[1]):
            energy = emap[x,y]
            if energy<=0: continue  # Skip invalid regions of energy map
            if isinstance(image, np.ma.MaskedArray):
                if image.mask[x,y] == True: continue    # Skip masked region of image
            emapenergies.append(energy)
            emapenergyweights.append(image[x,y])
            try:
                intensities[round((energy-minenergy)/evres)] += image[x,y]
            except Exception:
                print("Error computing spectra. Issue with energymap?")
                print("Energy assigned to pixel:", energy)
                print("Minimum energy of emap:", minenergy)
                print("Emap energy resolution", evres)
                print(Exception)
                return
    intensities /= np.max(intensities) # Scale to range 0-1
    hist_intensities,bin_edges = np.histogram(emapenergies, bins=len(energies),
        range=(minenergy, maxenergy), weights=emapenergyweights)
    if normalize: hist_intensities/=np.max(hist_intensities)
    spectra = np.stack((energies, hist_intensities)).T
    return spectra
