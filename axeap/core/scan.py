'''Scan.
The Scan class represents an image from a PAD detector. This file contains all
code relating to Scans.
'''

import warnings
import logging
logger = logging.getLogger('axeap')

from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from pathlib import Path
import pandas as pd
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .item import DataItem, DataItemSet, Saveable, Loadable
from .signal import Signal
from .conventions import X, Y, PathType
from ..utils import get_trailing_number


class Scan(DataItem, Saveable, Loadable):
    """
    Class representing a scan image taken with a XES detector.
    """

    TYPE_NAME = "Scan"
    LOAD_FILE_EXTENTIONS = ['tiff','tif','scan']
    SAVE_PATH_TYPE = PathType.FILE
    SAVE_FILE_EXTENTIONS = ['tif', 'png', 'scan']
    SAVE_PATH_TYPE = PathType.FILE
    name = "Scan"

    def __init__(self, img, imgspec=None, name=None, meta=None, cache=True):
        """
        Parameters
        ----------
        img : :obj:`numpy.ndarray`
            2D array of intensity values
        name : :obj:`str`
            Name for the scan.
        cache : :obj:`bool`, :obj:`dict`
            Pass :obj:`False` to turn off cacheing of images. An
            already-prepared cache of images in the form a dictionary of image
            specs to images can also be passed. Do not use unless you know what
            you're doing.
        """
        DataItem.__init__(self)
        self.img = img
        self.name = name
        self.default_imgspec = imgspec if imgspec is not None else IS.NOCHANGE
        if isinstance(cache, dict):
            self._imgcache = cache
        elif cache is False:
            self._imgcache = None
        elif cache is True:
            self._imgcache = {}
        if meta is not None:
            self.meta |= meta


    def __repr__(self):
        """
        Returns
        -------
        :obj:`str`
            Representation of scan consists of name and uuid.
        """
        return f"Scan(name:{self.name},uuid:{self.uuid})"

    @property
    def dims(self):
        """
        Dimensions of scan image.

        Returns
        -------
        :obj:`tuple`
            Dimensions of scan image in form (x,y).
        """
        return self.getImg().shape

    def getImgFromSpecs(self, imgspec):
        """Get image after applying imagespec.

        Parameters
        ----------
        imgspec : :obj:`ImageSpec`)
            Specifications for image.

        Returns
        -------
        :obj:`numpy.ndarray`
            2D array representing image.
        """
        if self._imgcache is not None and imgspec in self._imgcache:
            return self._imgcache[imgspec]
        if imgspec == IS.NOCHANGE:
            return self.img
        cuts = imgspec['cuts']
        scale = imgspec['scale']
        blur = imgspec['blur']
        roi = imgspec['roi']
        if cuts != -1 or blur != -1 or scale != -1:
            img = self.img.copy()
        else:
            img = self.img
        if cuts != -1:
            img[np.logical_or(img<cuts[0], img>cuts[1])] = 0
        if blur!=-1: img = gaussian_filter(img, sigma=blur)
        if scale != -1:
            img *= scale
        if roi!=-1: img = roi._selectArea(img)
        img.flags.writeable = False
        if self._imgcache is not None: self._imgcache[imgspec] = img
        return img

    def _getISFromArgs(self, *args, **kwargs):
        imgspec = None
        if len(args) == 0:
            if len(kwargs) == 0:
                imgspec = self.default_imgspec # Called without args so pass default
            else:
                imgspec = IS(self.default_imgspec, **kwargs) # No ImageSpec object passed so create it
        elif len(args)==1:  # ImageSpec object passed, use it
            if isinstance(args[0],IS):
                if len(kwargs) != 0:
                    warnings.warn('If ImageSpec object is passed, keyword args are ignored.')
                imgspec = args[0]
            elif isinstance(args[0],dict): imgspec = IS(**args[0])
            else:
                raise ValueError('Only an ImageSpec object can be passed as a positional argument.')
        else:
            raise ValueError('Only one ImageSpec object can be passed as a positional argument.')
        return imgspec


    def getImg(self, *args, **kwargs):
        """
        Get 2D array of intensity values associated with scan, optionally
        pre-processed.

        Parameters
        ----------
        cuts : :obj:`tuple`, optional
            Pair of values (l,h) where any pixel values with intensity <l or
            >h are set to 0.
        scale : :obj:`float`, optional
            Number to scale all intensity values in image by.
        blur : :obj:`int`, optional
            Kernel size of gaussian blur to apply to image.
        roi : :obj:`.core.ROI`, optional
            Region of interest to which to restrict image. The rest of the image
            will be masked off.

        Returns
        -------
        :obj:`numpy.ndarray`
            2D array of intensity values representing image.
        """
        imgspec = self._getISFromArgs(*args, **kwargs)
        return self.getImgFromSpecs(imgspec)


    def mod(self, *args, **kwargs):
        """
        Return scan with a different default imgspec.

        Parameters
        ----------
        cuts : :obj:`tuple`, optional
            Pair of values (l,h) where any pixel values with intensity <l or
            >h are set to 0.
        scale : :obj:`float`, optional
            Number to scale all intensity values in image by.
        blur : :obj:`int`, optional
            Kernel size of gaussian blur to apply to image.
        roi : :obj:`.core.ROI`, optional
            Region of interest to which to restrict image. The rest of the image
            will be masked off.

        Returns
        -------
        :obj:`Scan`
            Scan object with given default imgspec.
        """
        imgspec = self._getISFromArgs(*args, **kwargs)
        return Scan(self.img, name=self.name, imgspec=imgspec,
            meta=self.meta, cache=self._imgcache)


    def loadImageFromFile(fpath):
        """Load an image from a file.

        Parameters
        ----------
        fpath : :obj:`pathlib.Path`, :obj:`str`
            Path of image file (either TIF or NPY).

        Returns:
            `numpy.ndarray`: 2D image data array.
        """
        fpath = Path(fpath) # Convert to Path in case passed as str
        img = None
        if fpath.suffix.lower() == '.tif' or fpath.suffix.lower() == '.tiff':
            with open(fpath) as f:
                img = np.array(Image.open(fpath), dtype=np.float32)
                #img = np.array(Image.open(fpath))
                img = np.swapaxes(img, 0, 1)
                return img
        elif fpath.suffix.lower() == '.npy':    # Only for loading .npy files exported by Scan
            return np.load(fpath)
        else:
            raise ValueError(f'Cannot load Scan from "{fpath.suffix}" file. Only TIF and NPY supported.')


    def loadFromPath(fpath):
        """
        Load scan data from file.

        Parameters
        ----------
        fpath : :obj:`pathlib.Path`, :obj:`str`
            Path to file containing image data.

        Returns:
        :obj:`Scan`
            Scan object containing data from file.
        """
        fpath = Path(fpath) # Convert to Path object in case path is string
        s = Scan(Scan.loadImageFromFile(fpath),name=fpath.name)
        s.meta['fpath'] = fpath
        return s


    def saveToPath(self, fpath):
        """
        Save scan to file.

        Parameters
        ----------
        fpath : :obj:`pathlib.Path`, :obj:`str`
            Path of file to which to save scan.
        """
        fpath = Path(fpath) # Convert to Path in case passed as str
        if fpath.suffix.lower() == '.tif' or fpath.suffix.lower() == '.tiff':
            Image.fromarray(self._img).save(fpath)
        elif fpath.suffix.lower() == '.npy':
            return np.save(fpath, self._img, allow_pickle=False)
        else:
            raise ValueError(f'Cannot save Scan to "{fpath.suffix.upper()}" file. Only TIF and NPY supported.')

    def count(self, roi=None):
        """Count total intensity within ROI.

        Parameters
        ----------
        roi : :obj:`.core.roi.ROI`, optional
            Region of Interest to restrict count to. If not given, region is
            considered to be entire scan.

        Returns
        :obj:`int`
            Total intensity summed over region.
        """
        if roi:
            img = self.getImg(roi=roi)
        else:
            img = self.getImg()
        return np.sum(img)


class ImageSpec(dict):
    """
    Class representing specifications applied to scan image.
    """
    NOCUTS = -1 # (0,100000)
    NOSCALE = -1
    NOBLUR = -1
    NOROI = -1

    def __init__(self,
        basespec=None,
        cuts=None,
        scale=None,
        blur=None,
        roi=None):
        """
        Parameters
        ----------
        basespec : :obj:`ImageSpec`
            :obj:`ImageSpec` to use as default values.
        cuts : :obj:`tuple`, optional
            Pair of values (l,h) where any pixel values with intensity <l or
            >h are set to 0.
        scale : :obj:`float`, optional
            Number to scale all intensity values in image by.
        blur : :obj:`int`, optional
            Kernel size of gaussian blur to apply to image.
        roi : :obj:`.core.ROI`, optional
            Region of interest to which to restrict image. The rest of the image
            will be masked off.
        """
        default = basespec if basespec is not None else ImageSpec.NOCHANGE
        self['cuts'] = cuts if cuts is not None else default['cuts']
        self['scale'] = scale if scale is not None else default['scale']
        self['blur'] = blur if blur is not None else default['blur']
        self['roi'] = roi if roi is not None else default['roi']

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def __eq__(self, other):
        if not isinstance(other, ImageSpec) and isinstance(other, dict):
            other = ImageSpec(**other)
        return hash(self)==hash(other)

    @classmethod
    @property
    def NOCHANGE(cls):
        """An ImageSpec that signifies no change to be made to an image."""
        return ImageSpec({}, cuts=ImageSpec.NOCUTS, blur=ImageSpec.NOBLUR,
                        scale=ImageSpec.NOSCALE, roi=ImageSpec.NOROI)

IS = ImageSpec
"""Alias for ImageSpec."""


class ScanSet(DataItemSet, Loadable):
    """:obj:`.core.item.DataItemSet` for :obj:`Scan` objects."""

    TYPE_NAME = "Scan Set"
    LOAD_PATH_TYPE = PathType.DIR

    DEFAULT_IMGDIMS = (100,100)

    def __init__(self, scans=None, name=None, imgdims=None, selection_default=True):
        self.name = name if name is not None else "scanset"
        self.imgdims = imgdims
        DataItemSet.__init__(self, items=scans, selection_default=selection_default)

    @property
    def dims(self):
        return self.imgdims

    def loadFromPath(dpath, calibinfopath=None):
        dpath = Path(dpath)
        fpaths = [fpath for fpath in dpath.glob('*')
            if fpath.suffix.lstrip('.').lower() in Scan.LOAD_FILE_EXTENTIONS]
        fpaths = sorted(fpaths, key=lambda x: int(get_trailing_number(x.stem,default=0)))
        scans = [Scan.loadFromPath(fpath) for fpath in fpaths]
        ss = ScanSet(scans, name=dpath.stem)
        if calibinfopath is not None:
            ss.addCalibRunInfo(CalibRunInfo(calibinfopath))
        else:
            allfiles = dpath.glob("*")
            cri = None
            for p in allfiles:
                ext = p.suffix.lstrip('.')
                if ext.isnumeric():
                    try:
                        cri = CalibRunInfo(p)
                    except:
                        ...
            if cri is not None:
                ss.addCalibRunInfo(cri)
        return ss


    def add(self, scan):
        """Add Scan to set.

        See :obj:`.core.item.DataItemSet.add`.
        """
        if self.imgdims:
            if scan.dims != self.imgdims:
                raise ValueError(f'All scans in set must have dimensions {imgdims}. Scan has dimensions {s.dims}.')
        else:
            self.imgdims = scan.dims
        DataItemSet.add(self, scan)


    def composite(self, imgkwargs=None):
        """Create scan by adding images from all selected scans in set.

        Parameters
        ----------
        imgkwargs : :obj:`ImageSpec`
            Specifications to apply to image extracted from each
            :obj:`Scan`.

        Returns:
            :obj:`Scan`
                Scan object representing combined image.
        """
        selectedscans = self.getSelectedScans()
        if len(selectedscans)==0:
            logger.debug("No scans selected.")
            dims = self.imgdims if self.imgdims is not None else self.DEFAULT_IMGDIMS
            return Scan(np.zeros(dims))
        else:
            if not imgkwargs: imgkwargs = {}
            selectedscans = iter(selectedscans)
            i = next(selectedscans).getImg(**imgkwargs).copy()
            for s in selectedscans:
                i += s.getImg(**imgkwargs)
            return Scan(i)

    def addCalibRunInfo(self, runinfo):
        """Populate metadata of scans with calibration run information (incident
            energy, intensity, etc).

        Parameters
        ----------
        runinfo : :obj:`CalibRunInfo`
            Calibration run information.
        """
        if len(self) != len(runinfo.energies):
            raise ValueError("Number of scans in calibration run information" \
                f" file ({len(runinfo.energies)})does not match number of scans in" \
                f" scan set ({len(self)}).")
        else:
            for s, e, in zip(self, runinfo.energies):
                s.meta['IncidentEnergy'] = e


class CalibRunInfo:
    """
    Class that represents the settings used during the collection of a set of
    calibration scans.

    Attributes
    ----------
    energies : :obj:`list`
        List of energies scans were taken at.
    """
    def __init__(self, fpath):
        """
        Parameters
        ----------
        fpath : :obj:`pathlib.Path`, :obj:`str`
            Path of file containing information.
        """
        self._table = CalibRunInfo._readDataTable(fpath)
        self._table.rename(
            columns={'Mono Energy (alt) *':'Energy'},
            inplace = True)

    @property
    def energies(self):
        """Get list of energies used to take scans"""
        return self._table['Energy'].values

    def getEnergy(self, i):
        """
        Get energy used to take scan by scan number.

        Parameters
        ----------
        i : :obj:`int`
            Index of scan (0 for first scan, 1 for second, etc).

        Returns:
        :obj:`float`
            Energy of scan
        """
        return self.energies[i]

    def getIncidentIntensity(self, i):
        """
        Get incident intensity that a scan was taken with.

        Parameters
        ----------
        i : :obj:`int`, :obj:`float`
            Scan number or scan energy.

        Returns
        -------
        :obj:`float`
            Incident x-ray intensity scan was taken with.
        """
        if i in self.energies:
            return self._table.loc[self._table['Energy']==i, 'I0'].values[0]
        else:
            return self._table['I0'][i]

    def _readDataTable(fpath):
        """
        Function that reads settings file from run and loads data on each scan into
        a Pandas table with appropriate column headers.

        Parameters
        ----------
        fpath : :obj:`pathlib.Path`, :obj:`str`
            Path of file.

        Returns
        -------
        `pandas.DataFrame`
            Table containing data from file.
        """
        fpath = Path(fpath) # In case string was passed
        lines = fpath.read_text().splitlines()
        # Find index of heading block
        indices = [i for i in range(len(lines))
            if lines[i].endswith('Here is a readable list of column headings:')]
        if len(indices) == 0:
            raise ValueError("Could not find heading block in calibration info file.")
        else:
            hblockstart = indices[0]+1
        headings = {}
        for l in lines[hblockstart:]: # Looking at lines starting at beginning of heading block
            l = l.lstrip('#') # Remove comment markers
            # Strings are formatted like:
            # 1) headingA 3) headingC
            # 2) headingB 4) headingD
            hnums = [int(s.rstrip(')')) for s in re.findall(r'\d+\)', l)] # Split at the numbers
            hs = [s for s in re.split(r'\d+\) ', l) if not s.isspace()]
            if len(hs)==0: # Quit if strings no longer formatted like above
                break
            else:
                for n, h in zip(hnums, hs):
                    headings[n] = h.lstrip().rstrip() # Strip whitespace and associate heading with number
        headings = [headings[i] for i in range(1,len(headings)+1)] # Turn into list in-order
        # Read table and use extracted heading names
        data = pd.read_csv(fpath, delim_whitespace=True, comment='#',
            names=headings)
        return data
