'''Region of Interest (ROI)

ROIs define regions of an image.
'''

import numpy as np
from .conventions import X, Y
from .item import DataItem, DataItemSet

class ROI(DataItem):
    '''Region of Interest superclass.

    Used as parent of specific ROIs. Do not instantiate directly.
    '''

    def _selectArea(img):
        """
        Set all areas of image not in ROI to 0.

        Parameters
        ----------
        img : :obj:`numpy.ndarray`
            The image to process.
        """
        raise NotImplementedError("ROI is an abstract class and should not be instantiated")

    # def apply(self, scan:Scan) -> Scan:
    #     return Scan(self._selectArea(scan.getImg()).filled(0))

    def __repr__(self):
        return "ROI()"

    def __hash__(self):
        """Hash is computed as hash of string representation of ROI. String
        representation must therefore be comprehensive of data encoded by ROI.
        """
        return hash(tuple(bytes(repr(self),'ascii')))   # Hack much?


class ComboROI(ROI):
    """ROI composed of a combination of other ROIs.
    """
    def __init__(self, *rois, **kwargs):
        """
        Parameters
        ----------
        *rois : List of :obj:`ROI`)
            The component ROIs to combine.
        """
        ROI.__init__(self, **kwargs)
        self.comps = rois   # Component ROIs

    def _selectArea(self, img):
        invmask = None
        for roi in self.comps:
            if invmask is None: invmask = ~roi._selectArea(img).mask
            else: invmask = invmask + ~roi._selectArea(img).mask
        return np.ma.array(img, mask= ~invmask)

    def __repr__(self):
        return "ComboROI("+"".join([str(c)+',' for c in self.comps])+")"


class RectangleROI(ROI):
    '''Rectangular ROI
    '''

    def __init__(self, p1, p2, **kwargs):
        '''
        Parameters
        ----------
        p1 : array-like
            Coordinates (x,y) of one corner of rectangular.
        p2 : array-like
            Coordinates (x,y) of diagonally opposite corner to `p1`.
        '''
        ROI.__init__(self, **kwargs)
        self.lox, self.hix = sorted((p1[X],p2[X]))
        self.loy, self.hiy = sorted((p1[Y],p2[Y]))
        self.lox, self.loy, self.hix, self.hiy = \
            (int(np.ceil(self.lox)), int(np.ceil(self.loy)), \
            int(np.ceil(self.hix)), int(np.ceil(self.hiy)))

    def fromHVROIs(hroi, vroi):
        '''Create Rectangular ROI from horizontal and vertical ROIs.

        Parameters
        ----------
        hroi : :obj:`HROI`
            Horizontal ROI.
        vroi : :obj:`.VROI`
            Vertical ROI.

        Returns
        -------
        :obj:`RectangleROI`
            Rectangular ROI spanning vertical and horizontal regions.
        '''
        return RectangleROI((hroi.lo, vroi.lo), (hroi.hi, vroi.hi))

    def _selectArea(self, img):
        # Use slices to create no-copy view
        # return np.ma.array(img[self.lox:self.hix+1,
        #                     self.loy:self.hiy+1], copy=False)
        # Let's try masks instead
        mask = np.ones(img.shape)
        mask[self.lox:self.hix+1,self.loy:self.hiy+1] = 0
        return np.ma.array(img, mask=mask)

    def __repr__(self):
        return f"RectangleROI(({self.lox},{self.loy}), ({self.hix},{self.hiy}))"

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.lox==other.lox and self.loy==other.loy and \
                    self.hix==other.hix and self.hiy==other.hiy
        else:
            return False

    def __hash__(self):
        return ROI.__hash__(self)


class EllipseROI(ROI):
    """Ellipse-shaped ROI"""

    def __init__(self, p, rx, ry, **kwargs):
        """
        Parameters
        ----------
        p : array-like
            Coordinates (x,y) for center of ellipse.
        rx : :obj:`float`
            Radius along x-axis.
        ry : :obj:`float`
            Radius along y-axis.
        """
        ROI.__init__(self, **kwargs)
        self.x, self.y, self.rx, self.ry = *p, rx, ry

    def _selectArea(self, img):
        mask = np.ones(img.shape)
        for y in np.arange(np.ceil(self.y-self.ry), np.ceil(self.y+self.ry), 1):
            #print('y', y)
            halfwidth = self.rx*(1-((self.y-y)/self.ry)**2)**(1/2)
            lox, hix = self.x-halfwidth, self.x+halfwidth
            for x in np.arange(np.ceil(lox), np.ceil(hix), 1):
                #print('x', x)
                mask[int(x),int(y)] = 0
        return np.ma.array(img, mask=mask)
        # Could use cv2.ellipse instead

    def __repr__(self):
        return f"EllipseROI(({self.x},{self.y})," \
            f"{self.rx},{self.ry})"

class SpanROI(ROI):
    """
    Abstract superclass to :obj:`HROI` and :obj:`VROI`. Do not instantiate.
    """
    def __init__(self, p1, p2, **kwargs):
        """
        Parameters
        ----------
        p1 : :obj:`float`
            Lower end of span.
        p2 : :obj:`float`
            Higher end of span.
        """
        ROI.__init__(self, **kwargs)
        self.lo, self.hi = sorted((int(np.ceil(p1)), int(np.ceil(p2))))

    def __iter__(self):
        return iter((self.lo, self.hi))


class HROI(SpanROI):
    """ROI spanning range along x-axis."""

    def _selectArea(self, img):
        mask = np.ones(img.shape)
        mask[self.lo:self.hi+1,:] = 0
        return np.ma.array(img, mask=mask)

    def __repr__(self):
        return f"HROI(x={self.lo}:{self.hi})"

    def __mul__(self, other):
        return RectangleROI.fromHVROIs(self, other)


class VROI(SpanROI):
    """ROI spanning range along y-axis."""

    def _selectArea(self, img):
        mask = np.ones(img.shape)
        mask[:, self.lo:self.hi+1] = 0
        return np.ma.array(img, mask=mask)

    def __repr__(self):
        return f"VROI(y={self.lo}:{self.hi})"

    def __mul__(self, other):
        return RectangleROI.fromHVROIs(other, self)


class ROISet(DataItemSet):
    """:obj:`.core.item.DataItemSet` for :obj:`ROI`'s
    """

    def getROIsInside(self, nroi):
        """Get ROIs in set contained by given ROI

        Parameters
        ----------
        nroi : :obj:`ROI`
            ROI used to choose ROIs from set.

        Returns
        -------
        :obj:`list`
            List of ROIs contained in nroi.
        """
        if not isinstance(nroi, RectangleROI):
            return []   # TODO: Add functionality for non-rectangular ROIs
        inside = [roi for roi in self if \
            roi.lox >= nroi.lox and roi.loy >= nroi.loy and \
            roi.hix <= nroi.hix and roi.hiy <= nroi.hiy]
        return inside
