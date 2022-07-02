"""Automatic Calculation of Regions of Interest.
"""

import numpy as np
import cv2

from .conventions import X, Y
from .scan import Scan, ScanSet
from .roi import HROI, VROI
from ..utils import linearoverlap, separateSpans

# # Return ComboROI with background guesses
# def findBackground(ss, tile_dims=None, mindivisions=10):
#     if tile_dims is None:
#         shorterdim = np.argmin(ss.dims)
#         d = ss.dims[shorterdim]/mindivisions
#         tile_dims = (d,d)
#
#     tilemeans = np.zeros(shape=(20,10,maximg+1))
#     regs = [[0]*10 for i in range(20)]
#     for i in range(20):
#         for j in range(10):
#             reg = RROI((int(boxdims[X]*i),int(boxdims[Y]*j)),(int(boxdims[X]*(i+1)),int(boxdims[Y]*(j+1))))
#             regs[i][j] = reg
#             for n,s in enumerate(ss.scans[:maximg+1]):
#                 boxmeans[i,j,n] = np.mean(s.getImg(roi=reg))
#     regs = np.array(regs)
#     boxstds = np.std(boxmeans,axis=2)
#     bkgboxindices = np.where(boxstds<np.mean(boxstds)/10)
#     bkgregs = regs[bkgboxindices[0], bkgboxindices[1]]
#     return ComboROI(*bkgregs)
#
# def guessLowcut(s, background=None):
#     ... # Guess locut that will eliminate background

def calcHROIs(
    scan,
    min_width,
    group_buffer=0,
    return_lines=False
    ):
    """
    Static function to separate image into columns based on line segments found
    in image.

    This works by combining line segements that overlap in their horizontal
    ranges by more than 50% of their length to form groups. These groups are
    then trimmed so as to not overlap. This is meant to be used with _calcLines
    to find the horizontal ranges of pixels in the scan image that correspond to
    each crystal. However, if two crystals are lined up such that they produce
    a single line segment in the image, this function will create one group
    spanning both crystals.

    Parameters
    ----------
    scan : :obj:`.core.scan.Scan`
        Scan to calculate horizontal regions of interest for.
    group_buffer : :obj:`int`
        Minimum number of pixels to separate groups by.
    return_lines : :obj:`bool`
        Whether or not to return the lines segments detected in the image.

    Returns
    -------
    :obj:`list`
        List of :obj:`core.roi.HROI`'s representing regions corresponding to
        each crystal.
    :obj:`np.ndarray`, optional
        Array of line segments detected in image, encoded as [[x1,y1],[x2,y2]].
        Returned if ``return_lines`` is set :obj:`True`.
    """
    s = scan.mod(blur=3)
    lines2d = calcHorizontalLineSegments(s, round(min_width))
    flattened = [[l[0][X], l[1][X]] for l in lines2d] # Flatten to just x coords
    midyvals = [(l[0][Y]+l[1][Y])/2 for l in lines2d] # Midpoint y values
    lines = [{'flat':flattened[i], 'midy': midyvals[i]} \
        for i in range(len(flattened))]
    # Combine overlapping lines
    checkcomplete = False
    while not checkcomplete:
        matchfound = False
        for l1 in range(len(lines)):
            for l2 in range(len(lines)):
                if l1==l2: continue
                flat1 = lines[l1]['flat']
                flat2 = lines[l2]['flat']
                overlap = linearoverlap(flat1, flat2)
                if overlap[0]>0.5 or overlap[1]>0.5:
                    newflat = [min(flat1[0], flat2[0]), max(flat1[1],flat2[1])]
                    newmidy = (lines[l1]['midy']+lines[l2]['midy'])/2
                    newline = {'flat':newflat, 'midy':newmidy}
                    lines.pop(max(l1,l2))
                    lines.pop(min(l1,l2))
                    lines.append(newline)
                    matchfound = True
                    break
            if matchfound:
                break
        if not matchfound:
            checkcomplete = True
    groups = [l['flat'] for l in lines]
    # Eliminate overlap between groups
    groups = separateSpans(groups, group_buffer)
    groups = np.array(groups)
    groups = groups[np.argsort(groups[:,0])]
    hrois = [HROI(g[0],g[1]) for g in groups if g[1]-g[0]>min_width]
    if return_lines:
        return hrois, lines2d
    else:
        return hrois


def calcHorizontalLineSegments(scan, min_line_length, anglerange=None):
    """
    Function that calculates line segments from scan.

    Parameters
    ----------
    scan : :obj:`.scan.Scan`
        Scan to calculate lines from.
    min_line_length : :obj:`float`
        Minimum length line segments must be.
    anglerange : :obj:`tuple`
        Range of angles that line segments must be within. Horizontal is 0,
        vertical is pi/2. Range encoded as (t1,t2).

    Returns
    -------
    :obj:`numpy.ndarray`
        Array of line segments encoded as [[x1,y1],[x2,y2]]
    """
    anglerange = anglerange or (0, np.pi/6)
    fld = cv2.ximgproc.createFastLineDetector(length_threshold=min_line_length,\
                                                do_merge=True)
    img = scan.getImg().astype(int)
    img[np.where(img>0)] = 255
    rawlines = fld.detect(img.astype(np.uint8))
    if rawlines is None:
        return []
    else:
        lines = np.array(rawlines)[:,0,:]
    lines[:,[X,Y]] = lines[:,[Y,X]]     # Flip coords to x,y
    lines[:,[X+2,Y+2]] = lines[:,[Y+2,X+2]]
    tan_angles = np.abs((lines[:,Y]-lines[:,Y+2])/(lines[:,X]-lines[:,X+2]))
    filteredlines = np.delete(lines, np.where(np.logical_and(
        tan_angles<np.tan(anglerange[0]), tan_angles>np.tan(anglerange[1]))),
        axis=0) # Only take lines inside angle range
    # Split into pairs of points
    lines2d = np.array([[[l[X],l[Y]],[l[X+2],l[Y+2]]] for l in filteredlines])
    # Sort each line by x values of points
    lines2d = np.array([l[np.argsort(l[:,0])] for l in lines2d])
    return lines2d


def calcVROIs(
    scan,
    minheight,
    maxgap,
    cutoff_frac=0.1,
    buffer=0
):
    """Calculate vertical regions of interest (VROIs) for a scan.

    Parameters
    ----------
    scan : :obj:`.core.scan.Scan`
        Scan to calculate VROIs for.
    minheight : :obj:`float`
        Minimum height of VROI.
    maxgap : :obj:`int`
        Maximum empty (no non-zero pixels) space allowed within a VROI.
    cutoff_frac : :obj:`float`
        Fraction of peak intensity to use as low-cut.
    buffer : :obj:`int`
        Extra space to add around VROI.
    """
    img = scan.getImg().copy()
    img[img>0]=1
    vd = img.sum(0)
    cvd = vd.copy()
    cvd[cvd<(np.max(vd)*cutoff_frac)] = 0
    vroistart = -1
    lastsaw = None
    vrois = []
    for x in range(scan.dims[Y]):
        if cvd[x]>0: # If in a stretch
            lastsaw = x
            if vroistart == -1: vroistart = x
        elif vroistart != -1 and x-lastsaw > maxgap: # If at a zero and empty stretch is fraction of roi so far
            if lastsaw-vroistart > minheight: # If the length of thr roi is more than the minimum, complete it
                lo = vroistart-buffer if vroistart>buffer else 0
                hi = lastsaw+buffer if (lastsaw+buffer)<scan.dims[Y] else scan.dims[Y]
                vrois.append(VROI(lo, hi))
                vroistart = -1
            else: vroistart = -1
    if vroistart!=-1 and lastsaw-vroistart > minheight:
        vrois.append(vrois.append(VROI(vroistart, lastsaw)))
    return vrois


# def calcVROIs(
#     scan:Scan,
#     approx_height:Number,
#     min_separation:Number=None, #TODO
#     tightness=0.75
#     ) -> Tuple[VROI]:
#     """
#     Function that calculates vertical regions of interest (VROIs) for each scan
#     in a scan set. Each VROI is encoded as the y-pixel number of the center of
#     the region and the total width of the region in vertical pixels.
#
#     Args:
#         scanset: set of scans to calculate VROIs for
#
#     Returns:
#         VROIs: Dictionary keyed by scan energy of VROIs encoded as y pixel
#             values in tuple (center, width)
#         numregions: Dictionary keyed by scan energy of number of possible ROIs
#     """
#     img = scan.getImg().copy()
#     img[img>0]=1
#     print(np.max(img))
#     vertdensity = img.sum(0)
#     blurv = gaussian_filter(vertdensity, sigma=approx_height) # Assumes calbration range covers entire CCD
#     peaks, _ = find_peaks(blurv, height=blurv.min()+(blurv.max()-blurv.min())/10)  # peaks >10% max, measured from min
#     rel_height = tightness #if len(peaks) == 1 else 1
#     width_data = peak_widths(blurv, peaks, rel_height=rel_height)
#     vrois = [VROI(p-w/2,p+w/2) for p,w in zip(peaks, width_data[0])]
#     return (vrois)  # Convert to tuple since shouldn't be modified later
