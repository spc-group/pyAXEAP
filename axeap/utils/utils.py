"""Utility functions.

This file contains generic utility functions for doing simple simple math or
string operations.
"""

import numpy as np
import random
import re


def getCoordsFromImage(image, area=None, subsample=None):
    """Gets the list of non-zero pixels from an image.

    Parameters
    ----------
    image : array_like
        2D array of intensity values, indexable as `image[x][y]`.
    area : array_like, optional
        Rectangular area to extract coordinates of, given as bottom left corner
        and top right corner in form [[x1,y1],[x2,y2]].
    subsample : :obj:`float`, optional
        Fraction of points to randomly sample and return, default is to return
        all points.

    Returns
    -------
    :obj:`list`
        List of points in the form (x,y,z) where x and y are the pixel position
        and z is the pixel value.
    """

    RANDOM_SEED = 1234  # Totally random, I assure you

    if image is None:
        return ()
    if area is None:
        lox=0
        loy=0
        hix=image.shape[0]
        hiy=image.shape[1]
    else:
        lox = area[0][0]
        loy = area[0][1]
        hix = area[1][0]
        hiy = area[1][1]
    coords = list()
    for xval in range(int(np.ceil(lox)), int(np.ceil(hix))):
        for yval in range(int(np.ceil(loy)), int(np.ceil(hiy))):
            if image[xval][yval] != 0:
                coords.append((xval, yval, image[xval][yval]))
    if subsample is None: return coords
    else:
        random.seed(RANDOM_SEED)
        return random.sample(coords, round(len(coords)*subsample))


def linearoverlap(l1, l2):
    """Get overlap between two ranges as fraction of length of each range.

    Parameters
    ----------
    l1 : array_like
        First range, given in form [x1,x2] where x1 and x2 are ends of range.
    l2 : array_like
        Second range, same form as l1.

    Returns
    -------
    :obj:`tuple`
        Fraction of each range that overlaps with the other, in form
        (fraction of l1 that overlaps, fraction of l2 that overlaps).
    """
    l1.sort()
    l2.sort()
    l1_len = abs(l1[0]-l1[1])
    l2_len = abs(l2[0]-l2[1])
    overlap = (max(l1[0], l2[0]), min(l1[1], l2[1]))
    overlaplen = overlap[1]-overlap[0]
    return (overlaplen/l1_len, overlaplen/l2_len)


def linearlen(s):
    """Get length of a span from endpoints of span.

    Note
    ----
    Used to make some code look nicer, but may be removed in future due it being
    a very simple operation.

    Parameters
    ----------
    s : array_like
        Span in form [x1,x2] where x1 and x2 are ends of span.

    Returns
    -------
    number
        Length of span as absolute difference between x1 and x2.
    """
    return abs(s[0]-s[1])


def separateSpans(spans, buffer):
    """Remove overlap between spans.

    Parameters
    ----------
    spans : array_like
        List of spans in form [x1,x2] where x1 and x2 are endpoints of span.
    buffer : :obj:`float`
        Amount of space to separate overlapping spans by, must be a positive
        number.

    Returns
    -------
    :obj:`list`
        List of spans after being separated.
    """
    spans = sorted(spans)
    for s1 in range(len(spans)):
        for s2 in range(len(spans)):
            if s1==s2: continue
            span1 = spans[s1]
            span2 = spans[s2]

            if span1[1]<span2[1] and (span2[0]-span1[1])<buffer:
                midpt = (span2[0]+span1[1])/2
                span1[1] = midpt-buffer/2
                span2[0] = midpt+buffer/2
            if span2[1]<span1[1] and (span1[0]-span2[1])<buffer:
                midpt = (span1[0]+span2[1])/2
                span2[1] = midpt-buffer/2
                span1[0] = midpt+buffer/2
    return spans


def get_trailing_number(s, default=None):
    """Get number at the end of a string.

    Used primarily for extracting number from filename. For example, a file
    whose stem is 'Image021' will have the number 21 extracted.

    Parameters
    ----------
    s : :obj:`str`
        String to get trailing number from.
    default : any, optional
        Default value to return if no number found.

    Returns
    -------
    :obj:`int`
        Number extracted from end of string.
    """
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else default
