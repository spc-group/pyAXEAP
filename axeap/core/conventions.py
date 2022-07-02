'''Conventions.
Shorthands and enums used throughout axeap.
'''

import enum

'''Axis names for 2D arrays. Primary axis is x-axis, secondary is y-axis.'''
X = 0
Y = 1

class PathType(enum.Enum):
    FILE = enum.auto()
    FILES = enum.auto()
    DIR = enum.auto()
