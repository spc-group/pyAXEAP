from .conventions import PathType
from .signal import Signal

import uuid

# Dev Note: I could try metaclasses but that seems like a rabbit hole of
# abstract typing.
class DataItem:
    """Superclass for all data-containing objects.

    This is the superclass for objects such as :obj:`.core.scan.Scan` or
    :obj:`.core.spectra.Spectra`.

    Attributes
    ----------
    TYPE_NAME : :obj:`str`
        The name of the data item type, for example "Scan". Same across all
        instances.
    name : :obj:`str`
        The name of the data item, not necessarily unique.
    uuid : :obj:`uuid.UUID`
        UUID of the data item. Unique.
    meta : :obj:`dict`
        Dictionary of metadata describing instance of DataItem.
    """

    TYPE_NAME = "DataItem"

    def __init__(self, name=None, meta=None):
        """
        Parameters
        ----------
        name : :obj:`str`, optional
            Name to assign to item. Should be human-readable, not necessarily
            unique.
        meta : :obj:`dict`, optional
            Metadata concerning the item. Keys should be strings.
        """
        self.name = name if name else ""
        self.uuid = uuid.uuid4()
        self.meta = meta.copy() if meta else {}


class Saveable:
    """Superclass for any object that can be saved to a file.

    Attributes
    ----------
    SAVE_FILE_EXTENTIONS : :obj:`list`
        List of strings containing file extensions that the object can be saved
        under. Extensions are lowercase and do not include dots (example "txt"
        instead of ".TXT"). Should be overwritten in subclass definition.
    SAVE_PATH_TYPE : :obj:`.core.conventions.PathType`
            Specifies whether the object can be saved as a file or a directory.
            Should be overwritten by subclass definition.
    """

    SAVE_FILE_EXTENTIONS = []
    SAVE_PATH_TYPE = PathType.FILE

    def saveToPath(self, path):
        """
        Save data stored in object to a given path.

        Parameters
        ----------
        path : :obj:`pathlib.Path`, :obj:`str`
            Path to store object data to. Could be file or directory.
        """
        raise NotImplementedError('Saving to path has not been implemented!')


class Loadable:
    """Superclass for any object that can be loaded from a file.

    Attributes
    ----------
    LOAD_FILE_EXTENTIONS : :obj:`list`
        List of strings containing file extensions that the object can be loaded
        from. Extensions are lowercase and do not include dots (example "txt"
        instead of ".TXT"). Should be overwritten by subclass definition.
    SAVE_PATH_TYPE : :obj:`.core.conventions.PathType`
        Specifies whether the object can be loaded from a file, multiple files,
        or a directory. Should be overwritten by subclass but constant across
        instances.
    """

    LOAD_FILE_EXTENTIONS = []
    LOAD_PATH_TYPE = PathType.FILE

    def loadFromFile(path):
        """Load data from file and use to instantiate class.

        Parameters
        ----------
        path : :obj:`pathlib.Path`, :obj:`str`
            Path of file containing data.

        Returns
        -------
        :obj:`Loadable`
            Instance of object, populated with data from file.
        """
        raise NotImplementedError('Loading from path has not been implemented!')


class DataItemSet:
    """Set of :obj:`DataItem` objects.

    This is essentially just a list of :obj:`DataItem` objects but
    with the additional features of allowing items to be "selected" or not and
    emitting signals when the membership of the :obj:`DataItemSet` changes or
    the selection status of one of the :obj:`DataItem`'s changes. This is
    the superclass for :obj:`.core.scan.ScanSet`, :obj:`.core.roi.ROISet`, and
    :obj:`.core.spectra.SpectraSet`.

    Note
    ----
        Though this uses a list to store the underlying data, users should not
        depend on the order of the items in the set.

    Attributes
    ----------
    itemadded : :obj:`.core.signal.Signal`
        When item is added, signal is emitted with item as argument.
    itemremoved : :obj:`.core.signal.Signal`
        When item is removed, signal emitted with item as argument.
    selectionchanged : :obj:`.core.signal.Signal`
        When item selection is changed, signal emitted with signature (item,
        selection).
    """

    def __init__(self, items=None, selection_default=True, *args, **kwargs):
        """
        Parameters
        ----------
        items : :obj:`list`, optional
            List of :obj:`DataItem` objects to initially populate the
            :obj:`DataItemSet`.
        selection_default : :obj:`bool`, optional
            Default selection value for items after being added to set. Default
            default value is `True`.
        """
        self.itemadded = Signal()
        self.itemremoved = Signal()
        self.selectionchanged = Signal()

        self.items = []
        self.selections = {}
        self.selection_default = selection_default

        if items is not None:
            for item in items:
                self.add(item)

    def add(self, item):
        """Add :obj:`DataItem` to set."""
        self.items.append(item)
        self.selections[item] = self.selection_default
        self.itemadded.emit(item)

    def remove(self, item):
        """Remove :obj:`DataItem` from set."""
        self.items.remove(item)
        self.selections.pop(item)
        self.itemremoved.emit(item)

    def setSelection(self, item, sel):
        """Change selection status of :obj:`DataItem`

        Parameters
        ----------
        item : :obj:`DataItem`
            Item whose selection status to change.
        sel : :obj:`bool`
            New selection status.
        """
        if self.selections[item] != sel:
            self.selections[item] = sel
            self.selectionchanged.emit(item, sel)

    def getSelection(self, item):
        """Get selection status of item."""
        return self.selections[item]

    def getSelected(self):
        """Get list of selected items."""
        return [i for i in self if self.getSelection(i)]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        """Get iterator over all items in set (not just selected ones)."""
        return iter(self.items)
