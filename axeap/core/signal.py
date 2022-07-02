# Simple signal system
# Totally barebones, no typechecking

class Signal:
    """Signal that can trigger distribution of data to all connected functions.

    This class is used for a barebones event system. ``Signal`` objects are
    owned as visible attributes of objects. Callables, including methods of
    other objects, can be connected to this signal. These callables are referred
    to as the "listeners" of the signal. Then when the object that owns the
    signal emits it, all the connected listeners are called (in the order that
    they were connected) with arguments specified in the ``emit()`` call.
    """

    def __init__(self):
        self.listeners = []

    def connect(self, f):
        """Connects a listener callable.

        Parameters
        ----------
        f : :obj:`callable`
            The listener to connect to the signal.
        """
        self.listeners.append(f)

    def disconnect(self, f):
        """Disconnect a listener callable.

        Parameters
        ----------
        f : :obj:`callable`
            The listener to disconnect from the signal.
        """
        self.listeners.remove(f)

    def emit(self, *args, **kwargs):
        """Emit the signal.

        All connected listeners are called with the given arguments.

        Parameters
        ----------
        *args
            Positional arguments to send to listeners.
        **kwargs
            Keyword arguments to send to listeners.
        """
        for f in self.listeners:
            f(*args, **kwargs)
