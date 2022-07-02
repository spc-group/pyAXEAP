"""All experiment related classes and functions. (Used for live experimentation,
not just inspecting data after experiments)"""

import logging
logger = logging.getLogger('axeap')

from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ..core import Scan, ScanSet, CalibRunInfo


class ScanSetDirWatcher(FileSystemEventHandler):
    """Watch a directory and populate a `.scan.ScanSet` with any files that
    show up.
    """
    def __init__(self, scanset, dir, calibration_mode=False):
        """
        Parameters
        ----------
        scanset : :obj:`.core.scan.ScanSet`)
            ScanSet to populate.
        dir : :obj:`pathlib.Path`, :obj:`str`
            Path to directory to be watched.
        calibration_mode : :obj:`bool`
            If True, calibration info file will also be loaded if added to dir.
        """
        super().__init__()
        self.scanset = scanset
        self.dir = Path(dir) # Convert to Path in case dir is str
        self.is_calibration_run = calibration_mode
        self.observer = Observer()
        self.observer.schedule(self, self.dir, recursive=False)


    def on_any_event(self, event):
        # TODO: Catch file deletion events and remove corresponding SS scans
        p = Path(event.src_path)
        if event.is_directory or event.event_type != 'created':
            return
        extension = p.suffix.lstrip('.').lower()
        if extension in Scan.FILE_EXTENTIONS:   # New scan added
            logger.debug(f'New scan found: {p}')
            s = Scan.loadFromFile(p)
            self.scanset.addScan(s)
        elif self.is_calibration_run and extension.isnumeric(): # Config file added
            logger.debug(f'Calibration run info file found: {p}')
            logger.debug(f'Quitting observerving {self.dir} since calibration run info file detected.')
            self.observer.stop()
            self.scanset.addCalibRunInfo(CalibRunInfo(p))

    def start_watching(self):
        self.observer.start()

    def stop_watching(self):
        self.observer.stop()
