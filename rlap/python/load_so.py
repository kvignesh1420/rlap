"""
_librlap.so loader for importing into python.
"""
import os
import ctypes
import sys
import inspect
import importlib
import types


def _load_library(filename):
    """_load_library"""

    # Construct filename
    dirnames = []
    datapath = os.environ.get("RLAP_DATAPATH", "bazel-bin")
    f = os.path.join(datapath, "rlap")
    dirnames.append(f)

    site_packages_file = sys.modules["rlap"].__file__
    site_packages_dir = os.path.dirname(site_packages_file)
    dirnames.append(site_packages_dir)

    for dirname in dirnames:
        sys.path.append(dirname)

    return importlib.import_module(filename)

class LazyLoader(types.ModuleType):
    def __init__(self, name, library):
        self._mod = None
        self._module_name = name
        self._library = library
        super().__init__(self._module_name)

    def _load(self):
        if self._mod is None:
            self._mod = _load_library(self._library)
        return self._mod

    def __getattr__(self, attrb):
        return getattr(self._load(), attrb)

    def __dir__(self):
        return dir(self._load())


_pywrap_librlap = LazyLoader("_librlap", "_librlap")

