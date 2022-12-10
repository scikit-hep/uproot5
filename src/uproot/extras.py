# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines functions that import external libraries used by Uproot, but not
required by an Uproot installation. (Uproot only requires NumPy).

If a library cannot be imported, these functions raise ``ModuleNotFoundError`` with
error messages containing instructions on how to install the library.
"""


import atexit
import os
import sys

from uproot._util import parse_version

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


def awkward():
    """
    Imports and returns ``awkward``.
    """
    try:
        import awkward
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'awkward' package with:

    pip install awkward

Alternatively, you can use ``library="np"`` or globally set ``uproot.default_library``
to output as NumPy arrays, rather than Awkward arrays.
"""
        ) from err
    if parse_version("2") <= parse_version(awkward.__version__):
        return awkward
    else:
        raise ModuleNotFoundError(
            "Uproot 5.x can only be used with Awkward 2.x; you have Awkward {}".format(
                awkward.__version__
            )
        )


def pandas():
    """
    Imports and returns ``pandas``.
    """
    try:
        import pandas
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'pandas' package with:

    pip install pandas

or

    conda install pandas"""
        ) from err
    else:
        return pandas


def XRootD_client():
    """
    Imports and returns ``XRootD.client`` (after setting the
    ```XRD_RUNFORKHANDLER`` environment variable to ``"1"``, to allow
    multiprocessing).
    """
    os.environ["XRD_RUNFORKHANDLER"] = "1"  # set multiprocessing flag
    try:
        import XRootD
        import XRootD.client

    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """Install XRootD python bindings with:

    conda install -c conda-forge xrootd

(or download from http://xrootd.org/dload.html and manually compile with """
            """cmake; setting PYTHONPATH and LD_LIBRARY_PATH appropriately)."""
        ) from err

    if older_xrootd("5.1.0"):
        # This is registered after calling "import XRootD.client" so it is ran
        # before XRootD.client.finalize.finalize()
        @atexit.register
        def cleanup_open_files():
            """Clean up any open xrootd file objects at exit

            Required to avoid deadlocks from XRootD, for details see:
            * https://github.com/scikit-hep/uproot/issues/504
            * https://github.com/xrootd/xrootd/pull/1260
            """
            import gc

            for obj in gc.get_objects():
                try:
                    isopen = isinstance(obj, XRootD.client.file.File) and obj.is_open()
                except ReferenceError:
                    pass
                else:
                    if isopen:
                        obj.close()

    return XRootD.client


def older_xrootd(min_version):
    """
    Check if the installed XRootD bindings are newer than a given version
    without importing. Defaults to False if XRootD is not installed. Unrecognized
    versions (i.e. self-built XRootD, whose version numbers are strings)
    return False: that is, they're assumed to be new, so that no warnings
    are raised.
    """
    version = xrootd_version()
    if version is None:
        return False
    else:
        try:
            return parse_version(version) < parse_version(min_version)
        except TypeError:
            return False


def xrootd_version():
    """
    Gets the XRootD version if installed, otherwise returns None.
    """
    try:
        return importlib_metadata.version("xrootd")
    except ModuleNotFoundError:
        try:
            # Versions before 4.11.1 used pyxrootd as the package name
            return importlib_metadata.version("pyxrootd")
        except ModuleNotFoundError:
            return None


def lzma():
    """
    Imports and returns ``lzma`` (which is part of the Python 3 standard
    library, but not Python 2).
    """
    import lzma

    return lzma


def lz4_block():
    """
    Imports and returns ``lz4``.

    Attempts to import ``xxhash`` as well.
    """
    try:
        import lz4.block
        import xxhash  # noqa: F401
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'lz4' and `xxhash` packages with:

    pip install lz4 xxhash

or

    conda install lz4 python-xxhash"""
        ) from err
    else:
        return lz4.block


def xxhash():
    """
    Imports and returns ``xxhash``.

    Attempts to import ``lz4`` as well.
    """
    try:
        import lz4.block  # noqa: F401
        import xxhash
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'lz4' and `xxhash` packages with:

    pip install lz4 xxhash

or

    conda install lz4 python-xxhash"""
        ) from err
    else:
        return xxhash


def zstandard():
    """
    Imports and returns ``zstandard``.
    """
    try:
        import zstandard
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'zstandard' package with:

    pip install zstandard

or

    conda install zstandard"""
        ) from err
    else:
        return zstandard


def boost_histogram():
    """
    Imports and returns ``boost-histogram``.
    """
    try:
        import boost_histogram
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'boost-histogram' package with:

    pip install boost-histogram

or

    conda install -c conda-forge boost-histogram"""
        ) from err
    else:
        return boost_histogram


def hist():
    """
    Imports and returns ``hist``.
    """
    try:
        import hist
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'hist' package with:

    pip install hist"""
        ) from err
    else:
        return hist


def dask():
    """
    Imports and returns ``dask``.
    """
    try:
        import dask
        import dask.blockwise
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """for uproot.dask with 'library="np"', install the complete 'dask' package with:
    pip install "dask[complete]"
or
    conda install dask"""
        ) from err
    else:
        return dask


def dask_array():
    """
    Imports and returns ``dask.array``.
    """
    try:
        import dask.array as da
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """for uproot.dask with 'library="np"', install the complete 'dask' package with:
    pip install "dask[complete]"
or
    conda install dask"""
        ) from err
    else:
        return da


def dask_awkward():
    """
    Imports and returns ``dask_awkward``.
    """
    try:
        import dask_awkward
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """for uproot.dask, install 'dask' and the 'dask-awkward' package with:
    pip install "dask[complete] dask-awkward"
or
    conda install dask
    pip install dask-awkward   # not on conda-forge yet"""
        ) from err
    else:
        return dask_awkward


def awkward_pandas():
    """
    Imports and returns ``awkward_pandas``.
    """
    try:
        import awkward_pandas
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'awkward-pandas' package with:
    pip install awkward-pandas # not on conda-forge yet"""
        ) from err
    else:
        return awkward_pandas
