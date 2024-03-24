# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines functions that import external libraries used by Uproot, but not
required by an Uproot installation. (Uproot only requires NumPy).

If a library cannot be imported, these functions raise ``ModuleNotFoundError`` with
error messages containing instructions on how to install the library.
"""
from __future__ import annotations

import atexit
import importlib.metadata
import os

from uproot._util import parse_version


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
    if parse_version(awkward.__version__) >= parse_version("2.4.6"):
        return awkward
    else:
        raise ModuleNotFoundError(
            f"Uproot 5.1+ can only be used with Awkward 2.4.6 or newer; you have Awkward {awkward.__version__}"
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


def Minio_client():
    """
    Imports and returns ``minio.Minio``.
    """
    try:
        from minio import Minio
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'minio' package with:

    pip install minio

or

    conda install minio"""
        ) from err
    else:
        return Minio


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
        return importlib.metadata.version("xrootd")
    except ModuleNotFoundError:
        try:
            # Versions before 4.11.1 used pyxrootd as the package name
            return importlib.metadata.version("pyxrootd")
        except ModuleNotFoundError:
            return None


def isal():
    """
    Import and return ``isal``.
    """
    try:
        import isal
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'isal' package with:

    pip install isal

or

    conda install python-isal"""
        ) from err
    else:
        return isal


def deflate():
    """
    Import and return ``deflate``.
    """
    try:
        import deflate
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'deflate' package with:

    pip install deflate

or

    conda install libdeflate"""
        ) from err
    else:
        return deflate


def cramjam():
    """
    Import and returns ``cramjam``.
    """
    try:
        import cramjam
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'cramjam' package with:

    pip install cramjam

or

    conda install cramjam"""
        ) from err
    else:
        return cramjam


def xxhash():
    """
    Imports and returns ``xxhash``.
    """
    try:
        import xxhash
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the `xxhash` packages with:

    pip install xxhash

or

    conda install python-xxhash"""
        ) from err
    else:
        return xxhash


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
    conda install -c conda-forge dask dask-awkward"""
        ) from err
    if parse_version(dask_awkward.__version__) >= parse_version("2023.10.0"):
        return dask_awkward
    else:
        raise ModuleNotFoundError(
            f"Uproot 5.1+ can only be used with dask-awkward 2023.10.0 or newer; you have dask-awkward {dask_awkward.__version__}"
        )


def awkward_pandas():
    """
    Imports and returns ``awkward_pandas``.
    """
    try:
        import awkward_pandas
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'awkward-pandas' package with:
    pip install awkward-pandas
or
    conda install -c conda-forge awkward-pandas"""
        ) from err
    else:
        return awkward_pandas
