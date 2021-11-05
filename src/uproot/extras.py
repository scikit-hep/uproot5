# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines functions that import external libraries used by Uproot, but not
required by an Uproot installation. (Uproot only requires NumPy).

If a library cannot be imported, these functions raise ``ImportError`` with
error messages containing instructions on how to install the library.
"""

from __future__ import absolute_import

import atexit
import os
from distutils.version import LooseVersion

import pkg_resources


def awkward():
    """
    Imports and returns ``awkward``.
    """
    try:
        import awkward
    except ImportError:
        raise ImportError(
            """install the 'awkward' package with:

    pip install awkward

Alternatively, you can use ``library="np"`` or globally set ``uproot.default_library``
to output as NumPy arrays, rather than Awkward arrays.
"""
        )
    if LooseVersion("1") < LooseVersion(awkward.__version__) < LooseVersion("2"):
        return awkward
    else:
        raise ImportError(
            "Uproot 4.x can only be used with Awkward 1.x; you have Awkward {0}".format(
                awkward.__version__
            )
        )


def pandas():
    """
    Imports and returns ``pandas``.
    """
    try:
        import pandas
    except ImportError:
        raise ImportError(
            """install the 'pandas' package with:

    pip install pandas

or

    conda install pandas"""
        )
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

    except ImportError:
        raise ImportError(
            """Install XRootD python bindings with:

    conda install -c conda-forge xrootd

(or download from http://xrootd.org/dload.html and manually compile with """
            """cmake; setting PYTHONPATH and LD_LIBRARY_PATH appropriately)."""
        )

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
            return LooseVersion(version) < LooseVersion(min_version)
        except TypeError:
            return False


def xrootd_version():
    """
    Gets the XRootD version if installed, otherwise returns None.
    """
    try:
        version = pkg_resources.get_distribution("XRootD").version
    except pkg_resources.DistributionNotFound:
        try:
            # Versions before 4.11.1 used pyxrootd as the package name
            version = pkg_resources.get_distribution("pyxrootd").version
        except pkg_resources.DistributionNotFound:
            version = None
    return version


def lzma():
    """
    Imports and returns ``lzma`` (which is part of the Python 3 standard
    library, but not Python 2).
    """
    try:
        import lzma
    except ImportError:
        try:
            import backports.lzma as lzma
        except ImportError:
            raise ImportError(
                """install the 'lzma' package with:

    pip install backports.lzma

or

    conda install backports.lzma

or use Python >= 3.3."""
            )
        else:
            return lzma
    else:
        return lzma


def lz4_block():
    """
    Imports and returns ``lz4``.

    Attempts to import ``xxhash`` as well.
    """
    try:
        import lz4.block
        import xxhash  # noqa: F401
    except ImportError:
        raise ImportError(
            """install the 'lz4' and `xxhash` packages with:

    pip install lz4 xxhash

or

    conda install lz4 python-xxhash"""
        )
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
    except ImportError:
        raise ImportError(
            """install the 'lz4' and `xxhash` packages with:

    pip install lz4 xxhash

or

    conda install lz4 python-xxhash"""
        )
    else:
        return xxhash


def zstandard():
    """
    Imports and returns ``zstandard``.
    """
    try:
        import zstandard
    except ImportError:
        raise ImportError(
            """install the 'zstandard' package with:

    pip install zstandard

or

    conda install zstandard"""
        )
    else:
        return zstandard


def boost_histogram():
    """
    Imports and returns ``boost-histogram``.
    """
    try:
        import boost_histogram
    except ImportError:
        raise ImportError(
            """install the 'boost-histogram' package with:

    pip install boost-histogram

or

    conda install -c conda-forge boost-histogram"""
        )
    else:
        return boost_histogram


def hist():
    """
    Imports and returns ``hist``.
    """
    try:
        import hist
    except ImportError:
        raise ImportError(
            """install the 'hist' package with:

    pip install hist"""
        )
    else:
        return hist
