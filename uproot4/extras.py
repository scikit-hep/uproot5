# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import os


def awkward1():
    try:
        import awkward1
    except ImportError:
        raise ImportError(
            """install the 'awkward1' package with:

    pip install awkward1"""
        )
    else:
        return awkward1


def pandas():
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


def cupy():
    try:
        import cupy
    except ImportError:
        raise ImportError(
            """install the 'cupy' package with:

    pip install cupy

or

    conda install cupy"""
        )
    else:
        return cupy


def dask_array():
    try:
        import dask.array
    except ImportError:
        raise ImportError(
            """install the 'dask.array' package with:

    pip install "dask[array]"

or

    conda install dask"""
        )
    else:
        return dask.array


def dask_dataframe():
    try:
        import dask.dataframe
    except ImportError:
        raise ImportError(
            """install the 'dask.dataframe' package with:

    pip install "dask[dataframe]"

or

    conda install dask"""
        )
    else:
        return dask.dataframe


def pyxrootd_XRootD_client():
    os.environ["XRD_RUNFORKHANDLER"] = "1"  # set multiprocessing flag
    try:
        import pyxrootd
        import pyxrootd.client
        import XRootD
        import XRootD.client

    except ImportError:
        raise ImportError(
            """Install pyxrootd package with:

    conda install -c conda-forge xrootd

(or download from http://xrootd.org/dload.html and manually compile with """
            """cmake; setting PYTHONPATH and LD_LIBRARY_PATH appropriately)."""
        )

    else:
        return pyxrootd.client, XRootD.client


def lzma():
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


def lz4_block():
    try:
        import lz4.block
    except ImportError:
        raise ImportError(
            """install the 'lz4' package with (you probably also need 'xxhash'):

    pip install lz4 xxhash

or

    conda install lz4 python-xxhash"""
        )
    else:
        return lz4.block


def xxhash():
    try:
        import xxhash
    except ImportError:
        raise ImportError(
            """install the 'xxhash' package with (you probably also need 'lz4'):

    pip install xxhash lz4

or

    conda install python-xxhash lz4"""
        )
    else:
        return xxhash


def zstandard():
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
    try:
        import hist
    except ImportError:
        raise ImportError(
            """install the 'hist' package with:

    pip install hist"""
        )
    else:
        return hist
