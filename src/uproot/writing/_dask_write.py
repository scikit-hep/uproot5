from __future__ import annotations

import math
from typing import Any

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs

import uproot


class _ToROOTFn:
    def __init__(
        self,
        fs: AbstractFileSystem,
        path: str,
        npartitions: int,
        prefix: str | None = None,
        storage_options: dict | None = None,
        **kwargs: Any,
    ):
        self.fs = fs
        self.path = path
        self.prefix = prefix
        self.zfill = math.ceil(math.log(npartitions, 10))
        self.storage_options = storage_options
        self.fs.mkdirs(self.path, exist_ok=True)
        self.protocol = (
            self.fs.protocol
            if isinstance(self.fs.protocol, str)
            else self.fs.protocol[0]
        )
        self.kwargs = kwargs

    def __call__(self, data, block_index):
        filename = f"part{str(block_index[0]).zfill(self.zfill)}.root"
        if self.prefix is not None:
            filename = f"{self.prefix}-{filename}"
        filename = self.fs.unstrip_protocol(f"{self.path}{self.fs.sep}{filename}")
        return ak_to_root(
            filename, data, **self.kwargs, storage_options=self.storage_options
        )


def dask_write(
    array,
    destination,
    compute=True,
    storage_options=None,
    prefix: str | None = None,
    tree_name="tree",
    title="",
    field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner,
    initial_basket_capacity=10,
    counter_name=lambda counted: "n" + counted,
    resize_factor=10.0,
    compression="zlib",
    compression_level=1,
):
    """
    Args:
        array (`dask_awkward.Array`): The :obj:`dask_awkward.Array` collection to write to disk.
        destination (path-like): Where to store the output; this can be a local filesystem path
            or a remote filesystem path.
        compute (bool): If ``True``, immediately compute the result (write data to disk). If ``False`` a Scalar collection will be returned such that ``compute`` can be explicitly called.
        prefix (str): An addition prefix for output files. If ``None`` all parts
            inside the destination directory will be named ``?``; if
            defined, the names will be ``f"{prefix}-partN.root"``.
        tree_name (str): Name of ttree to be written to. Default is "tree".
        title (str): Title of ttree to be written to. Default is "".
        field_name (callable of str \u2192 str): Function to generate TBranch
            names for columns of an Awkward record array or a Pandas DataFrame.
        initial_basket_capacity (int): Number of TBaskets that can be written to the
            TTree without rewriting the TTree metadata to make room.
        resize_factor (float): When the TTree metadata needs to be rewritten,
            this specifies how many more TBasket slots to allocate as a multiplicative
            factor.
        compression (:doc:`uproot.compression.Compression` or None): Compression algorithm
            and level for new objects added to the file. Can be updated after creating
            the :doc:`uproot.writing.writable.WritableFile`. Default is ``uproot.ZLIB(1)``.

    Writes a dask-awkward array to a set of ROOT files. Data is written to a TTree

    .. code-block:: python
        import awkward as ak
        import dask_awkward as dak
        a = ak.Array([{"a": [1,2,3]}, {"a": [4, 5]}])
        d = dask_write(a, npartitions=2)
        d.nparatitions
        uproot.dask_write(d)

    """
    from dask.base import tokenize
    from dask.blockwise import BlockIndex
    from dask.highlevelgraph import HighLevelGraph
    from dask_awkward.layers.layers import AwkwardMaterializedLayer
    from dask_awkward.lib.core import map_partitions, new_scalar_object

    fs, path = url_to_fs(destination, **(storage_options or {}))
    name = f"write-root-{tokenize(fs, array, destination)}"

    map_res = map_partitions(
        _ToROOTFn(
            fs=fs,
            path=path,
            npartitions=array.npartitions,
            prefix=prefix,
            tree_name=tree_name,
            compression=compression,
            compression_level=compression_level,
            title=title,
            field_name=field_name,
            counter_name=counter_name,
            resize_factor=resize_factor,
            initial_basket_capacity=initial_basket_capacity,
        ),
        array,
        BlockIndex((array.npartitions,)),
        label="to-root",
        meta=array._meta,
    )
    map_res.dask.layers[map_res.name].annotations = {"ak_output": True}
    dsk = {}
    final_name = name + "-finalize"
    dsk[(final_name, 0)] = (lambda *_: None, map_res.__dask_keys__())

    graph = HighLevelGraph.from_collections(
        final_name,
        AwkwardMaterializedLayer(dsk, previous_layer_names=[map_res.name]),
        dependencies=[map_res],
    )
    out = new_scalar_object(graph, final_name, dtype="f8")
    if compute:
        out.compute()
        return None
    else:
        return out


def ak_to_root(
    destination,
    array,
    tree_name,
    compression,
    compression_level,
    title,
    counter_name,
    field_name,
    initial_basket_capacity,
    resize_factor,
    storage_options,
):
    if compression in ("LZMA", "lzma"):
        compression_code = uproot.const.kLZMA
    elif compression in ("ZLIB", "zlib"):
        compression_code = uproot.const.kZLIB
    elif compression in ("LZ4", "lz4"):
        compression_code = uproot.const.kLZ4
    elif compression in ("ZSTD", "zstd"):
        compression_code = uproot.const.kZSTD
    else:
        msg = f"unrecognized compression algorithm: {compression}. Only ZLIB, LZMA, LZ4, and ZSTD are accepted."
        raise ValueError(msg)

    out_file = uproot.recreate(
        destination,
        compression=uproot.compression.Compression.from_code_pair(
            compression_code, compression_level
        ),
        **(storage_options or {}),
    )

    branch_types = {name: array[name].type for name in array.fields}

    out_file.mktree(
        tree_name,
        branch_types,
        title=title,
        counter_name=counter_name,
        field_name=field_name,
        initial_basket_capacity=initial_basket_capacity,
        resize_factor=resize_factor,
    )

    out_file[tree_name].extend({name: array[name] for name in array.fields})
