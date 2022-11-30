# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines high-level functions and objects for file-writing.

The :doc:`uproot.writing.writable.create`, :doc:`uproot.writing.writable.recreate`, and :doc:`uproot.writing.writable.update`
functions open files for writing, overwriting, or updating, in a way that is similar
to :doc:`uproot.reading.open`.

The :doc:`uproot.writing.writable.WritableFile`, :doc:`uproot.writing.writable.WritableDirectory`,
:doc:`uproot.writing.writable.WritableTree`, and :doc:`uproot.writing.writable.WritableBranch`
classes are writable versions of :doc:`uproot.reading.ReadOnlyFile`, :doc:`uproot.reading.ReadOnlyDirectory`,
:doc:`uproot.behaviors.TTree.TTree`, and :doc:`uproot.behaviors.TBranch.TBranch`.

There is no feature parity between writable and readable versions of each of these
types. Writing and reading are considered separate projects with different capabilities.
"""


import datetime
import itertools
import os
import queue
import sys
import uuid
from collections.abc import Mapping, MutableMapping

import uproot._util
import uproot.compression
import uproot.deserialization
import uproot.exceptions
import uproot.model
import uproot.models.TObjString
import uproot.sink.file
import uproot.writing._cascade
from uproot._util import no_filter, no_rename


def create(file_path, **options):
    """
    Args:
        file_path (str, ``pathlib.Path`` or file-like object): The filesystem path of the
            file to open or an open file.
        compression (:doc:`uproot.compression.Compression` or None): Compression algorithm
            and level for new objects added to the file. Can be updated after creating
            the :doc:`uproot.writing.writable.WritableFile`. Default is ``uproot.ZLIB(1)``.
        options: See below.

    Opens a local file for writing. Like ROOT's ``"CREATE"`` option, this function
    raises an error (``OSError``) if a file already exists at ``file_path``.

    Returns a :doc:`uproot.writing.writable.WritableDirectory`.

    Options (type; default):

    * initial_directory_bytes (int; 256)
    * initial_streamers_bytes (int; 1024)
    * uuid_function (callable; ``uuid.uuid1``)

    See :doc:`uproot.writing.writable.WritableFile` for details on these options.
    """
    file_path = uproot._util.regularize_path(file_path)
    if uproot._util.isstr(file_path):
        if os.path.exists(file_path):
            raise OSError(
                "path exists and refusing to overwrite (use 'uproot.recreate' to "
                "overwrite)\n\nfor path {}".format(file_path)
            )
    return recreate(file_path, **options)


def recreate(file_path, **options):
    """
    Args:
        file_path (str, ``pathlib.Path`` or file-like object): The filesystem path of the
            file to open or an open file.
        compression (:doc:`uproot.compression.Compression` or None): Compression algorithm
            and level for new objects added to the file. Can be updated after creating
            the :doc:`uproot.writing.writable.WritableFile`. Default is ``uproot.ZLIB(1)``.
        options: See below.

    Opens a local file for writing. Like ROOT's ``"RECREATE"`` option, this function
    overwrites any file that already exists at ``file_path``.

    Returns a :doc:`uproot.writing.writable.WritableDirectory`.

    Options (type; default):

    * initial_directory_bytes (int; 256)
    * initial_streamers_bytes (int; 1024)
    * uuid_function (callable; ``uuid.uuid1``)

    See :doc:`uproot.writing.writable.WritableFile` for details on these options.
    """
    file_path = uproot._util.regularize_path(file_path)
    if uproot._util.isstr(file_path):
        # Truncate file
        open(file_path, "w").close()
        sink = uproot.sink.file.FileSink(file_path)
    else:
        sink = uproot.sink.file.FileSink.from_object(file_path)

    compression = options.pop("compression", create.defaults["compression"])

    initial_directory_bytes = options.pop(
        "initial_directory_bytes", create.defaults["initial_directory_bytes"]
    )
    initial_streamers_bytes = options.pop(
        "initial_streamers_bytes", create.defaults["initial_streamers_bytes"]
    )
    uuid_function = options.pop("uuid_function", create.defaults["uuid_function"])
    if options:
        raise TypeError(
            "unrecognized options for uproot.create or uproot.recreate: "
            + ", ".join(repr(x) for x in options)
        )

    cascading = uproot.writing._cascade.create_empty(
        sink,
        compression,
        initial_directory_bytes,
        initial_streamers_bytes,
        uuid_function,
    )
    return WritableFile(
        sink, cascading, initial_directory_bytes, uuid_function
    ).root_directory


def update(file_path, **options):
    """
    Args:
        file_path (str, ``pathlib.Path`` or file-like object): The filesystem path of the
            file to open or an open file.
        options: See below.

    Opens a local file for writing. Like ROOT's ``"UPDATE"`` option, this function
    expects a file to already exist at ``file_path`` and opens it so that new data
    can be added to it or individual objects may be deleted from it.

    Returns a :doc:`uproot.writing.writable.WritableDirectory`.

    Options (type; default):

    * initial_directory_bytes (int; 256)
    * uuid_function (callable; ``uuid.uuid1``)

    See :doc:`uproot.writing.writable.WritableFile` for details on these options.
    """
    file_path = uproot._util.regularize_path(file_path)
    if uproot._util.isstr(file_path):
        sink = uproot.sink.file.FileSink(file_path)
    else:
        sink = uproot.sink.file.FileSink.from_object(file_path)

    initial_directory_bytes = options.pop(
        "initial_directory_bytes", create.defaults["initial_directory_bytes"]
    )
    uuid_function = options.pop("uuid_function", create.defaults["uuid_function"])
    if options:
        raise TypeError(
            "unrecognized options for uproot.update: "
            + ", ".join(repr(x) for x in options)
        )

    cascading = uproot.writing._cascade.update_existing(
        sink,
        initial_directory_bytes,
        uuid_function,
    )
    return WritableFile(
        sink, cascading, initial_directory_bytes, uuid_function
    ).root_directory


create.defaults = {
    "compression": uproot.compression.ZLIB(1),
    "initial_directory_bytes": 256,
    "initial_streamers_bytes": 1024,  # 256,
    "uuid_function": uuid.uuid1,
}
recreate.defaults = create.defaults
update.defaults = create.defaults


class WritableFile(uproot.reading.CommonFileMethods):
    """
    Args:
        sink (:doc:`uproot.sink.file.FileSink`): The physical layer for file-writing.
        cascading (:doc:`uproot.writing._cascade.CascadingFile`): The low-level file
            object.
        initial_directory_bytes (int): Number of bytes to allocate for new directories,
            so that TKeys can be added to them without immediately needing to rewrite
            the block.
        uuid_function (zero-argument callable returning a ``uuid.UUID``): Function to
            create the file's UUID and/or any directory's UUID.

    Handle to a writable ROOT file, usually created by :doc:`uproot.writing.writable.create`,
    :doc:`uproot.writing.writable.recreate`, or :doc:`uproot.writing.writable.update` and
    accessed through a :doc:`uproot.writing.writable.WritableDirectory`.
    """

    def __init__(self, sink, cascading, initial_directory_bytes, uuid_function):
        self._sink = sink
        self._cascading = cascading
        self._initial_directory_bytes = initial_directory_bytes
        self._uuid_function = uuid_function

        self._file_path = sink.file_path
        self._fVersion = self._cascading.fileheader.version
        self._fBEGIN = self._cascading.fileheader.begin
        self._fNbytesName = self._cascading.fileheader.begin_num_bytes
        self._fUUID = self._cascading.fileheader.uuid.bytes

        self._trees = {}

    def __repr__(self):
        return f"<WritableFile {self.file_path!r} at 0x{id(self):012x}>"

    @property
    def sink(self):
        """
        Returns a :doc:`uproot.sink.file.FileSink`, the physical layer for writing
        (and sometimes reading) data.
        """
        return self._sink

    @property
    def initial_directory_bytes(self):
        """
        Number of bytes to allocate for new directories, so that TKeys can be added
        to them without immediately needing to rewrite the block.
        """
        return self._initial_directory_bytes

    @initial_directory_bytes.setter
    def initial_directory_bytes(self, value):
        self._initial_directory_bytes = value

    @property
    def uuid_function(self):
        """
        The function used to create the file's UUID and/or any directory's UUID.
        """
        return self._uuid_function

    @uuid_function.setter
    def uuid_function(self, value):
        self._uuid_function = value

    @property
    def options(self):
        """
        The options passed to :doc:`uproot.writing.writable.create`,
        :doc:`uproot.writing.writable.recreate`, or :doc:`uproot.writing.writable.update`
        when opening this file.
        """
        return {
            "initial_directory_bytes": self._initial_directory_bytes,
            "uuid_function": self._uuid_function,
        }

    @property
    def is_64bit(self):
        """
        True if the file has 8-byte pointers in its header; False if the pointers are 4-byte.
        """
        return self._cascading.fileheader.big

    @property
    def compression(self):
        """
        Compression algorithm and level (:doc:`uproot.compression.Compression` or None)
        for new objects added to the file.

        This property can be changed, which allows you to write different objects
        with different compression settings.

        See also :ref:`uproot.writing.writable.WritableFile.fCompress`.
        """
        return self._cascading.fileheader.compression

    @compression.setter
    def compression(self, value):
        if value is None or isinstance(value, uproot.compression.Compression):
            self._cascading.fileheader.compression = value
            self._cascading.fileheader.write(self._sink)
            self._sink.flush()
        else:
            raise TypeError(
                "compression must be None or a uproot.compression.Compression object, like uproot.ZLIB(4) or uproot.ZSTD(0)"
            )

    @property
    def fSeekFree(self):
        """
        The seek point (int) to the ``TFree`` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._cascading.fileheader.free_location

    @property
    def fNbytesFree(self):
        """
        The number of bytes in the ``TFree`` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._cascading.fileheader.free_num_bytes

    @property
    def nfree(self):
        """
        The number of objects in the ``TFree`` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._cascading.fileheader.free_num_slices + 1

    @property
    def fUnits(self):
        """
        Number of bytes in the serialization of file seek points, which can either
        be 4 or 8.
        """
        return 8 if self._cascading.fileheader.big else 4

    @property
    def fCompress(self):
        """
        Compression algorithm and level (as an integer code) for new objects added
        to the file.

        This property can be changed, which allows you to write different objects
        with different compression settings.

        See also :ref:`uproot.writing.writable.WritableFile.compression`.
        """
        if self._cascading.fileheader.compression is None:
            return uproot.compression.ZLIB(0).code
        else:
            return self._cascading.fileheader.compression.code

    @property
    def fSeekInfo(self):
        """
        The seek point (int) to the ``TStreamerInfo`` data, where
        TStreamerInfo records are located.
        """
        return self._cascading.fileheader.info_location

    @property
    def fNbytesInfo(self):
        """
        The number of bytes in the ``TStreamerInfo`` data, where
        TStreamerInfo records are located.
        """
        return self._cascading.fileheader.info_num_bytes

    @property
    def uuid(self):
        """
        The unique identifier (UUID) of the ROOT file expressed as a Python
        ``uuid.UUID`` object.
        """
        return self._cascading.fileheader.uuid

    @property
    def root_directory(self):
        """
        The root (first) directory in the file as a :doc:`uproot.writing.writable.WritableDirectory`.
        """
        return WritableDirectory((), self, self._cascading.rootdirectory)

    def update_streamers(self, streamers):
        """
        Overwrite the TStreamerInfo in this file with a new list of :doc:`uproot.streamers.Model_TStreamerInfo`
        or :doc:`uproot.writable._cascade.RawStreamerInfo`.
        """
        self._cascading.streamers.update_streamers(self.sink, streamers)

    @property
    def file_path(self):
        """
        Filesystem path of the open file, or None if using a file-like object.
        """
        return self._file_path

    def close(self):
        """
        Explicitly close the file.

        (Files can also be closed with the Python ``with`` statement, as context
        managers.)

        After closing, objects cannot be read from or written to the file.
        """
        self._sink.close()

    @property
    def closed(self):
        """
        True if the file has been closed; False otherwise.

        The file may have been closed explicitly with
        :ref:`uproot.writing.writable.WritableFile.close` or implicitly in the Python
        ``with`` statement, as a context manager.

        After closing, objects cannot be read from or written to the file.
        """
        return self._sink.closed

    def __enter__(self):
        self._sink.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._sink.__exit__(exception_type, exception_value, traceback)

    def _new_tree(self, tree):
        self._trees[tree._cascading.key.seek_location] = tree

    def _has_tree(self, loc):
        return loc in self._trees

    def _get_tree(self, loc):
        return self._trees[loc]

    def _move_tree(self, oldloc, newloc):
        tree = self._trees[oldloc]
        del self._trees[oldloc]
        self._trees[newloc] = tree


class WritableDirectory(MutableMapping):
    """
    Args:
        path (tuple of str): Path of directory names to this subdirectory; ``()`` for
            the root (first) directory.
        file (:doc:`uproot.writing.writable.WritableFile`): Handle to the file in
            which this directory can be found.
        cascading (:doc:`uproot.writing._cascade.CascadingDirectory`): The low-level
            directory object.

    Represents a writable ``TDirectory`` from a ROOT file.

    Be careful not to confuse :doc:`uproot.writing.writable.WritableFile` and
    :doc:`uproot.writing.writable.WritableDirectory`: files are for modifying global
    information such as the TStreamerInfo and FreeSegments, whereas directories
    are for data in local hierarchies.

    A :doc:`uproot.writing.writable.WritableDirectory` is a Python ``MutableMapping``,
    which uses square bracket syntax to read, write, and delete objects:

    .. code-block:: python

        my_directory["histogram"]
        my_directory["histogram"] = np.histogram(...)
        del my_directory["histogram"]

    Objects in ROOT files also have "cycle numbers," which allow multiple versions
    of an object to exist with the same name. A cycle number may be specified after
    a semicolon for *reading* and *deleting* only:

    .. code-block:: python

        my_directory["histogram;2"]
        del my_directory["histogram;2"]

    When *writing*, cycle numbers are generated to avoid overwriting previous objects:

    .. code-block:: python

        my_directory["histogram"] = np.histogram(...)   # creates a new histogram
        my_directory["histogram"] = np.histogram(...)   # creates another histogram

    Note that this is unlike a Python ``MutableMapping``, which would overwrite the
    object in the second assignment. However, it is the way ROOT I/O works; use ``del``
    to remove unwanted versions of objects.

    Any types of objects that can be read from a :doc:`uproot.reading.ReadOnlyDirectory`
    can be read from a :doc:`uproot.writing.writable.WritableDirectory` *except TTrees*. A
    TTree can only be read from a :doc:`uproot.reading.ReadOnlyDirectory` if it was
    created in this open file handle, and then it returns a :doc:`uproot.writing.writable.WritableTree`
    instead of the :doc:`uproot.behaviors.TTree.TTree` that you would get from a
    :doc:`uproot.reading.ReadOnlyDirectory`. Readable TTrees and writable TTrees are
    distinct, with separate sets of features.

    Note that subdirectories can be created by assigning to path names that include
    slashes:

    .. code-block:: python

        my_directory["subdir1/subdir2/new_object"] = new_object

    Subdirectories created this way will never be empty; to make an empty directory,
    use :ref:`uproot.writing.writable.WritableDirectory.mkdir`.

    Similarly, non-empty TTrees can be created by assignment (see :doc:`uproot.writing.writable.WritableTree`
    for recognized TTree-like data), but empty TTrees require the
    :ref:`uproot.writing.writable.WritableDirectory.mktree` method.
    """

    def __init__(self, path, file, cascading):
        self._path = path
        self._file = file
        self._cascading = cascading
        self._subdirs = {}

    def __repr__(self):
        return "<WritableDirectory {} at 0x{:012x}>".format(
            repr("/" + "/".join(self._path)), id(self)
        )

    @property
    def path(self):
        """
        Path of directory names to this subdirectory as a tuple of strings; e.g. ``()``
        for the root (first) directory.
        """
        return self._path

    @property
    def object_path(self):
        """
        Path of directory names to this subdirectory as a single string, delimited
        by slashes.
        """
        return "/".join(("",) + self._path + ("",)).replace("//", "/")

    @property
    def file_path(self):
        """
        Filesystem path of the open file, or None if using a file-like object.
        """
        return self._file.file_path

    @property
    def file(self):
        """
        Handle to the :doc:`uproot.writing.writable.WritableDirectory` in which
        this directory can be found.
        """
        return self._file

    def close(self):
        """
        Explicitly close the file.

        (Files can also be closed with the Python ``with`` statement, as context
        managers.)

        After closing, objects cannot be read from or written to the file.
        """
        self._file.close()

    @property
    def closed(self):
        """
        True if the file has been closed; False otherwise.

        The file may have been closed explicitly with
        :ref:`uproot.writing.writable.WritableFile.close` or implicitly in the Python
        ``with`` statement, as a context manager.

        After closing, objects cannot be read from or written to the file.
        """
        return self._file.closed

    def __enter__(self):
        self._file.sink.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.sink.__exit__(exception_type, exception_value, traceback)

    @property
    def compression(self):
        """
        Compression algorithm and level (:doc:`uproot.compression.Compression` or None)
        for new objects added to the file.

        This property can be changed, which allows you to write different objects
        with different compression settings.
        """
        return self._file.compression

    @compression.setter
    def compression(self, value):
        self._file.compression = value

    def __len__(self):
        return self._cascading.data.num_keys + sum(
            len(self._subdir(x)) for x in self._cascading.data.dir_names
        )

    def __contains__(self, where):
        if self._cascading.data.haskey(where):
            return True
        for x in self._cascading.data.dir_names:
            if where in self._subdir(x):
                return True
        return False

    def __iter__(self):
        return self.iterkeys()

    def _ipython_key_completions_(self):
        """
        Supports key-completion in an IPython or Jupyter kernel.
        """
        return self.iterkeys()

    def keys(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return the names of objects directly accessible
                in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in those names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns the names of the objects in this ``TDirectory`` as a list of
        strings.

        Note that this does not read any data from the file.
        """
        return list(
            self.iterkeys(
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def values(
        self,
        *,
        recursive=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return objects directly accessible in this
                ``TDirectory``.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns objects in this ``TDirectory`` as a list of
        :doc:`uproot.model.Model`.

        Note that this reads all objects that are selected by ``filter_name``
        and ``filter_classname``.
        """
        return list(
            self.itervalues(
                recursive=recursive,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def items(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return (name, object) pairs directly accessible
                in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in the names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns (name, object) pairs for objects in this ``TDirectory`` as a
        list of 2-tuples of (str, :doc:`uproot.model.Model`).

        Note that this reads all objects that are selected by ``filter_name``
        and ``filter_classname``.
        """
        return list(
            self.iteritems(
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def classnames(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return the names and classnames of objects
                directly accessible in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in the names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns the names and C++ (decoded) classnames of the objects in this
        ``TDirectory`` as a dict of str \u2192 str.

        Note that this does not read any data from the file.
        """
        return dict(
            self.iterclassnames(
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def iterkeys(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return the names of objects directly accessible
                in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in those names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns the names of the objects in this ``TDirectory`` as an iterator
        over strings.

        Note that this does not read any data from the file.
        """
        filter_name = uproot._util.regularize_filter(filter_name)
        filter_classname = uproot._util.regularize_filter(filter_classname)
        for keyname, cyclenum, classname in self._cascading.data.key_triples:
            if (filter_name is no_filter or filter_name(keyname)) and (
                filter_classname is no_filter or filter_classname(classname)
            ):
                if cycle:
                    yield f"{keyname};{cyclenum}"
                else:
                    yield keyname

            if recursive and classname in ("TDirectory", "TDirectoryFile"):
                for k1 in self._get(keyname, cyclenum).iterkeys(
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=filter_name,
                    filter_classname=filter_classname,
                ):
                    k2 = f"{keyname}/{k1}"
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2

    def itervalues(
        self,
        *,
        recursive=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return objects directly accessible in this
                ``TDirectory``.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns objects in this ``TDirectory`` as an iterator over
        :doc:`uproot.model.Model`.

        Note that this reads all objects that are selected by ``filter_name``
        and ``filter_classname``.
        """
        for keyname in self.iterkeys(
            recursive=recursive,
            cycle=True,
            filter_name=filter_name,
            filter_classname=filter_classname,
        ):
            yield self[keyname]

    def iteritems(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return (name, object) pairs directly accessible
                in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in the names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns (name, object) pairs for objects in this ``TDirectory`` as an
        iterator over 2-tuples of (str, :doc:`uproot.model.Model`).

        Note that this reads all objects that are selected by ``filter_name``
        and ``filter_classname``.
        """
        for keyname in self.iterkeys(
            recursive=recursive,
            cycle=True,
            filter_name=filter_name,
            filter_classname=filter_classname,
        ):
            if not cycle:
                at = keyname.index(";")
                keyname = keyname[:at]
            yield keyname, self[keyname]

    def iterclassnames(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return the names and classnames of objects
                directly accessible in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in the names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns the names and C++ (decoded) classnames of the objects in this
        ``TDirectory`` as an iterator of 2-tuples of (str, str).

        Note that this does not read any data from the file.
        """
        filter_name = uproot._util.regularize_filter(filter_name)
        filter_classname = uproot._util.regularize_filter(filter_classname)
        for keyname, cyclenum, classname in self._cascading.data.key_triples:
            if (filter_name is no_filter or filter_name(keyname)) and (
                filter_classname is no_filter or filter_classname(classname)
            ):
                if cycle:
                    yield f"{keyname};{cyclenum}", classname
                else:
                    yield keyname, classname

            if recursive and classname in ("TDirectory", "TDirectoryFile"):
                for k1, c1 in self._get(keyname, cyclenum).iterclassnames(
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=filter_name,
                    filter_classname=filter_classname,
                ):
                    k2 = f"{keyname}/{k1}"
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2, c1

    def _get_del_search(self, where, isget):
        if "/" in where or ":" in where:
            items = where.split("/")
            step = last = self

            for i, item in enumerate(items):
                if item != "":
                    if isinstance(step, WritableDirectory):
                        if ":" in item and not step._cascading.data.haskey(item):
                            raise uproot.KeyInFileError(
                                where,
                                because="TTrees in writable files can't be indexed by TBranch name",
                                file_path=self.file_path,
                            )
                        else:
                            last = step
                            step = step[item]

                    elif isinstance(step, WritableTree):
                        rest = items[i:]
                        if len(rest) != 0:
                            raise uproot.KeyInFileError(
                                where,
                                because="TTrees in writable files can't be indexed by TBranch name",
                                file_path=self.file_path,
                            )
                        return step

                    else:
                        raise uproot.KeyInFileError(
                            where,
                            because="/".join(items[:i]) + " is not a TDirectory",
                            keys=last._cascading.data.key_names,
                            file_path=self.file_path,
                        )

            return step

        else:
            if ";" in where:
                at = where.rindex(";")
                item, cycle = where[:at], where[at + 1 :]
                try:
                    cycle = int(cycle)
                except ValueError:
                    item, cycle = where, None
            else:
                item, cycle = where, None

            if isget:
                return self._get(item, cycle)
            else:
                return self._del(item, cycle)

    def __getitem__(self, where):
        if self._file.sink.closed:
            raise ValueError("cannot get data from a closed file")
        return self._get_del_search(where, True)

    def __setitem__(self, where, what):
        if self._file.sink.closed:
            raise ValueError("cannot write data to a closed file")
        self.update({where: what})

    def __delitem__(self, where):
        if self._file.sink.closed:
            raise ValueError("cannot delete data from a closed file")
        return self._get_del_search(where, False)

    def _get(self, name, cycle):
        key = self._cascading.data.get_key(name, cycle)
        if key is None:
            raise uproot.exceptions.KeyInFileError(
                name,
                cycle="any" if cycle is None else cycle,
                keys=self._cascading.data.key_names,
                file_path=self.file_path,
                object_path=self.object_path,
            )

        if key.classname.string in ("TDirectory", "TDirectoryFile"):
            return self._subdir(key)

        elif key.classname.string == "TTree":
            if self._file._has_tree(key.seek_location):
                return self._file._get_tree(key.seek_location)
            else:
                raise TypeError(
                    "WritableDirectory cannot view preexisting TTrees; open the file with uproot.open instead of uproot.recreate or uproot.update"
                )

        else:

            def get_chunk(start, stop):
                raw_bytes = self._file.sink.read(start, stop - start)
                return uproot.source.chunk.Chunk.wrap(
                    readforupdate, raw_bytes, start=start
                )

            readforupdate = uproot.writing._cascade._ReadForUpdate(
                self._file.file_path,
                self._file.uuid,
                get_chunk,
                self._file._cascading.tlist_of_streamers,
            )

            raw_bytes = self._file.sink.read(
                key.seek_location,
                key.num_bytes + key.compressed_bytes,
            )

            chunk = uproot.source.chunk.Chunk.wrap(readforupdate, raw_bytes)
            cursor = uproot.source.cursor.Cursor(0, origin=key.num_bytes)

            readonlykey = uproot.reading.ReadOnlyKey(
                chunk, cursor, {}, readforupdate, self, read_strings=True
            )

            return readonlykey.get()

    def _del(self, name, cycle):
        key = self._cascading.data.get_key(name, cycle)
        start = key.location
        stop = start + key.num_bytes + key.compressed_bytes
        self._cascading.freesegments.release(start, stop)

        self._cascading._data.remove_key(key)
        self._cascading.header.modified_on = datetime.datetime.now()

        self._cascading.write(self._file.sink)
        self._file.sink.set_file_length(self._cascading.freesegments.fileheader.end)
        self._file.sink.flush()

    def _subdir(self, key):
        name = key.name.string

        if name in self._subdirs:
            sub = self._subdirs[name]
            for tree in self._file._trees.values():
                if (
                    tree._cascading.directory.key.location
                    == sub._cascading.key.location
                    and tree._cascading.directory is not sub._cascading
                ):
                    self._subdirs[name] = WritableDirectory(
                        self._path + (name,), self._file, tree._cascading.directory
                    )
                    break

        if name not in self._subdirs:
            raw_bytes = self._file.sink.read(
                key.seek_location,
                key.num_bytes + uproot.reading._directory_format_big.size + 18,
            )
            directory_key = uproot.writing._cascade.Key.deserialize(
                raw_bytes, key.seek_location, self._file.sink.in_path
            )
            position = key.seek_location + directory_key.num_bytes

            directory_header = uproot.writing._cascade.DirectoryHeader.deserialize(
                raw_bytes[position - key.seek_location :],
                position,
                self._file.sink.in_path,
            )
            assert directory_header.begin_location == key.seek_location

            # # FIXME: why was this here?
            # assert (
            #     directory_header.parent_location
            #     == self._file._cascading.fileheader.begin
            # )

            if directory_header.data_num_bytes == 0:
                directory_datakey = uproot.writing._cascade.Key(
                    None,
                    None,
                    None,
                    uproot.writing._cascade.String(None, "TDirectory"),
                    uproot.writing._cascade.String(None, name),
                    uproot.writing._cascade.String(None, name),
                    directory_key.cycle,
                    directory_header.parent_location,
                    None,
                )

                requested_num_bytes = (
                    directory_datakey.num_bytes + self._file._initial_directory_bytes
                )
                directory_datakey.location = self._cascading.freesegments.allocate(
                    requested_num_bytes
                )
                might_be_slightly_more = (
                    requested_num_bytes - directory_datakey.num_bytes
                )
                directory_data = uproot.writing._cascade.DirectoryData(
                    directory_datakey.location + directory_datakey.num_bytes,
                    might_be_slightly_more,
                    [],
                )

                directory_datakey.uncompressed_bytes = directory_data.allocation
                directory_datakey.compressed_bytes = (
                    directory_datakey.uncompressed_bytes
                )

                subdirectory = uproot.writing._cascade.SubDirectory(
                    directory_key,
                    directory_header,
                    directory_datakey,
                    directory_data,
                    self._cascading,
                    self._cascading.freesegments,
                )

                directory_header.data_location = directory_datakey.location
                directory_header.data_num_bytes = (
                    directory_datakey.num_bytes + directory_data.allocation
                )

                subdirectory.write(self._file.sink)

                self._file.sink.set_file_length(
                    self._cascading.freesegments.fileheader.end
                )
                self._file.sink.flush()

                self._subdirs[name] = WritableDirectory(
                    self._path + (name,), self._file, subdirectory
                )

            else:
                raw_bytes = self._file.sink.read(
                    directory_header.data_location, directory_header.data_num_bytes
                )

                directory_datakey = uproot.writing._cascade.Key.deserialize(
                    raw_bytes, directory_header.data_location, self._file.sink.in_path
                )
                directory_data = uproot.writing._cascade.DirectoryData.deserialize(
                    raw_bytes[directory_datakey.num_bytes :],
                    directory_header.data_location + directory_datakey.num_bytes,
                    self._file.sink.in_path,
                )

                subdirectory = uproot.writing._cascade.SubDirectory(
                    directory_key,
                    directory_header,
                    directory_datakey,
                    directory_data,
                    self._cascading,
                    self._cascading.freesegments,
                )

                self._subdirs[name] = WritableDirectory(
                    self._path + (name,), self._file, subdirectory
                )

        return self._subdirs[name]

    def mkdir(self, name, *, initial_directory_bytes=None):
        """
        Args:
            name (str): Name of the new subdirectory.
            initial_directory_bytes (None or int): Number of bytes to allocate
                for the new directory, so that TKeys can be added to it without
                immediately needing to rewrite the block. If None, the
                :doc:`uproot.writing.writable.WritableFile`'s value is used.

        Creates an empty subdirectory in this directory.

        Note that subdirectories can be created by assigning to path names that
        include slashes:

        .. code-block:: python

            my_directory["subdir1/subdir2/new_object"] = new_object

        but subdirectories created this way will never be empty. Use this method
        to make an empty directory or to control directory parameters.
        """
        if self._file.sink.closed:
            raise ValueError("cannot create a TDirectory in a closed file")

        stripped = name.strip("/")
        try:
            at = stripped.index("/")
        except ValueError:
            head, tail = stripped, None
        else:
            head, tail = stripped[:at], stripped[at + 1 :]

        key = self._cascading.data.get_key(head)
        if key is None:
            if initial_directory_bytes is None:
                initial_directory_bytes = self._file.initial_directory_bytes
            directory = WritableDirectory(
                self._path + (head,),
                self._file,
                self._cascading.add_directory(
                    self._file.sink,
                    head,
                    initial_directory_bytes,
                    self._file.uuid_function(),
                ),
            )

        elif key.classname.string not in ("TDirectory", "TDirectoryFile"):
            raise TypeError(
                """cannot make a directory named {} because a {} already has that name
in file {} in directory {}""".format(
                    repr(name), key.classname.string, self.file_path, self.path
                )
            )

        else:
            directory = self._subdir(key)

        if tail is None:
            return directory

        else:
            return directory.mkdir(tail)

    def mktree(
        self,
        name,
        branch_types,
        title="",
        *,
        counter_name=lambda counted: "n" + counted,
        field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner,
        initial_basket_capacity=10,
        resize_factor=10.0,
    ):
        """
        Args:
            name (str): Name of the new TTree.
            branch_types (dict or pairs of str \u2192 NumPy dtype/Awkward type): Name
                and type specification for the TBranches.
            title (str): Title for the new TTree.
            counter_name (callable of str \u2192 str): Function to generate counter-TBranch
                names for Awkward Arrays of variable-length lists.
            field_name (callable of str \u2192 str): Function to generate TBranch
                names for columns of an Awkward record array or a Pandas DataFrame.
            initial_basket_capacity (int): Number of TBaskets that can be written to the
                TTree without rewriting the TTree metadata to make room.
            resize_factor (float): When the TTree metadata needs to be rewritten,
                this specifies how many more TBasket slots to allocate as a multiplicative
                factor.

        Creates an empty TTree in this directory.

        Note that TTrees can be created by assigning TTree-like data to a directory
        (see :doc:`uproot.writing.writable.WritableTree` for recognized TTree-like types):

        .. code-block:: python

            my_directory["tree"] = {"branch1": np.array(...), "branch2": ak.Array(...)}

        but TTrees created this way will never be empty. Use this method
        to make an empty TTree or to control its parameters.
        """
        if self._file.sink.closed:
            raise ValueError("cannot create a TTree in a closed file")

        try:
            at = name.rindex("/")
        except ValueError:
            treename = name
            directory = self
        else:
            dirpath, treename = name[:at], name[at + 1 :]
            directory = self.mkdir(dirpath)

        path = directory._path + (treename,)

        tree = WritableTree(
            path,
            directory._file,
            directory._cascading.add_tree(
                directory._file.sink,
                treename,
                title,
                branch_types,
                counter_name,
                field_name,
                initial_basket_capacity,
                resize_factor,
            ),
        )
        directory._file._new_tree(tree)

        seen = set()
        streamers = []
        for model in (
            uproot.models.TLeaf.Model_TLeafB_v1,
            uproot.models.TLeaf.Model_TLeafS_v1,
            uproot.models.TLeaf.Model_TLeafI_v1,
            uproot.models.TLeaf.Model_TLeafL_v1,
            uproot.models.TLeaf.Model_TLeafF_v1,
            uproot.models.TLeaf.Model_TLeafD_v1,
            uproot.models.TLeaf.Model_TLeafC_v1,
            uproot.models.TLeaf.Model_TLeafO_v1,
            uproot.models.TBranch.Model_TBranch_v13,
            uproot.models.TTree.Model_TTree_v20,
        ):
            for rawstreamer in model.class_rawstreamers:
                classname_version = rawstreamer[-2], rawstreamer[-1]
                if classname_version not in seen:
                    seen.add(classname_version)
                    streamers.append(
                        uproot.writing._cascade.RawStreamerInfo(*rawstreamer)
                    )

        directory._file._cascading.streamers.update_streamers(
            directory._file.sink, streamers
        )

        return tree

    def copy_from(
        self,
        source,
        *,
        filter_name=no_filter,
        filter_classname=no_filter,
        rename=no_rename,
        require_matches=True,
    ):
        """
        Args:
            source (:doc:`uproot.writing.writable.WritableDirectory` or :doc:`uproot.reading.ReadOnlyDirectory`): Directory from which to copy.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.
            rename (None, regex string in ``"/from/to/"`` syntax, dict of str \u2192 str, function of str \u2192 str, or iterable of the above): A
                function to convert old names into new names.
            require_matches (bool): If True and the filters do not match any data, raise
                a ``ValueError``.

        Bulk-copy method to copy data from one ROOT file to another without interpretation
        or even decompression/recompression.

        This method will likely have performance advantages over copying objects one
        at a time, in part because it avoids interpretation and decompression/recompression,
        and also because it collects TStreamerInfo from all of the data types and
        rewrites the output file's TStreamerInfo exactly once.
        """
        if isinstance(source, WritableDirectory):
            raise NotImplementedError(
                "copying from a WritableDirectory is not yet supported; open the "
                "'source' as a ReadOnlyDirectory (with uproot.open)"
            )
        elif not isinstance(source, uproot.reading.ReadOnlyDirectory):
            raise TypeError("'source' must be a TDirectory")

        old_names = source.keys(
            filter_name=filter_name, filter_classname=filter_classname, cycle=False
        )
        if len(old_names) == 0:
            if require_matches:
                raise ValueError(
                    """no objects found with names matching {}
in file {} in directory {}""".format(
                        repr(filter_name), source.file_path, source.path
                    )
                )
            else:
                return

        keys = [source.key(x) for x in old_names]

        for key in keys:
            if key.fClassName == "TTree" or key.fClassName.split("::")[-1] == "RNTuple":
                raise NotImplementedError(
                    f"copy_from cannot copy {key.fClassName} objects yet"
                )

        rename = uproot._util.regularize_rename(rename)
        new_names = [rename(x) for x in old_names]

        notifications = queue.Queue()
        ranges = {}
        for new_name, old_key in zip(new_names, keys):
            if old_key.fClassName not in ("TDirectory", "TDirectoryFile"):
                start = old_key.data_cursor.index
                stop = start + old_key.data_compressed_bytes
                ranges[start, stop] = new_name, old_key

        source.file.source.chunks(list(ranges), notifications=notifications)

        classversion_pairs = set()
        for classname in {x.fClassName for x in keys}:
            for streamer in source.file.streamers_named(classname):
                batch = []
                streamer._dependencies(source.file.streamers, batch)
                classversion_pairs.update(batch)

        streamers = [source.file.streamer_named(c, v) for c, v in classversion_pairs]

        self._file._cascading.streamers.update_streamers(self._file.sink, streamers)

        new_dirs = {}
        for new_name, old_key in zip(new_names, keys):
            classname = old_key.fClassName
            path = new_name.strip("/").split("/")
            if classname not in ("TDirectory", "TDirectoryFile"):
                path = path[:-1]
            path = "/".join(path)
            if path not in new_dirs:
                new_dirs[path] = 4
            new_dirs[path] += (
                uproot.reading._key_format_big.size
                + 5
                + len(old_key.fClassName)
                + 5
                + len(old_key.fName)
                + 5
                + len(old_key.fTitle)
            )

        for name, allocation in new_dirs.items():
            self.mkdir(
                name,
                initial_directory_bytes=max(
                    self._file.initial_directory_bytes, allocation
                ),
            )

        for _ in range(len(ranges)):
            chunk = notifications.get()
            assert isinstance(chunk, uproot.source.chunk.Chunk)

            raw_data = uproot._util.tobytes(chunk.raw_data)

            new_name, old_key = ranges[chunk.start, chunk.stop]
            path = new_name.strip("/").split("/")
            directory = self
            for item in path[:-1]:
                directory = directory[item]

            directory._cascading.add_object(
                self._file.sink,
                old_key.fClassName,
                path[-1],
                old_key.fTitle,
                raw_data,
                old_key.data_uncompressed_bytes,
            )

    def update(self, pairs=None, **more_pairs):
        """
        Args:
            pairs (dict or pairs of str \u2192 writable data): Names and data to write.
            more_pairs (dict or pairs of str \u2192 writable data): More names and data to write.

        Bulk-update function, like assignment, but it collects TStreamerInfo for a single
        update.
        """
        streamers = []

        if pairs is not None:
            if hasattr(pairs, "keys"):
                all_pairs = itertools.chain(
                    ((k, pairs[k]) for k in pairs.keys()), more_pairs.items()
                )
            else:
                all_pairs = itertools.chain(pairs, more_pairs.items())
        else:
            all_pairs = more_pairs.items()

        for k, v in all_pairs:
            fullpath = k.strip("/").split("/")
            path, name = fullpath[:-1], fullpath[-1]

            if len(path) != 0:
                self.mkdir(
                    "/".join(path),
                    initial_directory_bytes=self._file.initial_directory_bytes,
                )

            directory = self
            for item in path:
                directory = directory[item]

            uproot.writing.identify.add_to_directory(v, name, directory, streamers)

        self._file._cascading.streamers.update_streamers(self._file.sink, streamers)


class WritableTree:
    """
    Args:
        path (tuple of str): Path of directory names to this TTree.
        file (:doc:`uproot.writing.writable.WritableFile`): Handle to the file in
            which this TTree can be found.
        cascading (:doc:`uproot.writing._cascadetree.Tree`): The low-level
            directory object.

    Represents a writable ``TTree`` from a ROOT file.

    This object would normally be created by assigning a TTree-like data to a
    :doc:`uproot.writing.writable.WritableDirectory`. For instance:

    .. code-block:: python

        my_directory["tree1"] = {"branch1": np.array(...), "branch2": ak.Array(...)}
        my_directory["tree2"] = numpy_structured_array
        my_directory["tree3"] = awkward_record_array
        my_directory["tree4"] = pandas_dataframe

    Recognized data types:

    * dict of NumPy arrays (flat, multidimensional, and/or structured), Awkward Arrays containing one level of variable-length lists and/or one level of records, or a Pandas DataFrame with a numeric index
    * a single NumPy structured array (one level deep)
    * a single Awkward Array containing one level of variable-length lists and/or one level of records
    * a single Pandas DataFrame with a numeric index

    The arrays may have different types, but their lengths must be identical, at
    least in the first dimension (i.e. number of entries).

    If the Awkward Array contains variable-length lists (i.e. it is "jagged"), a
    counter TBranch will be created along with the data TBranch. ROOT needs the
    counter TBranch to quantify the size of the variable-size arrays. Combining
    Awkward Arrays with the same number of nested items using
    `ak.zip <https://awkward-array.readthedocs.io/en/latest/_auto/ak.zip.html>`__ prevents
    a proliferation of counter TBranches:

    .. code-block:: python

        my_directory["tree5"] = ak.zip({"branch1": array1, "branch2": array2, "branch3": array3})

    would produce only one counter TBranch.

    Assigning TTree-like data to a directory creates the TTree object with all of
    its metadata and fills it with the contents of the arrays in one step. To separate
    the process of creating the TTree metadata from filling the first TBasket, use the
    :doc:`uproot.writing.writable.WritableDirectory.mktree` method:

    .. code-block:: python

        my_directory.mktree("tree6", {"branch1": numpy_dtype, "branch2": awkward_type})

    The :doc:`uproot.writing.writable.WritableDirectory.mktree` method can also control the
    title of the TTree and the rules used to name counter TBranches and nested field TBranches.

    The ``numpy_dtype`` is any data that NumPy recognizes as a ``np.dtype``, and the
    ``awkward_type`` is an `ak.types.Type <https://awkward-array.readthedocs.io/en/latest/ak.types.Type.html>`__ from
    `ak.type <https://awkward-array.readthedocs.io/en/latest/_auto/ak.type.html>`__ or
    a string in that form, such as ``"var * float64"`` for variable-length doubles.

    TBaskets can be added to each TBranch using the :ref:`uproot.writing.writable.WritableTree.extend`
    method:

    .. code-block:: python

        my_directory["tree6"].extend({"branch1": another_numpy_array,
                                      "branch2": another_awkward_array})

    Be sure to make these extensions as large as is feasible within memory constraints,
    because a ROOT file full of small TBaskets is bloated (larger than it needs to be)
    and slow to read (especially for Uproot, but also for ROOT).

    For instance, if you want to write a million events and have enough memory
    available to do that 100 thousand events at a time (total of 10 TBaskets),
    then do so. Filling the TTree a hundred events at a time (total of 10000 TBaskets)
    would be considerably slower for writing and reading, and the file would be much
    larger than it could otherwise be, even with compression.
    """

    def __init__(self, path, file, cascading):
        self._path = path
        self._file = file
        self._cascading = cascading

    def __repr__(self):
        return "<WritableTree {} at 0x{:012x}>".format(
            repr("/" + "/".join(self._path)), id(self)
        )

    @property
    def path(self):
        """
        Path of directory names to this TTree as a tuple of strings.
        """
        return self._path

    @property
    def object_path(self):
        """
        Path of directory names to this TTree as a single string, delimited by
        slashes.
        """
        return "/".join(("",) + self._path + ("",)).replace("//", "/")

    @property
    def file_path(self):
        """
        Filesystem path of the open file, or None if using a file-like object.
        """
        return self._file.file_path

    @property
    def file(self):
        """
        Handle to the :doc:`uproot.writing.writable.WritableDirectory` in which
        this directory can be found.
        """
        return self._file

    def close(self):
        """
        Explicitly close the file.

        (Files can also be closed with the Python ``with`` statement, as context
        managers.)

        After closing, objects cannot be read from or written to the file.
        """
        self._file.close()

    @property
    def closed(self):
        """
        True if the file has been closed; False otherwise.

        The file may have been closed explicitly with
        :ref:`uproot.writing.writable.WritableFile.close` or implicitly in the Python
        ``with`` statement, as a context manager.

        After closing, objects cannot be read from or written to the file.
        """
        return self._file.closed

    def __enter__(self):
        self._file.sink.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.sink.__exit__(exception_type, exception_value, traceback)

    @property
    def compression(self):
        """
        Compression algorithm and level (:doc:`uproot.compression.Compression` or None)
        for new TBaskets added to the TTree.

        This property can be changed and doesn't have to be the same as the compression
        of the file, which allows you to write different objects with different
        compression settings.

        The following are equivalent:

        .. code-block:: python

            my_directory["tree"]["branch1"].compression = uproot.ZLIB(1)
            my_directory["tree"]["branch2"].compression = uproot.LZMA(9)

        and

        .. code-block:: python

            my_directory["tree"].compression = {"branch1": uproot.ZLIB(1),
                                                "branch2": uproot.LZMA(9)}
        """
        out = {}
        last = None
        for datum in self._cascading._branch_data:
            if datum["kind"] != "record":
                last = out[datum["fName"]] = datum["compression"]
        if all(x == last for x in out.values()):
            return last
        else:
            return out

    @compression.setter
    def compression(self, value):
        if value is None or isinstance(value, uproot.compression.Compression):
            for datum in self._cascading._branch_data:
                if datum["kind"] != "record":
                    datum["compression"] = value

        elif (
            isinstance(value, Mapping)
            and all(
                uproot._util.isstr(k)
                and (v is None or isinstance(v, uproot.compression.Compression))
                for k, v in value.items()
            )
            and all(
                datum["fName"] in value
                for datum in self._cascading._branch_data
                if datum["kind"] != "record"
            )
            and len(value)
            == len(
                [
                    datum
                    for datum in self._cascading._branch_data
                    if datum["kind"] != "record"
                ]
            )
        ):
            for datum in self._cascading._branch_data:
                if datum["kind"] != "record":
                    datum["compression"] = value[datum["fName"]]

        else:
            raise TypeError(
                "compression must be None, a uproot.compression.Compression object, like uproot.ZLIB(4) or uproot.ZSTD(0), or a mapping of branch names to such objects"
            )

    def __getitem__(self, where):
        for datum in self._cascading._branch_data:
            if datum["kind"] != "record" and datum["fName"] == where:
                return WritableBranch(self, datum)
        else:
            raise uproot.KeyInFileError(
                where,
                because="no such branch in writable tree",
                file_path=self.file_path,
            )

    @property
    def num_entries(self):
        """
        The number of entries accumulated so far.
        """
        return self._cascading.num_entries

    @property
    def num_baskets(self):
        """
        The number of TBaskets accumulated so far.
        """
        return self._cascading.num_baskets

    def extend(self, data):
        """
        Args:
            data (dict of str \u2192 arrays): More array data to add to the TTree.

        This method adds data to an existing TTree, whether it was created through
        assignment or :doc:`uproot.writing.writable.WritableDirectory.mktree`.

        The arrays must be a dict, but the values of the dict can be any of the
        array/DataFrame types described in :doc:`uproot.writing.writable.WritableTree`.
        However, these types must be compatible with the established TBranch
        types, the dict must contain a key for every TBranch, and the arrays must have
        the same lengths (in their first dimension).

        For example,

        .. code-block:: python

            my_directory.mktree("tree6", {"branch1": numpy_dtype, "branch2": awkward_type})

            my_directory["tree6"].extend({"branch1": another_numpy_array,
                                          "branch2": another_awkward_array})

        .. warning::

            **As a word of warning,** be sure that each call to :ref:`uproot.writing.writable.WritableTree.extend` includes at least 100 kB per branch/array. (NumPy and Awkward Arrays have an `nbytes <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nbytes.html>`__ property; you want at least ``100000`` per array.) If you ask Uproot to write very small TBaskets, it will spend more time working on TBasket overhead than actually writing data. The absolute worst case is one-entry-per-:ref:`uproot.writing.writable.WritableTree.extend`. See `#428 (comment) <https://github.com/scikit-hep/uproot4/pull/428#issuecomment-908703486>`__.
        """
        self._cascading.extend(self._file, self._file.sink, data)

    def show(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
        name_width=20,
        typename_width=24,
        interpretation_width=30,
        stream=sys.stdout,
    ):
        """
        Opens the TTree for reading and calls :doc:`uproot.behaviors.TBranch.HasBranches.show`
        on it (follow link for documentation of this method).
        """
        uproot.open(self._file.sink._file)[self.object_path].show(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=full_paths,
            name_width=name_width,
            typename_width=typename_width,
            interpretation_width=interpretation_width,
            stream=stream,
        )


class WritableBranch:
    """
    Represents a TBranch from a :doc:`uproot.writing.writable.WritableTree`.

    This object exists only to be able to assign compression settings differently
    on each TBranch:

    .. code-block:: python

        my_directory["tree"]["branch1"].compression = uproot.ZLIB(1)
        my_directory["tree"]["branch2"].compression = uproot.LZMA(9)

    Note that compression settings on all TBranches can be set through
    :doc:`uproot.writing.writable.WritableTree.compression`:

    .. code-block:: python

        my_directory["tree"].compression = {"branch1": uproot.ZLIB(1),
                                            "branch2": uproot.LZMA(9)}
    """

    def __init__(self, tree, datum):
        self._tree = tree
        self._datum = datum

    def __repr__(self):
        return "<WritableBranch {} in {} at 0x{:012x}>".format(
            repr(self._datum["fName"]), repr("/" + "/".join(self._tree.path)), id(self)
        )

    @property
    def type(self):
        """
        The type used to initialize this TBranch.
        """
        return self._datum["branch_type"]

    @property
    def compression(self):
        """
        Compression algorithm and level (:doc:`uproot.compression.Compression` or None)
        for new TBaskets added to the TBranch.

        This property can be changed and doesn't have to be the same as the compression
        of the file or the rest of the TTree, which allows you to write different objects
        with different compression settings.

        The following are equivalent:

        .. code-block:: python

            my_directory["tree"]["branch1"].compression = uproot.ZLIB(1)
            my_directory["tree"]["branch2"].compression = uproot.LZMA(9)

        and

        .. code-block:: python

            my_directory["tree"].compression = {"branch1": uproot.ZLIB(1),
                                                "branch2": uproot.LZMA(9)}
        """
        return self._datum["compression"]

    @compression.setter
    def compression(self, value):
        if value is None or isinstance(value, uproot.compression.Compression):
            self._datum["compression"] = value
        else:
            raise TypeError(
                "compression must be None or a uproot.compression.Compression object, like uproot.ZLIB(4) or uproot.ZSTD(0)"
            )
