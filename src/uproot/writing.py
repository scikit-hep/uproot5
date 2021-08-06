# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import itertools
import os
import uuid

try:
    import queue
except ImportError:
    import Queue as queue

import numpy

import uproot._util
import uproot._writing
import uproot.compression
import uproot.deserialization
import uproot.exceptions
import uproot.model
import uproot.models.TObjString
import uproot.sink.file
from uproot._util import no_filter, no_rename


def create(file_path, **options):
    """
    FIXME: docstring
    """
    file_path = uproot._util.regularize_path(file_path)
    if uproot._util.isstr(file_path):
        if os.path.exists(file_path):
            raise OSError(
                "path exists and refusing to overwrite (use 'uproot.recreate' to "
                "overwrite)\n\nfor path {0}".format(file_path)
            )
    return recreate(file_path, **options)


def recreate(file_path, **options):
    """
    FIXME: docstring
    """
    file_path = uproot._util.regularize_path(file_path)
    if uproot._util.isstr(file_path):
        if not os.path.exists(file_path):
            with open(file_path, "a") as tmp:
                tmp.seek(0)
                tmp.truncate()
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
    if len(options) != 0:
        raise TypeError(
            "unrecognized options for uproot.create or uproot.recreate: "
            + ", ".join(repr(x) for x in options)
        )

    cascading = uproot._writing.create_empty(
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
    FIXME: docstring
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
    if len(options) != 0:
        raise TypeError(
            "unrecognized options for uproot.update: "
            + ", ".join(repr(x) for x in options)
        )

    cascading = uproot._writing.update_existing(
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
    FIXME: docstring
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

    def __repr__(self):
        return "<WritableFile {0} at 0x{1:012x}>".format(repr(self.file_path), id(self))

    @property
    def sink(self):
        return self._sink

    @property
    def initial_directory_bytes(self):
        return self._initial_directory_bytes

    @initial_directory_bytes.setter
    def initial_directory_bytes(self, value):
        self._initial_directory_bytes = value

    @property
    def uuid_function(self):
        return self._uuid_function

    @uuid_function.setter
    def uuid_function(self, value):
        self._uuid_function = value

    @property
    def options(self):
        return {
            "initial_directory_bytes": self._initial_directory_bytes,
            "uuid_function": self._uuid_function,
        }

    @property
    def is_64bit(self):
        return self._cascading.fileheader.big

    @property
    def compression(self):
        return self._cascading.fileheader.compression

    @compression.setter
    def compression(self, value):
        self._cascading.fileheader.compression = value

    @property
    def fSeekFree(self):
        return self._cascading.fileheader.free_location

    @property
    def fNbytesFree(self):
        return self._cascading.fileheader.free_num_bytes

    @property
    def nfree(self):
        return self._cascading.fileheader.free_num_slices + 1

    @property
    def fUnits(self):
        return 8 if self._cascading.fileheader.big else 4

    @property
    def fCompress(self):
        return self._cascading.fileheader.compression.code

    @property
    def fSeekInfo(self):
        return self._cascading.fileheader.info_location

    @property
    def fNbytesInfo(self):
        return self._cascading.fileheader.info_num_bytes

    @property
    def uuid(self):
        return self._cascading.fileheader.uuid

    @property
    def root_directory(self):
        return WritableDirectory((), self, self._cascading.rootdirectory)

    def update_streamers(self, streamers):
        self._cascading.streamers.update_streamers(self.sink, streamers)

    def close(self):
        self._sink.close()

    @property
    def closed(self):
        return self._sink.closed

    def __enter__(self):
        self._sink.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._sink.__exit__(exception_type, exception_value, traceback)


class WritableDirectory(object):
    """
    FIXME: docstring
    """

    def __init__(self, path, file, cascading):
        self._path = path
        self._file = file
        self._cascading = cascading
        self._subdirs = {}

    def __repr__(self):
        return "<WritableDirectory {0} at 0x{1:012x}>".format(
            repr("/" + "/".join(self._path)), id(self)
        )

    @property
    def path(self):
        return self._path

    @property
    def object_path(self):
        return "/".join(("",) + self._path + ("",)).replace("//", "/")

    @property
    def file_path(self):
        return self._file.file_path

    @property
    def file(self):
        return self._file

    def close(self):
        self._file.close()

    @property
    def closed(self):
        return self._file.closed

    def __enter__(self):
        self._file.sink.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.sink.__exit__(exception_type, exception_value, traceback)

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
        return self.iterkeys()  # noqa B301 (not a dict)

    def _ipython_key_completions_(self):
        """
        Supports key-completion in an IPython or Jupyter kernel.
        """
        return self.iterkeys()  # noqa: B301 (not a dict)

    def keys(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        u"""
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
            self.iterkeys(  # noqa: B301 (not a dict)
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def values(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        u"""
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
            self.itervalues(  # noqa: B301 (not a dict)
                recursive=recursive,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def items(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        u"""
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
            self.iteritems(  # noqa: B301 (not a dict)
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def classnames(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        u"""
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
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        u"""
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
                    yield "{0};{1}".format(keyname, cyclenum)
                else:
                    yield keyname

            if recursive and classname in ("TDirectory", "TDirectoryFile"):
                for k1 in self._get(  # noqa: B301 (not a dict)
                    keyname, cyclenum
                ).iterkeys(
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=filter_name,
                    filter_classname=filter_classname,
                ):
                    k2 = "{0}/{1}".format(keyname, k1)
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2

    def itervalues(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        u"""
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
        for keyname in self.iterkeys(  # noqa: B301 (not a dict)
            recursive=recursive,
            cycle=True,
            filter_name=filter_name,
            filter_classname=filter_classname,
        ):
            yield self[keyname]

    def iteritems(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        u"""
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
        for keyname in self.iterkeys(  # noqa: B301 (not a dict)
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
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        u"""
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
                    yield "{0};{1}".format(keyname, cyclenum), classname
                else:
                    yield keyname, classname

            if recursive and classname in ("TDirectory", "TDirectoryFile"):
                for k1, c1 in self._get(
                    keyname, cyclenum
                ).iterclassnames(  # noqa: B301 (not a dict)
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=filter_name,
                    filter_classname=filter_classname,
                ):
                    k2 = "{0}/{1}".format(keyname, k1)
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2, c1

    def __getitem__(self, where):
        if "/" in where or ":" in where:
            items = where.split("/")
            step = last = self

            for i, item in enumerate(items):
                if item != "":
                    if isinstance(step, WritableDirectory):
                        if ":" in item and not step._cascading.data.haskey(item):
                            index = item.index(":")
                            head, tail = item[:index], item[index + 1 :]
                            last = step
                            step = step.get(head)
                            if isinstance(step, uproot.behaviors.TBranch.HasBranches):
                                return step["/".join([tail] + items[i + 1 :])]
                            else:
                                raise uproot.KeyInFileError(
                                    where,
                                    because=repr(head)
                                    + " is not a TDirectory, TTree, or TBranch",
                                    keys=last._cascading.data.key_names,
                                    file_path=self.file_path,
                                )
                        else:
                            last = step
                            step = step[item]

                    elif isinstance(step, uproot.behaviors.TBranch.HasBranches):
                        return step["/".join(items[i:])]

                    else:
                        raise uproot.KeyInFileError(
                            where,
                            because=repr(item)
                            + " is not a TDirectory, TTree, or TBranch",
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

            return self._get(item, cycle)

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

        else:

            def get_chunk(start, stop):
                raw_bytes = self._file.sink.read(start, stop - start)
                return uproot.source.chunk.Chunk.wrap(
                    readforupdate, raw_bytes, start=start
                )

            readforupdate = uproot._writing._ReadForUpdate(
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

    def _subdir(self, key):
        name = key.name.string

        if name not in self._subdirs:
            raw_bytes = self._file.sink.read(
                key.seek_location,
                key.num_bytes + uproot.reading._directory_format_big.size + 18,
            )
            directory_key = uproot._writing.Key.deserialize(
                raw_bytes, key.seek_location, self._file.sink.in_path
            )
            position = key.seek_location + directory_key.num_bytes

            directory_header = uproot._writing.DirectoryHeader.deserialize(
                raw_bytes[position - key.seek_location :],
                position,
                self._file.sink.in_path,
            )
            assert directory_header.begin_location == key.seek_location
            assert (
                directory_header.parent_location
                == self._file._cascading.fileheader.begin
            )

            if directory_header.data_num_bytes == 0:
                directory_datakey = uproot._writing.Key(
                    None,
                    None,
                    None,
                    uproot._writing.String(None, "TDirectory"),
                    uproot._writing.String(None, name),
                    uproot._writing.String(None, name),
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
                directory_data = uproot._writing.DirectoryData(
                    directory_datakey.location + directory_datakey.num_bytes,
                    might_be_slightly_more,
                    [],
                )

                directory_datakey.uncompressed_bytes = directory_data.allocation
                directory_datakey.compressed_bytes = (
                    directory_datakey.uncompressed_bytes
                )

                subdirectory = uproot._writing.SubDirectory(
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

                directory_datakey = uproot._writing.Key.deserialize(
                    raw_bytes, directory_header.data_location, self._file.sink.in_path
                )
                directory_data = uproot._writing.DirectoryData.deserialize(
                    raw_bytes[directory_datakey.num_bytes :],
                    directory_header.data_location + directory_datakey.num_bytes,
                    self._file.sink.in_path,
                )

                subdirectory = uproot._writing.SubDirectory(
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

    def mkdir(self, name, initial_directory_bytes=None):
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
                """cannot make a directory named {0} because a {1} already has that name
in file {2} in directory {3}""".format(
                    repr(name), key.classname.string, self.file_path, self.path
                )
            )

        else:
            directory = self._subdir(key)

        if tail is None:
            return directory

        else:
            return directory.mkdir(tail)

    def copy_from(
        self,
        source,
        filter_name=no_filter,
        filter_classname=no_filter,
        rename=no_rename,
        require_matches=True,
    ):
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
                    """no objects found with names matching {0}
in file {1} in directory {2}""".format(
                        repr(filter_name), source.file_path, source.path
                    )
                )
            else:
                return

        keys = [source.key(x) for x in old_names]

        for key in keys:
            if key.fClassName == "TTree" or key.fClassName.split("::")[-1] == "RNTuple":
                raise NotImplementedError(
                    "copy_from cannot copy {0} objects yet".format(key.fClassName)
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
        for classname in set(x.fClassName for x in keys):
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
            self.mkdir(name, max(self._file.initial_directory_bytes, allocation))

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

    def __setitem__(self, where, what):
        self.update({where: what})

    def update(self, pairs=None, **more_pairs):
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
            writable = to_writable(v)

            for rawstreamer in writable.class_rawstreamers:
                streamers.append(rawstreamer)

            fullpath = k.strip("/").split("/")
            path, name = fullpath[:-1], fullpath[-1]

            raw_data = writable.serialize(name=name)

            if hasattr(writable, "fTitle"):
                title = writable.fTitle
            elif writable.has_member("fTitle"):
                title = writable.member("fTitle")
            else:
                title = ""

            if len(path) != 0:
                self.mkdir("/".join(path), self._file.initial_directory_bytes)

            directory = self
            for item in path:
                directory = directory[item]

            directory._cascading.add_object(
                self._file.sink,
                writable.classname,
                name,
                title,
                raw_data,
                len(raw_data),
            )

        self._file._cascading.streamers.update_streamers(self._file.sink, streamers)


def to_writable(obj):
    """
    FIXME: docstring
    """
    if isinstance(obj, uproot.model.Model):
        return obj.to_writable()

        raise NotImplementedError(
            "this ROOT type is not writable: {0}".format(obj.classname)
        )

    elif uproot._util.isstr(obj):
        return to_TObjString(obj)

    else:
        raise TypeError(
            "unrecognized type cannot be written to a ROOT file: " + type(obj).__name__
        )


def to_TString(string):
    """
    This function is for developers to create TString objects that can be
    written to ROOT files, to implement conversion routines.
    """
    tstring = uproot.models.TString.Model_TString(str(string))
    tstring._deeply_writable = True
    tstring._cursor = None
    tstring._file = None
    tstring._parent = None
    tstring._members = {}
    tstring._bases = []
    tstring._num_bytes = None
    tstring._instance_version = None
    return tstring


def to_TObjString(string):
    """
    This function is for developers to create TObjString objects that can be
    written to ROOT files, to implement conversion routines.
    """
    tobjstring = uproot.models.TObjString.Model_TObjString(str(string))
    tobjstring._deeply_writable = True
    tobjstring._cursor = None
    tobjstring._parent = None
    tobjstring._members = {}
    tobjstring._bases = (uproot.models.TObject.Model_TObject(),)
    tobjstring._num_bytes = len(string) + (1 if len(string) < 255 else 5) + 16
    tobjstring._instance_version = 1
    return tobjstring


def to_TList(data, name=""):
    """
    Args:
        data (:doc:`uproot.model.Model`): Python iterable to convert into a TList.
        name (str): Name of the list (usually empty: ``""``).

    This function is for developers to create TList objects that can be
    written to ROOT files, to implement conversion routines.
    """
    if not all(isinstance(x, uproot.model.Model) for x in data):
        raise TypeError(
            "list to convert to TList must only contain ROOT objects (uproot.Model)"
        )

    tobject = uproot.models.TObject.Model_TObject.empty()
    tobject._members["@fUniqueID"] = 0
    tobject._members["@fBits"] = 0

    tlist = uproot.models.TList.Model_TList.empty()
    tlist._bases.append(tobject)
    tlist._members["fName"] = name
    tlist._data = list(data)
    tlist._members["fSize"] = len(tlist._data)
    tlist._options = [b""] * len(tlist._data)

    if all(x._deeply_writable for x in tlist._data):
        tlist._deeply_writable = True

    return tlist


def to_TArray(data):
    """
    Args:
        data (numpy.ndarray): The array to convert to big-endian and wrap as
            TArrayC, TArrayS, TArrayI, TArrayL, TArrayF, or TArrayD, depending
            on its dtype.

    This function is for developers to create TArray objects that can be
    written to ROOT files, to implement conversion routines.
    """
    if data.ndim != 1:
        raise ValueError("data to convert to TArray must be one-dimensional")

    if issubclass(data.dtype.type, numpy.int8):
        cls = uproot.models.TArray.Model_TArrayC
    elif issubclass(data.dtype.type, numpy.int16):
        cls = uproot.models.TArray.Model_TArrayS
    elif issubclass(data.dtype.type, numpy.int32):
        cls = uproot.models.TArray.Model_TArrayI
    elif issubclass(data.dtype.type, numpy.int64):
        cls = uproot.models.TArray.Model_TArrayL
    elif issubclass(data.dtype.type, numpy.float32):
        cls = uproot.models.TArray.Model_TArrayF
    elif issubclass(data.dtype.type, numpy.float64):
        cls = uproot.models.TArray.Model_TArrayD
    else:
        raise ValueError(
            "data to convert to TArray must have signed integer or floating-point type, not {0}".format(
                repr(data.dtype)
            )
        )

    tarray = cls.empty()
    tarray._deeply_writable = True
    tarray._members["fN"] = len(data)
    tarray._data = data.astype(data.dtype.newbyteorder(">"))
    return tarray


def to_TAxis(
    fName,
    fTitle,
    fNbins,
    fXmin,
    fXmax,
    fXbins=None,
    fFirst=0,
    fLast=0,
    fBits2=0,
    fTimeDisplay=False,
    fTimeFormat="",
    fLabels=None,
    fModLabs=None,
    fNdivisions=510,
    fAxisColor=1,
    fLabelColor=1,
    fLabelFont=42,
    fLabelOffset=0.005,
    fLabelSize=0.035,
    fTickLength=0.03,
    fTitleOffset=1.0,
    fTitleSize=0.035,
    fTitleColor=1,
    fTitleFont=42,
):
    """
    Args:
        fName (str): Internal name of axis, usually ``"xaxis"``, ``"yaxis"``, ``"zaxis"``.
        fTitle (str): Internal title of axis, usually empty: ``""``.
        fNbins (int): Number of bins. (https://root.cern.ch/doc/master/classTAxis.html)
        fXmin (float): Low edge of first bin.
        fXmax (float): Upper edge of last bin.
        fXbins (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Bin
            edges array in X. None generates an empty array.
        fFirst (int): First bin to display. 1 if no range defined NOTE: in some cases a zero is returned (see TAxis::SetRange)
        fLast (int): Last bin to display. fNbins if no range defined NOTE: in some cases a zero is returned (see TAxis::SetRange)
        fBits2 (int): Second bit status word.
        fTimeDisplay (bool): On/off displaying time values instead of numerics.
        fTimeFormat (str or :doc:`uproot.models.TString.Model_TString`): Date&time format, ex: 09/12/99 12:34:00.
        fLabels (None or :doc:`uproot.models.THashList.Model_THashList`): List of labels.
        fModLabs (None or :doc:`uproot.models.List.Model_TList`): List of modified labels.
        fNdivisions (int): Number of divisions(10000*n3 + 100*n2 + n1). (https://root.cern.ch/doc/master/classTAttAxis.html)
        fAxisColor (int): Color of the line axis.
        fLabelColor (int): Color of labels.
        fLabelFont (int): Font for labels.
        fLabelOffset (float): Offset of labels.
        fLabelSize (float): Size of labels.
        fTickLength (float): Length of tick marks.
        fTitleOffset (float): Offset of axis title.
        fTitleSize (float): Size of axis title.
        fTitleColor (int): Color of axis title.
        fTitleFont (int): Font for axis title.

    This function is for developers to create TAxis objects that can be
    written to ROOT files, to implement conversion routines.
    """
    tobject = uproot.models.TObject.Model_TObject.empty()
    tobject._members["@fUniqueID"] = 0
    tobject._members["@fBits"] = 0

    tnamed = uproot.models.TNamed.Model_TNamed.empty()
    tnamed._deeply_writable = True
    tnamed._bases.append(tobject)
    tnamed._members["fName"] = fName
    tnamed._members["fTitle"] = fTitle

    tattaxis = uproot.models.TAtt.Model_TAttAxis_v4.empty()
    tattaxis._deeply_writable = True
    tattaxis._members["fNdivisions"] = fNdivisions
    tattaxis._members["fAxisColor"] = fAxisColor
    tattaxis._members["fLabelColor"] = fLabelColor
    tattaxis._members["fLabelFont"] = fLabelFont
    tattaxis._members["fLabelOffset"] = fLabelOffset
    tattaxis._members["fLabelSize"] = fLabelSize
    tattaxis._members["fTickLength"] = fTickLength
    tattaxis._members["fTitleOffset"] = fTitleOffset
    tattaxis._members["fTitleSize"] = fTitleSize
    tattaxis._members["fTitleColor"] = fTitleColor
    tattaxis._members["fTitleFont"] = fTitleFont

    if fXbins is None:
        fXbins = numpy.array([], dtype=numpy.float64)

    if isinstance(fXbins, uproot.models.TArray.Model_TArrayD):
        tarray_fXbins = fXbins
    else:
        tarray_fXbins = to_TArray(fXbins)

    if isinstance(fTimeFormat, uproot.models.TString.Model_TString):
        tstring_fTimeFormat = fTimeFormat
    else:
        tstring_fTimeFormat = to_TString(fTimeFormat)

    taxis = uproot.models.TH.Model_TAxis_v10.empty()
    taxis._deeply_writable = True
    taxis._bases.append(tnamed)
    taxis._bases.append(tattaxis)
    taxis._members["fNbins"] = fNbins
    taxis._members["fXmin"] = fXmin
    taxis._members["fXmax"] = fXmax
    taxis._members["fXbins"] = tarray_fXbins
    taxis._members["fFirst"] = fFirst
    taxis._members["fLast"] = fLast
    taxis._members["fBits2"] = fBits2
    taxis._members["fTimeDisplay"] = fTimeDisplay
    taxis._members["fTimeFormat"] = tstring_fTimeFormat
    taxis._members["fLabels"] = fLabels
    taxis._members["fModLabs"] = fModLabs

    return taxis


def to_TH1x(
    fName,
    fTitle,
    data,
    fEntries,
    fTsumw,
    fTsumw2,
    fTsumwx,
    fTsumwx2,
    fSumw2,
    fXaxis,
    fYaxis=None,
    fZaxis=None,
    fNcells=None,
    fBarOffset=0,
    fBarWidth=1000,
    fMaximum=-1111.0,
    fMinimum=-1111.0,
    fNormFactor=0.0,
    fContour=None,
    fOption="",
    fFunctions=None,
    fBufferSize=0,
    fBuffer=None,
    fBinStatErrOpt=0,
    fStatOverflows=2,
    fLineColor=602,
    fLineStyle=1,
    fLineWidth=1,
    fFillColor=0,
    fFillStyle=1001,
    fMarkerColor=1,
    fMarkerStyle=1,
    fMarkerSize=1.0,
):
    """
    Args:
        fName (None or str): Temporary name, will be overwritten by the writing
            process because Uproot's write syntax is ``file[name] = histogram``.
        fTitle (str): Real title of the histogram.
        data (numpy.ndarray or :doc:`uproot.models.TArray.Model_TArray`): Bin contents
            with first bin as underflow, last bin as overflow. The dtype of this array
            determines the return type of this function (TH1C, TH1D, TH1F, TH1I, or TH1S).
        fEntries (float): Number of entries. (https://root.cern.ch/doc/master/classTH1.html)
        fTsumw (float): Total Sum of weights.
        fTsumw2 (float): Total Sum of squares of weights.
        fTsumwx (float): Total Sum of weight*X.
        fTsumwx2 (float): Total Sum of weight*X*X.
        fSumw2 (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights.
        fXaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.to_TAxis`
            with ``fName="xaxis"`` and ``fTitle=""``.
        fYaxis (None or :doc:`uproot.models.TH.Model_TAxis_v10`): None generates a
            default for 1D histograms.
        fZaxis (None or :doc:`uproot.models.TH.Model_TAxis_v10`): None generates a
            default for 1D and 2D histograms.
        fNcells (None or int): Number of bins(1D), cells (2D) +U/Overflows. Computed
            from ``data`` if None.
        fBarOffset (int): (1000*offset) for bar charts or legos
        fBarWidth (int): (1000*width) for bar charts or legos
        fMaximum (float): Maximum value for plotting.
        fMinimum (float): Minimum value for plotting.
        fNormFactor (float): Normalization factor.
        fContour (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            to display contour levels. None generates an empty array.
        fOption (str or :doc:`uproot.models.TString.Model_TString`): Histogram options.
        fFunctions (None, list, or :doc:`uproot.models.TList.Model_TList`): ->Pointer to
            list of functions (fits and user). None generates an empty list.
        fBufferSize (None or int): fBuffer size. Computed from ``fBuffer`` if None.
        fBuffer (None or numpy.ndarray of numpy.float64): Buffer of entries accumulated
            before automatically choosing the binning. (Irrelevant for serialization?)
            None generates an empty array.
        fBinStatErrOpt (int): Option for bin statistical errors.
        fStatOverflows (int): Per object flag to use under/overflows in statistics.
        fLineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        fLineStyle (int): Line style.
        fLineWidth (int): Line width.
        fFillColor (int): Fill area color. (https://root.cern.ch/doc/master/classTAttFill.html)
        fFillStyle (int): Fill area style.
        fMarkerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        fMarkerStyle (int): Marker style.
        fMarkerSize (float): Marker size.

    This function is for developers to create TH1* objects that can be
    written to ROOT files, to implement conversion routines. The choice of
    TH1C, TH1D, TH1F, TH1I, or TH1S depends on the dtype of the ``data`` array.
    """
    tobject = uproot.models.TObject.Model_TObject.empty()
    tobject._members["@fUniqueID"] = 0
    tobject._members["@fBits"] = 0

    tnamed = uproot.models.TNamed.Model_TNamed.empty()
    tnamed._deeply_writable = True
    tnamed._bases.append(tobject)
    tnamed._members["fName"] = fName
    tnamed._members["fTitle"] = fTitle

    tattline = uproot.models.TAtt.Model_TAttLine_v2.empty()
    tattline._deeply_writable = True
    tattline._members["fLineColor"] = fLineColor
    tattline._members["fLineStyle"] = fLineStyle
    tattline._members["fLineWidth"] = fLineWidth

    tattfill = uproot.models.TAtt.Model_TAttFill_v2.empty()
    tattfill._deeply_writable = True
    tattfill._members["fFillColor"] = fFillColor
    tattfill._members["fFillStyle"] = fFillStyle

    tattmarker = uproot.models.TAtt.Model_TAttMarker_v2.empty()
    tattmarker._deeply_writable = True
    tattmarker._members["fMarkerColor"] = fMarkerColor
    tattmarker._members["fMarkerStyle"] = fMarkerStyle
    tattmarker._members["fMarkerSize"] = fMarkerSize

    th1 = uproot.models.TH.Model_TH1_v8.empty()

    th1._bases.append(tnamed)
    th1._bases.append(tattline)
    th1._bases.append(tattfill)
    th1._bases.append(tattmarker)

    if fYaxis is None:
        fYaxis = to_TAxis(fName="yaxis", fTitle="", fNbins=1, fXmin=0.0, fXmax=1.0)
    if fZaxis is None:
        fZaxis = to_TAxis(fName="zaxis", fTitle="", fNbins=1, fXmin=0.0, fXmax=1.0)
    if fContour is None:
        fContour = numpy.array([], dtype=numpy.float64)
    if fFunctions is None:
        fFunctions = []
    if fBuffer is None:
        fBuffer = numpy.array([], dtype=numpy.float64)

    if isinstance(data, uproot.models.TArray.Model_TArray):
        tarray_data = data
    else:
        tarray_data = to_TArray(data)

    if isinstance(fSumw2, uproot.models.TArray.Model_TArray):
        tarray_fSumw2 = fSumw2
    else:
        tarray_fSumw2 = to_TArray(fSumw2)
    if not isinstance(tarray_fSumw2, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fSumw2 must be an array of float64 (TArrayD)")

    if isinstance(fContour, uproot.models.TArray.Model_TArray):
        tarray_fContour = fContour
    else:
        tarray_fContour = to_TArray(fContour)
    if not isinstance(tarray_fContour, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fContour must be an array of float64 (TArrayD)")

    if isinstance(fOption, uproot.models.TString.Model_TString):
        tstring_fOption = fOption
    else:
        tstring_fOption = to_TString(fOption)

    if isinstance(fFunctions, uproot.models.TList.Model_TList):
        tlist_fFunctions = fFunctions
    else:
        tlist_fFunctions = to_TList(fFunctions, name="")
    # FIXME: require all list items to be the appropriate class (TFunction?)

    th1._members["fNcells"] = len(tarray_data) if fNcells is None else fNcells
    th1._members["fXaxis"] = fXaxis
    th1._members["fYaxis"] = fYaxis
    th1._members["fZaxis"] = fZaxis
    th1._members["fBarOffset"] = fBarOffset
    th1._members["fBarWidth"] = fBarWidth
    th1._members["fEntries"] = fEntries
    th1._members["fTsumw"] = fTsumw
    th1._members["fTsumw2"] = fTsumw2
    th1._members["fTsumwx"] = fTsumwx
    th1._members["fTsumwx2"] = fTsumwx2
    th1._members["fMaximum"] = fMaximum
    th1._members["fMinimum"] = fMinimum
    th1._members["fNormFactor"] = fNormFactor
    th1._members["fContour"] = tarray_fContour
    th1._members["fSumw2"] = tarray_fSumw2
    th1._members["fOption"] = tstring_fOption
    th1._members["fFunctions"] = tlist_fFunctions
    th1._members["fBufferSize"] = len(fBuffer) if fBufferSize is None else fBufferSize
    th1._members["fBuffer"] = fBuffer
    th1._members["fBinStatErrOpt"] = fBinStatErrOpt
    th1._members["fStatOverflows"] = fStatOverflows

    th1._speedbump1 = b"\x00"

    th1._deeply_writable = tlist_fFunctions._deeply_writable

    if isinstance(tarray_data, uproot.models.TArray.Model_TArrayC):
        cls = uproot.models.TH.Model_TH1C_v3
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayS):
        cls = uproot.models.TH.Model_TH1S_v3
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayI):
        cls = uproot.models.TH.Model_TH1I_v3
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayF):
        cls = uproot.models.TH.Model_TH1F_v3
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayD):
        cls = uproot.models.TH.Model_TH1D_v3
    else:
        raise TypeError(
            "no TH1* subclasses correspond to {0}".format(tarray_data.classname)
        )

    th1x = cls.empty()
    th1x._bases.append(th1)
    th1x._bases.append(tarray_data)

    th1x._deeply_writable = th1._deeply_writable

    return th1x


def to_TH2x(
    fName,
    fTitle,
    data,
    fEntries,
    fTsumw,
    fTsumw2,
    fTsumwx,
    fTsumwx2,
    fTsumwy,
    fTsumwy2,
    fTsumwxy,
    fSumw2,
    fXaxis,
    fYaxis,
    fZaxis=None,
    fScalefactor=1.0,
    fNcells=None,
    fBarOffset=0,
    fBarWidth=1000,
    fMaximum=-1111.0,
    fMinimum=-1111.0,
    fNormFactor=0.0,
    fContour=None,
    fOption="",
    fFunctions=None,
    fBufferSize=0,
    fBuffer=None,
    fBinStatErrOpt=0,
    fStatOverflows=2,
    fLineColor=602,
    fLineStyle=1,
    fLineWidth=1,
    fFillColor=0,
    fFillStyle=1001,
    fMarkerColor=1,
    fMarkerStyle=1,
    fMarkerSize=1.0,
):
    """
    Args:
        fName (None or str): Temporary name, will be overwritten by the writing
            process because Uproot's write syntax is ``file[name] = histogram``.
        fTitle (str): Real title of the histogram.
        data (numpy.ndarray or :doc:`uproot.models.TArray.Model_TArray`): Bin contents
            with first bin as underflow, last bin as overflow. The dtype of this array
            determines the return type of this function (TH2C, TH2D, TH2F, TH2I, or TH2S).
        fEntries (float): Number of entries. (https://root.cern.ch/doc/master/classTH1.html)
        fTsumw (float): Total Sum of weights.
        fTsumw2 (float): Total Sum of squares of weights.
        fTsumwx (float): Total Sum of weight*X.
        fTsumwx2 (float): Total Sum of weight*X*X.
        fTsumwy (float): Total Sum of weight*Y. (TH2 only: https://root.cern.ch/doc/master/classTH2.html)
        fTsumwy2 (float): Total Sum of weight*Y*Y. (TH2 only.)
        fTsumwxy (float): Total Sum of weight*X*Y. (TH2 only.)
        fSumw2 (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights.
        fXaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.to_TAxis`
            with ``fName="xaxis"`` and ``fTitle=""``.
        fYaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.to_TAxis`
            with ``fName="yaxis"`` and ``fTitle=""``.
        fZaxis (None or :doc:`uproot.models.TH.Model_TAxis_v10`): None generates a
            default for 1D and 2D histograms.
        fScalefactor (float): Scale factor. (TH2 only.)
        fNcells (None or int): Number of bins(1D), cells (2D) +U/Overflows. Computed
            from ``data`` if None.
        fBarOffset (int): (1000*offset) for bar charts or legos
        fBarWidth (int): (1000*width) for bar charts or legos
        fMaximum (float): Maximum value for plotting.
        fMinimum (float): Minimum value for plotting.
        fNormFactor (float): Normalization factor.
        fContour (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            to display contour levels. None generates an empty array.
        fOption (str or :doc:`uproot.models.TString.Model_TString`): Histogram options.
        fFunctions (None, list, or :doc:`uproot.models.TList.Model_TList`): ->Pointer to
            list of functions (fits and user). None generates an empty list.
        fBufferSize (None or int): fBuffer size. Computed from ``fBuffer`` if None.
        fBuffer (None or numpy.ndarray of numpy.float64): Buffer of entries accumulated
            before automatically choosing the binning. (Irrelevant for serialization?)
            None generates an empty array.
        fBinStatErrOpt (int): Option for bin statistical errors.
        fStatOverflows (int): Per object flag to use under/overflows in statistics.
        fLineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        fLineStyle (int): Line style.
        fLineWidth (int): Line width.
        fFillColor (int): Fill area color. (https://root.cern.ch/doc/master/classTAttFill.html)
        fFillStyle (int): Fill area style.
        fMarkerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        fMarkerStyle (int): Marker style.
        fMarkerSize (float): Marker size.

    This function is for developers to create TH2* objects that can be
    written to ROOT files, to implement conversion routines. The choice of
    TH2C, TH2D, TH2F, TH2I, or TH2S depends on the dtype of the ``data`` array.
    """
    th1x = to_TH1x(
        fName=fName,
        fTitle=fTitle,
        data=data,
        fEntries=fEntries,
        fTsumw=fTsumw,
        fTsumw2=fTsumw2,
        fTsumwx=fTsumwx,
        fTsumwx2=fTsumwx2,
        fSumw2=fSumw2,
        fXaxis=fXaxis,
        fYaxis=fYaxis,
        fZaxis=fZaxis,
        fNcells=fNcells,
        fBarOffset=fBarOffset,
        fBarWidth=fBarWidth,
        fMaximum=fMaximum,
        fMinimum=fMinimum,
        fNormFactor=fNormFactor,
        fContour=fContour,
        fOption=fOption,
        fFunctions=fFunctions,
        fBufferSize=fBufferSize,
        fBuffer=fBuffer,
        fBinStatErrOpt=fBinStatErrOpt,
        fStatOverflows=fStatOverflows,
        fLineColor=fLineColor,
        fLineStyle=fLineStyle,
        fLineWidth=fLineWidth,
        fFillColor=fFillColor,
        fFillStyle=fFillStyle,
        fMarkerColor=fMarkerColor,
        fMarkerStyle=fMarkerStyle,
        fMarkerSize=fMarkerSize,
    )

    th1 = th1x._bases[0]
    tarray_data = th1x._bases[1]

    th2 = uproot.models.TH.Model_TH2_v5.empty()
    th2._bases.append(th1)
    th2._members["fScalefactor"] = fScalefactor
    th2._members["fTsumwy"] = fTsumwy
    th2._members["fTsumwy2"] = fTsumwy2
    th2._members["fTsumwxy"] = fTsumwxy

    if isinstance(tarray_data, uproot.models.TArray.Model_TArrayC):
        cls = uproot.models.TH.Model_TH2C_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayS):
        cls = uproot.models.TH.Model_TH2S_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayI):
        cls = uproot.models.TH.Model_TH2I_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayF):
        cls = uproot.models.TH.Model_TH2F_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayD):
        cls = uproot.models.TH.Model_TH2D_v4
    else:
        raise TypeError(
            "no TH2* subclasses correspond to {0}".format(tarray_data.classname)
        )

    th2x = cls.empty()
    th2x._bases.append(th2)
    th2x._bases.append(tarray_data)

    th2._deeply_writable = th1._deeply_writable
    th2x._deeply_writable = th2._deeply_writable

    return th2x


def to_TH3x(
    fName,
    fTitle,
    data,
    fEntries,
    fTsumw,
    fTsumw2,
    fTsumwx,
    fTsumwx2,
    fTsumwy,
    fTsumwy2,
    fTsumwxy,
    fTsumwz,
    fTsumwz2,
    fTsumwxz,
    fTsumwyz,
    fSumw2,
    fXaxis,
    fYaxis,
    fZaxis,
    fNcells=None,
    fBarOffset=0,
    fBarWidth=1000,
    fMaximum=-1111.0,
    fMinimum=-1111.0,
    fNormFactor=0.0,
    fContour=None,
    fOption="",
    fFunctions=None,
    fBufferSize=0,
    fBuffer=None,
    fBinStatErrOpt=0,
    fStatOverflows=2,
    fLineColor=602,
    fLineStyle=1,
    fLineWidth=1,
    fFillColor=0,
    fFillStyle=1001,
    fMarkerColor=1,
    fMarkerStyle=1,
    fMarkerSize=1.0,
):
    """
    Args:
        fName (None or str): Temporary name, will be overwritten by the writing
            process because Uproot's write syntax is ``file[name] = histogram``.
        fTitle (str): Real title of the histogram.
        data (numpy.ndarray or :doc:`uproot.models.TArray.Model_TArray`): Bin contents
            with first bin as underflow, last bin as overflow. The dtype of this array
            determines the return type of this function (TH3C, TH3D, TH3F, TH3I, or TH3S).
        fEntries (float): Number of entries. (https://root.cern.ch/doc/master/classTH1.html)
        fTsumw (float): Total Sum of weights.
        fTsumw2 (float): Total Sum of squares of weights.
        fTsumwx (float): Total Sum of weight*X.
        fTsumwx2 (float): Total Sum of weight*X*X.
        fTsumwy (float): Total Sum of weight*Y. (TH3 only: https://root.cern.ch/doc/master/classTH3.html)
        fTsumwy2 (float): Total Sum of weight*Y*Y. (TH3 only.)
        fTsumwxy (float): Total Sum of weight*X*Y. (TH3 only.)
        fTsumwz (float): Total Sum of weight*Z. (TH3 only.)
        fTsumwz2 (float): Total Sum of weight*Z*Z. (TH3 only.)
        fTsumwxz (float): Total Sum of weight*X*Z. (TH3 only.)
        fTsumwyz (float): Total Sum of weight*Y*Z. (TH3 only.)
        fSumw2 (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights.
        fXaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.to_TAxis`
            with ``fName="xaxis"`` and ``fTitle=""``.
        fYaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.to_TAxis`
            with ``fName="yaxis"`` and ``fTitle=""``.
        fZaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.to_TAxis`
            with ``fName="zaxis"`` and ``fTitle=""``.
        fNcells (None or int): Number of bins(1D), cells (2D) +U/Overflows. Computed
            from ``data`` if None.
        fBarOffset (int): (1000*offset) for bar charts or legos
        fBarWidth (int): (1000*width) for bar charts or legos
        fMaximum (float): Maximum value for plotting.
        fMinimum (float): Minimum value for plotting.
        fNormFactor (float): Normalization factor.
        fContour (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            to display contour levels. None generates an empty array.
        fOption (str or :doc:`uproot.models.TString.Model_TString`): Histogram options.
        fFunctions (None, list, or :doc:`uproot.models.TList.Model_TList`): ->Pointer to
            list of functions (fits and user). None generates an empty list.
        fBufferSize (None or int): fBuffer size. Computed from ``fBuffer`` if None.
        fBuffer (None or numpy.ndarray of numpy.float64): Buffer of entries accumulated
            before automatically choosing the binning. (Irrelevant for serialization?)
            None generates an empty array.
        fBinStatErrOpt (int): Option for bin statistical errors.
        fStatOverflows (int): Per object flag to use under/overflows in statistics.
        fLineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        fLineStyle (int): Line style.
        fLineWidth (int): Line width.
        fFillColor (int): Fill area color. (https://root.cern.ch/doc/master/classTAttFill.html)
        fFillStyle (int): Fill area style.
        fMarkerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        fMarkerStyle (int): Marker style.
        fMarkerSize (float): Marker size.

    This function is for developers to create TH3* objects that can be
    written to ROOT files, to implement conversion routines. The choice of
    TH3C, TH3D, TH3F, TH3I, or TH3S depends on the dtype of the ``data`` array.
    """
    th1x = to_TH1x(
        fName=fName,
        fTitle=fTitle,
        data=data,
        fEntries=fEntries,
        fTsumw=fTsumw,
        fTsumw2=fTsumw2,
        fTsumwx=fTsumwx,
        fTsumwx2=fTsumwx2,
        fSumw2=fSumw2,
        fXaxis=fXaxis,
        fYaxis=fYaxis,
        fZaxis=fZaxis,
        fNcells=fNcells,
        fBarOffset=fBarOffset,
        fBarWidth=fBarWidth,
        fMaximum=fMaximum,
        fMinimum=fMinimum,
        fNormFactor=fNormFactor,
        fContour=fContour,
        fOption=fOption,
        fFunctions=fFunctions,
        fBufferSize=fBufferSize,
        fBuffer=fBuffer,
        fBinStatErrOpt=fBinStatErrOpt,
        fStatOverflows=fStatOverflows,
        fLineColor=fLineColor,
        fLineStyle=fLineStyle,
        fLineWidth=fLineWidth,
        fFillColor=fFillColor,
        fFillStyle=fFillStyle,
        fMarkerColor=fMarkerColor,
        fMarkerStyle=fMarkerStyle,
        fMarkerSize=fMarkerSize,
    )

    th1 = th1x._bases[0]
    tarray_data = th1x._bases[1]

    tatt3d = uproot.models.TAtt.Model_TAtt3D_v1.empty()
    tatt3d._deeply_writable = True

    th3 = uproot.models.TH.Model_TH3_v6.empty()
    th3._bases.append(th1)
    th3._bases.append(tatt3d)
    th3._members["fTsumwy"] = fTsumwy
    th3._members["fTsumwy2"] = fTsumwy2
    th3._members["fTsumwxy"] = fTsumwxy
    th3._members["fTsumwz"] = fTsumwz
    th3._members["fTsumwz2"] = fTsumwz2
    th3._members["fTsumwxz"] = fTsumwxz
    th3._members["fTsumwyz"] = fTsumwyz

    if isinstance(tarray_data, uproot.models.TArray.Model_TArrayC):
        cls = uproot.models.TH.Model_TH3C_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayS):
        cls = uproot.models.TH.Model_TH3S_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayI):
        cls = uproot.models.TH.Model_TH3I_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayF):
        cls = uproot.models.TH.Model_TH3F_v4
    elif isinstance(tarray_data, uproot.models.TArray.Model_TArrayD):
        cls = uproot.models.TH.Model_TH3D_v4
    else:
        raise TypeError(
            "no TH3* subclasses correspond to {0}".format(tarray_data.classname)
        )

    th3x = cls.empty()
    th3x._bases.append(th3)
    th3x._bases.append(tarray_data)

    th3._deeply_writable = th1._deeply_writable
    th3x._deeply_writable = th3._deeply_writable

    return th3x


def to_TProfile(
    fName,
    fTitle,
    data,
    fEntries,
    fTsumw,
    fTsumw2,
    fTsumwx,
    fTsumwx2,
    fTsumwy,
    fTsumwy2,
    fSumw2,
    fBinEntries,
    fBinSumw2,
    fXaxis,
    fYaxis=None,
    fZaxis=None,
    fYmin=0.0,
    fYmax=0.0,
    fErrorMode=0,
    fNcells=None,
    fBarOffset=0,
    fBarWidth=1000,
    fMaximum=-1111.0,
    fMinimum=-1111.0,
    fNormFactor=0.0,
    fContour=None,
    fOption="",
    fFunctions=None,
    fBufferSize=0,
    fBuffer=None,
    fBinStatErrOpt=0,
    fStatOverflows=2,
    fLineColor=602,
    fLineStyle=1,
    fLineWidth=1,
    fFillColor=0,
    fFillStyle=1001,
    fMarkerColor=1,
    fMarkerStyle=1,
    fMarkerSize=1.0,
):
    """
    Args:
        fName (None or str): Temporary name, will be overwritten by the writing
            process because Uproot's write syntax is ``file[name] = histogram``.
        fTitle (str): Real title of the histogram.
        data (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Bin
            contents with first bin as underflow, last bin as overflow. The dtype of this array
            must be float64.
        fEntries (float): Number of entries. (https://root.cern.ch/doc/master/classTH1.html)
        fTsumw (float): Total Sum of weights.
        fTsumw2 (float): Total Sum of squares of weights.
        fTsumwx (float): Total Sum of weight*X.
        fTsumwx2 (float): Total Sum of weight*X*X.
        fTsumwy (float): Total Sum of weight*Y. (TProfile only: https://root.cern.ch/doc/master/classTProfile.html)
        fTsumwy2 (float): Total Sum of weight*Y*Y. (TH3 only.)
        fSumw2 (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights.
        fBinEntries (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Number
            of entries per bin. (TProfile only.)
        fBinSumw2 (numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            of sum of squares of weights per bin. (TProfile only.)
        fXaxis (:doc:`uproot.models.TH.Model_TAxis_v10`): Use :doc:`uproot.writing.to_TAxis`
            with ``fName="xaxis"`` and ``fTitle=""``.
        fYaxis (None or :doc:`uproot.models.TH.Model_TAxis_v10`): None generates a
            default for 1D histograms.
        fZaxis (None or :doc:`uproot.models.TH.Model_TAxis_v10`): None generates a
            default for 1D and 2D histograms.
        fYmin (float): Lower limit in Y (if set). (TProfile only.)
        fYmax (float): Upper limit in Y (if set). (TProfile only.)
        fErrorMode (int): Option to compute errors. (TProfile only.)
        fNcells (None or int): Number of bins(1D), cells (2D) +U/Overflows. Computed
            from ``data`` if None.
        fBarOffset (int): (1000*offset) for bar charts or legos
        fBarWidth (int): (1000*width) for bar charts or legos
        fMaximum (float): Maximum value for plotting.
        fMinimum (float): Minimum value for plotting.
        fNormFactor (float): Normalization factor.
        fContour (None or numpy.ndarray of numpy.float64 or :doc:`uproot.models.TArray.Model_TArrayD`): Array
            to display contour levels. None generates an empty array.
        fOption (str or :doc:`uproot.models.TString.Model_TString`): Histogram options.
        fFunctions (None, list, or :doc:`uproot.models.TList.Model_TList`): ->Pointer to
            list of functions (fits and user). None generates an empty list.
        fBufferSize (None or int): fBuffer size. Computed from ``fBuffer`` if None.
        fBuffer (None or numpy.ndarray of numpy.float64): Buffer of entries accumulated
            before automatically choosing the binning. (Irrelevant for serialization?)
            None generates an empty array.
        fBinStatErrOpt (int): Option for bin statistical errors.
        fStatOverflows (int): Per object flag to use under/overflows in statistics.
        fLineColor (int): Line color. (https://root.cern.ch/doc/master/classTAttLine.html)
        fLineStyle (int): Line style.
        fLineWidth (int): Line width.
        fFillColor (int): Fill area color. (https://root.cern.ch/doc/master/classTAttFill.html)
        fFillStyle (int): Fill area style.
        fMarkerColor (int): Marker color. (https://root.cern.ch/doc/master/classTAttMarker.html)
        fMarkerStyle (int): Marker style.
        fMarkerSize (float): Marker size.

    This function is for developers to create TProfile objects that can be
    written to ROOT files, to implement conversion routines.
    """
    th1x = to_TH1x(
        fName=fName,
        fTitle=fTitle,
        data=data,
        fEntries=fEntries,
        fTsumw=fTsumw,
        fTsumw2=fTsumw2,
        fTsumwx=fTsumwx,
        fTsumwx2=fTsumwx2,
        fSumw2=fSumw2,
        fXaxis=fXaxis,
        fYaxis=fYaxis,
        fZaxis=fZaxis,
        fNcells=fNcells,
        fBarOffset=fBarOffset,
        fBarWidth=fBarWidth,
        fMaximum=fMaximum,
        fMinimum=fMinimum,
        fNormFactor=fNormFactor,
        fContour=fContour,
        fOption=fOption,
        fFunctions=fFunctions,
        fBufferSize=fBufferSize,
        fBuffer=fBuffer,
        fBinStatErrOpt=fBinStatErrOpt,
        fStatOverflows=fStatOverflows,
        fLineColor=fLineColor,
        fLineStyle=fLineStyle,
        fLineWidth=fLineWidth,
        fFillColor=fFillColor,
        fFillStyle=fFillStyle,
        fMarkerColor=fMarkerColor,
        fMarkerStyle=fMarkerStyle,
        fMarkerSize=fMarkerSize,
    )
    if not isinstance(th1x, uproot.models.TH.Model_TH1D_v3):
        raise TypeError("TProfile requires an array of float64 (TArrayD)")

    if isinstance(fBinEntries, uproot.models.TArray.Model_TArray):
        tarray_fBinEntries = fBinEntries
    else:
        tarray_fBinEntries = to_TArray(fBinEntries)
    if not isinstance(tarray_fBinEntries, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fBinEntries must be an array of float64 (TArrayD)")

    if isinstance(fBinSumw2, uproot.models.TArray.Model_TArray):
        tarray_fBinSumw2 = fBinSumw2
    else:
        tarray_fBinSumw2 = to_TArray(fBinSumw2)
    if not isinstance(tarray_fBinSumw2, uproot.models.TArray.Model_TArrayD):
        raise TypeError("fBinSumw2 must be an array of float64 (TArrayD)")

    tprofile = uproot.models.TH.Model_TProfile_v7.empty()
    tprofile._bases.append(th1x)
    tprofile._members["fBinEntries"] = tarray_fBinEntries
    tprofile._members["fErrorMode"] = fErrorMode
    tprofile._members["fYmin"] = fYmin
    tprofile._members["fYmax"] = fYmax
    tprofile._members["fTsumwy"] = fTsumwy
    tprofile._members["fTsumwy2"] = fTsumwy2
    tprofile._members["fBinSumw2"] = tarray_fBinSumw2

    tprofile._deeply_writable = th1x._deeply_writable

    return tprofile
