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

            raw_data = chunk.raw_data
            if hasattr(raw_data, "tobytes"):
                raw_data = raw_data.tobytes()
            else:
                raw_data = raw_data.tostring()

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
        if obj.writable:
            return obj
        else:
            raise NotImplementedError(
                "this ROOT type is not writable: " + obj.classname
            )

    elif uproot._util.isstr(obj):
        return to_TObjString(obj)

    else:
        raise TypeError(
            "unrecognized type cannot be written to a ROOT file: " + type(obj).__name__
        )


def to_TObjString(string):
    """
    FIXME: docstring
    """
    out = uproot.models.TObjString.Model_TObjString(str(string))
    out._cursor = None
    out._parent = None
    out._members = {}
    out._bases = (uproot.models.TObject.Model_TObject(),)
    out._num_bytes = len(string) + (1 if len(string) < 255 else 5) + 16
    out._instance_version = 1
    return out
