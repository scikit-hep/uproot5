# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import os
import uuid

import uproot._util
import uproot._writing
import uproot.compression
import uproot.deserialization
import uproot.exceptions
import uproot.sink.file


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
    uuid_version = options.pop("uuid_version", create.defaults["uuid_version"])
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
        uuid_version,
        uuid_function,
    )
    return WritableFile(
        sink, cascading, initial_directory_bytes, uuid_version, uuid_function
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
    uuid_version = options.pop("uuid_version", create.defaults["uuid_version"])
    uuid_function = options.pop("uuid_function", create.defaults["uuid_function"])
    if len(options) != 0:
        raise TypeError(
            "unrecognized options for uproot.update: "
            + ", ".join(repr(x) for x in options)
        )

    cascading = uproot._writing.update_existing(
        sink,
        initial_directory_bytes,
        uuid_version,
        uuid_function,
    )
    return WritableFile(
        sink, cascading, initial_directory_bytes, uuid_version, uuid_function
    ).root_directory


create.defaults = {
    "compression": uproot.compression.ZLIB(1),
    "initial_directory_bytes": 256,
    "initial_streamers_bytes": 1024,  # 256,
    "uuid_version": 1,
    "uuid_function": uuid.uuid1,
}
recreate.defaults = create.defaults
update.defaults = create.defaults


class WritableFile(uproot.reading.CommonFileMethods):
    """
    FIXME: docstring
    """

    def __init__(
        self, sink, cascading, initial_directory_bytes, uuid_version, uuid_function
    ):
        self._sink = sink
        self._cascading = cascading
        self._initial_directory_bytes = initial_directory_bytes
        self._uuid_version = uuid_version
        self._uuid_function = uuid_function

        self._file_path = sink.file_path
        self._fVersion = self._cascading.fileheader.version
        self._fBEGIN = self._cascading.fileheader.begin
        self._fNbytesName = self._cascading.fileheader.begin_num_bytes
        self._fUUID_version = self._cascading.fileheader.uuid_version
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
    def uuid_version(self):
        return self._uuid_version

    @uuid_version.setter
    def uuid_version(self, value):
        self._uuid_version = value

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
            "uuid_version": uuid_version,
            "uuid_function": uuid_function,
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

        if key.classname.string == "TDirectory":
            return self._subdir(key)

        else:





            raise Exception

    def _subdir(self, key):
        raw_bytes = self._file.sink.read(
            key.seek_location,
            key.num_bytes + uproot.reading._directory_format_big.size + 18,
        )
        directory_key = uproot._writing.Key.deserialize(
            raw_bytes, key.seek_location, self._file.sink.in_path
        )
        position = key.seek_location + directory_key.num_bytes

        directory_header = uproot._writing.DirectoryHeader.deserialize(
            raw_bytes[position - key.seek_location :], position, self._file.sink.in_path
        )
        assert directory_header.begin_location == key.seek_location
        assert directory_header.parent_location == self._cascading.key.location

        name = key.name.string

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
                directory_datakey.num_bytes + self.file._initial_directory_bytes
            )
            directory_datakey.location = self._cascading.freesegments.allocate(
                requested_num_bytes
            )
            might_be_slightly_more = requested_num_bytes - directory_datakey.num_bytes
            directory_data = uproot._writing.DirectoryData(
                directory_datakey.location + directory_datakey.num_bytes,
                might_be_slightly_more,
                [],
            )

            directory_datakey.uncompressed_bytes = directory_data.allocation
            directory_datakey.compressed_bytes = directory_datakey.uncompressed_bytes

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

            self._file.sink.set_file_length(self._cascading.freesegments.fileheader.end)
            self._file.sink.flush()

            return WritableDirectory(self._path + (name,), self._file, subdirectory)

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

            return WritableDirectory(self._path + (name,), self._file, subdirectory)

    def mkdir(self, name):
        return WritableDirectory(
            self._path + (name,),
            self._file,
            self._cascading.add_directory(
                self._file.sink,
                name,
                self._file.initial_directory_bytes,
                self._file.uuid_version,
                self._file.uuid_function(),
            ),
        )
