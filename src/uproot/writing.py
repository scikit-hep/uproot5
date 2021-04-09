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
import uproot.const
import uproot.reading
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
    "initial_streamers_bytes": 256,
    "uuid_version": 1,
    "uuid_function": uuid.uuid1,
}
recreate.defaults = create.defaults
update.defaults = create.defaults


class WritableFile(object):
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

    def __repr__(self):
        return "<WritableFile {0} at 0x{1:012x}>".format(repr(self.file_path), id(self))

    @property
    def sink(self):
        return self._sink

    @property
    def file_path(self):
        return self._sink.file_path

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
    def root_directory(self):
        return WritableDirectory((), self, self._cascading.rootdirectory)

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
    def file(self):
        return self._file

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
