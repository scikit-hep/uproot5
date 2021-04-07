# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import uuid

import uproot._util
import uproot.compression
import uproot.const
import uproot.reading
import uproot.sink.file

# def create(file_path, **options):
#     """
#     FIXME: docstring
#     """
#     file_path = uproot._util.regularize_path(file_path)
#     if uproot._util.isstr(file_path):
#         if os.path.exists(file_path):
#             raise OSError(
#                 "path exists and refusing to overwrite (use 'uproot.recreate' to "
#                 "overwrite)\n\nin path {0}".format(file_path)
#             )
#         file = uproot.sink.file.FileSink()
#     else:
#         file = uproot.sink.file.FileSink.from_object(file_path)

#     compression = options.pop("compression", update.defaults["compression"])
#     initial_directory_bytes = options.pop(
#         "initial_directory_bytes", create.defaults["initial_directory_bytes"]
#     )
#     uuid_version = options.pop("uuid_version", create.defaults["uuid_version"])
#     uuid_function = options.pop("uuid", create.defaults["uuid"])
#     if len(options) != 0:
#         raise TypeError("unrecognized options: " + ", ".join(repr(x) for x in options))

#     _create_empty_root(
#         file, compression, initial_directory_bytes, uuid_version, uuid_function
#     )
#     return WritableFile(file, compression, initial_directory_bytes)


# create.defaults = {
defaults = {
    "compression": uproot.compression.ZLIB(1),
    "initial_directory_bytes": 256,
    "initial_streamers_bytes": 256,
    "uuid_version": 1,
    "uuid_function": uuid.uuid1,
}


# def recreate(file_path, **options):
#     """
#     FIXME: docstring
#     """
#     file_path = uproot._util.regularize_path(file_path)
#     if uproot._util.isstr(file_path):
#         file = uproot.sink.file.FileSink()
#     else:
#         file = uproot.sink.file.FileSink.from_object(file_path)

#     compression = options.pop("compression", update.defaults["compression"])
#     initial_directory_bytes = options.pop(
#         "initial_directory_bytes", recreate.defaults["initial_directory_bytes"]
#     )
#     uuid_version = options.pop("uuid_version", create.defaults["uuid_version"])
#     uuid_function = options.pop("uuid", create.defaults["uuid"])
#     if len(options) != 0:
#         raise TypeError("unrecognized options: " + ", ".join(repr(x) for x in options))

#     _create_empty_root(
#         file, compression, initial_directory_bytes, uuid_version, uuid_function
#     )
#     return WritableFile(file, compression, initial_directory_bytes)


# recreate.defaults = {
#     "compression": uproot.compression.ZLIB(1),
#     "initial_directory_bytes": 256,
#     "initial_streamers_bytes": 256,
#     "uuid_version": 1,
#     "uuid_function": uuid.uuid1,
# }


# def update(file_path, **options):
#     """
#     FIXME: docstring
#     """
#     file_path = uproot._util.regularize_path(file_path)
#     if uproot._util.isstr(file_path):
#         file = uproot.sink.file.FileSink()
#     else:
#         file = uproot.sink.file.FileSink.from_object(file_path)

#     compression = options.pop("compression", update.defaults["compression"])
#     initial_directory_bytes = options.pop(
#         "initial_directory_bytes", update.defaults["initial_directory_bytes"]
#     )
#     uuid_version = options.pop("uuid_version", create.defaults["uuid_version"])
#     uuid_function = options.pop("uuid", create.defaults["uuid"])
#     if len(options) != 0:
#         raise TypeError("unrecognized options: " + ", ".join(repr(x) for x in options))

#     if file.read_raw(4) != b"root":
#         _create_empty_root(
#             file, compression, initial_directory_bytes, uuid_version, uuid_function
#         )
#     return WritableFile(file, compression, initial_directory_bytes)


# update.defaults = {
#     "compression": uproot.compression.ZLIB(1),
#     "initial_directory_bytes": 256,
#     "initial_streamers_bytes": 256,
#     "uuid_version": 1,
#     "uuid_function": uuid.uuid1,
# }
