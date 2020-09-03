# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the entry-point for opening a file, :doc:`uproot4.reading.open`, and
the classes that are too fundamental to be models:
:doc:`uproot4.reading.ReadOnlyFile` (``TFile``),
:doc:`uproot4.reading.ReadOnlyDirectory` (``TDirectory`` or ``TDirectoryFile``),
and :doc:`uproot4.reading.ReadOnlyKey` (``TKey``).
"""

from __future__ import absolute_import

import sys
import struct
import uuid

try:
    from collections.abc import Mapping
    from collections.abc import MutableMapping
except ImportError:
    from collections import Mapping
    from collections import MutableMapping

import uproot4.compression
import uproot4.cache
import uproot4.source.cursor
import uproot4.source.chunk
import uproot4.source.http
import uproot4.source.xrootd
import uproot4.streamers
import uproot4.model
import uproot4.behaviors.TBranch
import uproot4._util
from uproot4._util import no_filter


def open(
    path,
    object_cache=100,
    array_cache="100 MB",
    custom_classes=None,
    **options  # NOTE: a comma after **options breaks Python 2
):
    """
    Args:
        path (str or ``pathlib.Path``): The filesystem path or remote URL of
            the file to open. If a string, it may be followed by a colon (``:``)
            and an object path within the ROOT file, to return an object,
            rather than a file. Path objects are interpreted strictly as
            filesystem paths or URLs.
            Examples: ``"rel/file.root"``, ``"C:\abs\file.root"``,
            ``"http://where/what.root"``, ``"rel/file.root:tdirectory/ttree"``,
            ``Path("rel:/file.root")``, ``Path("/abs/path:stuff.root")``
        object_cache (None, MutableMapping, or int): Cache of objects drawn
            from ROOT directories (e.g histograms, TTrees, other directories);
            if None, do not use a cache; if an int, create a new cache of this
            size.
        array_cache (None, MutableMapping, or memory size): Cache of arrays
            drawn from ``TTrees``; if None, do not use a cache; if a memory
            size, create a new cache of this size.
        custom_classes (None or MutableMapping): If None, classes come from
            uproot4.classes; otherwise, a container of class definitions that
            is both used to fill with new classes and search for dependencies.
        options: See below.

    Opens a ROOT file, possibly through a remote protocol.

    If an object path is given, the return type of this function can be anything
    that can be extracted from a ROOT file (subclass of
    :doc:`uproot4.model.Model`).

    If an object path is not given, the return type is a
    :doc:`uproot4.reading.ReadOnlyDirectory` *and not*
    :doc:`uproot4.reading.ReadOnlyFile`. ROOT objects can be extracted from a
    :doc:`uproot4.reading.ReadOnlyDirectory` but not a
    :doc:`uproot4.reading.ReadOnlyFile`.

    Options (type; default):

    * file_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.file.MemmapSource`)
    * xrootd_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.xrootd.XRootDSource`)
    * http_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.http.HTTPSource`)
    * timeout (float for HTTP, int for XRootD; 30)
    * max_num_elements (None or int; None)
    * num_workers (int; 1)
    * num_fallback_workers (int; 10)
    * begin_chunk_size (memory_size; 512)
    * minimal_ttree_metadata (bool; True)

    Any object derived from a ROOT file is a context manager (works in Python's
    ``with`` statement) that closes the file when exiting the ``with`` block.
    Therefore, the :doc:`uproot4.reading.open` function can and usually should
    be used in a ``with`` statement to clean up file handles and threads
    associated with open files:

    .. code-block:: python

        with uproot4.open("/path/to/file.root:path/to/histogram") as h:
            h.to_hist().plot()

        # file is now closed, even if an exception was raised in the block

    Alternative entry points:

    * :doc:`uproot4.reading.open` (this function): opens one file to read any
      of its objects.
    * :doc:`uproot4.behaviors.TBranch.iterate`: iterates through chunks of
      contiguous entries in ``TTrees``.
    * :doc:`uproot4.behaviors.TBranch.concatenate`: returns a single concatenated
      array from ``TTrees``.
    * :doc:`uproot4.behaviors.TBranch.lazy`: returns a lazily read array from
      ``TTrees``.
    """

    if isinstance(path, dict) and len(path) == 1:
        ((file_path, object_path),) = path.items()

    elif uproot4._util.isstr(path):
        file_path, object_path = uproot4._util.file_object_path_split(path)

    else:
        file_path = path
        object_path = None

    file_path = uproot4._util.regularize_path(file_path)

    if not uproot4._util.isstr(file_path):
        raise ValueError(
            "'path' must be a string, pathlib.Path, or a length-1 dict of "
            "{{file_path: object_path}}, not {0}".format(repr(path))
        )

    file = ReadOnlyFile(
        file_path,
        object_cache=object_cache,
        array_cache=array_cache,
        custom_classes=custom_classes,
        **options  # NOTE: a comma after **options breaks Python 2
    )

    if object_path is None:
        return file.root_directory
    else:
        return file.root_directory[object_path]


open.defaults = {
    "file_handler": uproot4.source.file.MemmapSource,
    "xrootd_handler": uproot4.source.xrootd.XRootDSource,
    "http_handler": uproot4.source.http.HTTPSource,
    "timeout": 30,
    "max_num_elements": None,
    "num_workers": 1,
    "num_fallback_workers": 10,
    "begin_chunk_size": 512,
    "minimal_ttree_metadata": True,
}


must_be_attached = [
    "TROOT",
    "TDirectory",
    "TDirectoryFile",
    "RooWorkspace::WSDir",
    "TTree",
    "TChain",
    "TProofChain",
    "THbookTree",
    "TNtuple",
    "TNtupleD",
    "TTreeSQL",
]


class CommonFileMethods(object):
    """
    Abstract class for :doc:`uproot4.reading.ReadOnlyFile` and
    :doc:`uproot4.reading.DetachedFile`. The latter is a placeholder for file
    information, such as the :doc:`uproot4.reading.CommonFileMethods.file_path`
    used in many error messages, without holding a reference to the active
    :doc:`uproot4.source.chunk.Source`.

    This allows the file to be closed and deleted while objects that were read
    from it still exist. Also, only objects that hold detached file references,
    rather than active ones, can be pickled.

    The (unpickleable) objects that must hold a reference to an active
    :doc:`uproot4.reading.ReadOnlyFile` are listed by C++ (decoded) classname
    in ``uproot4.must_be_attached``.
    """

    @property
    def file_path(self):
        """
        The original path to the file (converted to ``str`` if it was originally
        a ``pathlib.Path``).
        """
        return self._file_path

    @property
    def options(self):
        """
        The dict of ``options`` originally passed to the
        :doc:`uproot4.reading.ReadOnlyFile` constructor.
        """
        return self._options

    @property
    def root_version(self):
        """
        Version of ROOT used to write the file as a string.

        See :doc:`uproot4.reading.CommonFileMethods.root_version_tuple` and
        :doc:`uproot4.reading.CommonFileMethods.fVersion`.
        """
        return "{0}.{1:02d}/{2:02d}".format(*self.root_version_tuple)

    @property
    def root_version_tuple(self):
        """
        Version of ROOT used to write teh file as a tuple.

        See :doc:`uproot4.reading.CommonFileMethods.root_version` and
        :doc:`uproot4.reading.CommonFileMethods.fVersion`.
        """
        version = self._fVersion
        if version >= 1000000:
            version -= 1000000

        major = version // 10000
        version %= 10000
        minor = version // 100
        version %= 100

        return major, minor, version

    @property
    def is_64bit(self):
        """
        True if the ROOT file is 64-bit ready; False otherwise.

        A file that is larger than 4 GiB must be 64-bit ready, though any file
        might be. This refers to seek points like
        :doc:`uproot4.reading.ReadOnlyFile.fSeekFree` being 64-bit integers,
        rather than 32-bit.

        Note that a file being 64-bit is distinct from a ``TDirectory`` being
        64-bit; see :doc:`uproot4.reading.ReadOnlyDirectory.is_64bit`.
        """
        return self._fVersion >= 1000000

    @property
    def compression(self):
        """
        A :doc:`uproot4.compression.Compression` object describing the
        compression setting for the ROOT file.

        Note that different objects (even different ``TBranches`` within a
        ``TTree``) can be compressed differently, so this file-level
        compression is only a strong hint of how the objects are likely to
        be compressed.

        For some versions of ROOT ``TStreamerInfo`` is always compressed with
        :doc:`uproot4.compression.ZLIB`, even if the compression is set to a
        different algorithm.

        See :doc:`uproot4.reading.CommonFileMethods.fCompress`.
        """
        return uproot4.compression.Compression.from_code(self._fCompress)

    @property
    def hex_uuid(self):
        """
        The unique identifier (UUID) of the ROOT file expressed as a hexadecimal
        string.

        See :doc:`uproot4.reading.CommonFileMethods.uuid` and
        :doc:`uproot4.reading.CommonFileMethods.fUUID`.
        """
        if uproot4._util.py2:
            out = "".join("{0:02x}".format(ord(x)) for x in self._fUUID)
        else:
            out = "".join("{0:02x}".format(x) for x in self._fUUID)
        return "-".join([out[0:8], out[8:12], out[12:16], out[16:20], out[20:32]])

    @property
    def uuid(self):
        """
        The unique identifier (UUID) of the ROOT file expressed as a Python
        ``uuid.UUID`` object.

        See :doc:`uproot4.reading.CommonFileMethods.hex_uuid` and
        :doc:`uproot4.reading.CommonFileMethods.fUUID`.
        """
        return uuid.UUID(self.hex_uuid.replace("-", ""))

    @property
    def fVersion(self):
        """
        Raw version information for the ROOT file; this number is used to derive
        :doc:`uproot4.reading.CommonFileMethods.root_version`,
        :doc:`uproot4.reading.CommonFileMethods.root_version_tuple`, and
        :doc:`uproot4.reading.CommonFileMethods.is_64bit`.
        """
        return self._fVersion

    @property
    def fBEGIN(self):
        """
        The seek point (int) for the first data record, past the TFile header.

        Usually 100.
        """
        return self._fBEGIN

    @property
    def fEND(self):
        """
        The seek point (int) to the last free word at the end of the ROOT file.
        """
        return self._fEND

    @property
    def fSeekFree(self):
        """
        The seek point (int) to the ``TFree`` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._fSeekFree

    @property
    def fNbytesFree(self):
        """
        The number of bytes in the ``TFree` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._fNbytesFree

    @property
    def nfree(self):
        """
        The number of objects in the ``TFree`` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._nfree

    @property
    def fNbytesName(self):
        """
        The number of bytes in the filename (``TNamed``) that is embedded in
        the ROOT file.
        """
        return self._fNbytesName

    @property
    def fUnits(self):
        """
        Number of bytes in the serialization of file seek points.

        Usually 4 or 8.
        """
        return self._fUnits

    @property
    def fCompress(self):
        """
        The raw integer describing the compression setting for the ROOT file.

        Note that different objects (even different ``TBranches`` within a
        ``TTree``) can be compressed differently, so this file-level
        compression is only a strong hint of how the objects are likely to
        be compressed.

        For some versions of ROOT ``TStreamerInfo`` is always compressed with
        :doc:`uproot4.compression.ZLIB`, even if the compression is set to a
        different algorithm.

        See :doc:`uproot4.reading.CommonFileMethods.compression`.
        """
        return self._fCompress

    @property
    def fSeekInfo(self):
        """
        The seek point (int) to the ``TStreamerInfo`` data, where
        :doc:`uproot4.reading.ReadOnlyFile.streamers` are located.
        """
        return self._fSeekInfo

    @property
    def fNbytesInfo(self):
        """
        The number of bytes in the ``TStreamerInfo`` data, where
        :doc:`uproot4.reading.ReadOnlyFile.streamers` are located.
        """
        return self._fNbytesInfo

    @property
    def fUUID(self):
        """
        The unique identifier (UUID) of the ROOT file as a raw bytestring
        (Python ``bytes``).

        See :doc:`uproot4.reading.CommonFileMethods.hex_uuid` and
        :doc:`uproot4.reading.CommonFileMethods.uuid`.
        """
        return self._fUUID


class DetachedFile(CommonFileMethods):
    """
    Args:
        file (:doc:`uproot4.reading.ReadOnlyFile`): The active file object to
            convert into a detached file.

    A placeholder for a :doc:`uproot4.reading.ReadOnlyFile` with useful
    information, such as the :doc:`uproot4.reading.CommonFileMethods.file_path`
    used in many error messages, without holding a reference to the active
    :doc:`uproot4.source.chunk.Source`.

    This allows the file to be closed and deleted while objects that were read
    from it still exist. Also, only objects that hold detached file references,
    rather than active ones, can be pickled.

    The (unpickleable) objects that must hold a reference to an active
    :doc:`uproot4.reading.ReadOnlyFile` are listed by C++ (decoded) classname
    in ``uproot4.must_be_attached``.
    """

    def __init__(self, file):
        self._file_path = file._file_path
        self._options = file._options
        self._fVersion = file._fVersion
        self._fBEGIN = file._fBEGIN
        self._fEND = file._fEND
        self._fSeekFree = file._fSeekFree
        self._fNbytesFree = file._fNbytesFree
        self._nfree = file._nfree
        self._fNbytesName = file._fNbytesName
        self._fUnits = file._fUnits
        self._fCompress = file._fCompress
        self._fSeekInfo = file._fSeekInfo
        self._fNbytesInfo = file._fNbytesInfo
        self._fUUID_version = file._fUUID_version
        self._fUUID = file._fUUID


_file_header_fields_small = struct.Struct(">4siiiiiiiBiiiH16s")
_file_header_fields_big = struct.Struct(">4siiqqiiiBiqiH16s")


class ReadOnlyFile(CommonFileMethods):
    """
    Args:
        file_path (str or ``pathlib.Path``): The filesystem path or remote URL
            of the file to open. Unlike :doc:`uproot4.reading.open`, it cannot
            be followed by a colon (``:``) and an object path within the ROOT
            file.
        object_cache (None, MutableMapping, or int): Cache of objects drawn
            from ROOT directories (e.g histograms, TTrees, other directories);
            if None, do not use a cache; if an int, create a new cache of this
            size.
        array_cache (None, MutableMapping, or memory size): Cache of arrays
            drawn from TTrees; if None, do not use a cache; if a memory size,
            create a new cache of this size.
        custom_classes (None or MutableMapping): If None, classes come from
            uproot4.classes; otherwise, a container of class definitions that
            is both used to fill with new classes and search for dependencies.
        options: See below.

    Handle to an open ROOT file, the way to access data in ``TDirectories``
    (:doc:`uproot4.reading.ReadOnlyDirectory`) and create new classes from
    ``TStreamerInfo`` (:doc:`uproot4.reading.ReadOnlyFile.streamers`).

    All objects derived from ROOT files have a pointer back to the file,
    though this is a :doc:`uproot4.reading.DetachedFile` (no active connection,
    cannot read more data) if the object's :doc:`uproot4.model.Model.classname`
    is not in ``uproot4.reading.must_be_attached``: objects that can read
    more data and need to have an active connection (like ``TTree``,
    ``TBranch``, and ``TDirectory``).

    Note that a :doc:`uproot4.reading.ReadOnlyFile` can't be directly used to
    extract objects. To read data, use the :doc:`uproot4.reading.ReadOnlyDirectory`
    returned by :doc:`uproot4.reading.ReadOnlyFile.root_directory`. This is why
    :doc:`uproot4.reading.open` returns a :doc:`uproot4.reading.ReadOnlyDirectory`
    and not a :doc:`uproot4.reading.ReadOnlyFile`.

    Options (type; default):

    * file_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.file.MemmapSource`)
    * xrootd_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.xrootd.XRootDSource`)
    * http_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.http.HTTPSource`)
    * timeout (float for HTTP, int for XRootD; 30)
    * max_num_elements (None or int; None)
    * num_workers (int; 1)
    * num_fallback_workers (int; 10)
    * begin_chunk_size (memory_size; 512)
    * minimal_ttree_metadata (bool; True)

    See the `ROOT TFile documentation <https://root.cern.ch/doc/master/classTFile.html>`__
    for a specification of ``TFile`` header fields.
    """

    def __init__(
        self,
        file_path,
        object_cache=100,
        array_cache="100 MB",
        custom_classes=None,
        **options  # NOTE: a comma after **options breaks Python 2
    ):
        self._file_path = file_path
        self.object_cache = object_cache
        self.array_cache = array_cache
        self.custom_classes = custom_classes

        self._options = dict(open.defaults)
        self._options.update(options)
        for option in ["begin_chunk_size"]:
            self._options[option] = uproot4._util.memory_size(self._options[option])

        self._streamers = None
        self._streamer_rules = None

        self.hook_before_create_source()

        Source, file_path = uproot4._util.file_path_to_source_class(
            file_path, self._options
        )
        self._source = Source(
            file_path, **self._options  # NOTE: a comma after **options breaks Python 2
        )

        self.hook_before_get_chunks()

        if self._options["begin_chunk_size"] < _file_header_fields_big.size:
            raise ValueError(
                "begin_chunk_size={0} is not enough to read the TFile header ({1})".format(
                    self._options["begin_chunk_size"], _file_header_fields_big.size,
                )
            )

        self._begin_chunk = self._source.chunk(0, self._options["begin_chunk_size"])

        self.hook_before_interpret()

        (
            magic,
            self._fVersion,
            self._fBEGIN,
            self._fEND,
            self._fSeekFree,
            self._fNbytesFree,
            self._nfree,
            self._fNbytesName,
            self._fUnits,
            self._fCompress,
            self._fSeekInfo,
            self._fNbytesInfo,
            self._fUUID_version,
            self._fUUID,
        ) = uproot4.source.cursor.Cursor(0).fields(
            self._begin_chunk, _file_header_fields_small, {}
        )

        if self.is_64bit:
            (
                magic,
                self._fVersion,
                self._fBEGIN,
                self._fEND,
                self._fSeekFree,
                self._fNbytesFree,
                self._nfree,
                self._fNbytesName,
                self._fUnits,
                self._fCompress,
                self._fSeekInfo,
                self._fNbytesInfo,
                self._fUUID_version,
                self._fUUID,
            ) = uproot4.source.cursor.Cursor(0).fields(
                self._begin_chunk, _file_header_fields_big, {}
            )

        self.hook_after_interpret(magic=magic)

        if magic != b"root":
            raise ValueError(
                """not a ROOT file: first four bytes are {0}
in file {1}""".format(
                    repr(magic), file_path
                )
            )

    def __repr__(self):
        return "<ReadOnlyFile {0} at 0x{1:012x}>".format(
            repr(self._file_path), id(self)
        )

    @property
    def detached(self):
        """
        A :doc:`uproot4.reading.DetachedFile` version of this file.
        """
        return DetachedFile(self)

    def close(self):
        """
        Explicitly close the file.

        (Files can also be closed with the Python ``with`` statement, as context
        managers.)

        After closing, new objects and classes cannot be extracted from the file,
        but objects with :doc:`uproot4.reading.DetachedFile` references instead
        of :doc:`uproot4.reading.ReadOnlyFile` that are still in the
        :doc:`uproot4.reading.ReadOnlyFile.object_cache` would still be
        accessible.
        """
        self._source.close()

    @property
    def closed(self):
        """
        True if the file has been closed; False otherwise.

        The file may have been closed explicitly with
        :doc:`uproot4.reading.ReadOnlyFile.close` or implicitly in the Python
        ``with`` statement, as a context manager.

        After closing, new objects and classes cannot be extracted from the file,
        but objects with :doc:`uproot4.reading.DetachedFile` references instead
        of :doc:`uproot4.reading.ReadOnlyFile` that are still in the
        :doc:`uproot4.reading.ReadOnlyFile.object_cache` would still be
        accessible.
        """
        return self._source.closed

    def __enter__(self):
        self._source.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._source.__exit__(exception_type, exception_value, traceback)

    @property
    def source(self):
        """
        The :doc:`uproot4.source.chunk.Source` associated with this file, which
        is the "physical layer" that knows how to communicate with local file
        systems or through remote protocols like HTTP(S) or XRootD, but does not
        know what the bytes mean.
        """
        return self._source

    @property
    def object_cache(self):
        """
        A cache used to hold previously extracted objects, so that code like

        .. code-block:: python

            h = my_file["histogram"]
            h = my_file["histogram"]
            h = my_file["histogram"]

        only reads the ``"histogram"`` once.

        Any Python ``MutableMapping`` can be used as a cache (i.e. a Python
        dict would be a cache that never evicts old objects), though
        :doc:`uproot4.cache.LRUCache` is a good choice because it is thread-safe
        and evicts least-recently used objects when a maximum number of objects
        is reached.
        """
        return self._object_cache

    @object_cache.setter
    def object_cache(self, value):
        if value is None or isinstance(value, MutableMapping):
            self._object_cache = value
        elif uproot4._util.isint(value):
            self._object_cache = uproot4.cache.LRUCache(value)
        else:
            raise TypeError("object_cache must be None, a MutableMapping, or an int")

    @property
    def array_cache(self):
        """
        A cache used to hold previously extracted arrays, so that code like

        .. code-block:: python

            a = my_tree["branch"].array()
            a = my_tree["branch"].array()
            a = my_tree["branch"].array()

        only reads the ``"branch"`` once.

        Any Python ``MutableMapping`` can be used as a cache (i.e. a Python
        dict would be a cache that never evicts old objects), though
        :doc:`uproot4.cache.LRUArrayCache` is a good choice because it is
        thread-safe and evicts least-recently used objects when a size limit is
        reached.
        """
        return self._array_cache

    @array_cache.setter
    def array_cache(self, value):
        if value is None or isinstance(value, MutableMapping):
            self._array_cache = value
        elif uproot4._util.isint(value) or uproot4._util.isstr(value):
            self._array_cache = uproot4.cache.LRUArrayCache(value)
        else:
            raise TypeError(
                "array_cache must be None, a MutableMapping, or a memory size"
            )

    @property
    def root_directory(self):
        """
        The root ``TDirectory`` of the file
        (:doc:`uproot4.reading.ReadOnlyDirectory`).
        """
        return ReadOnlyDirectory(
            (),
            uproot4.source.cursor.Cursor(self._fBEGIN + self._fNbytesName),
            {},
            self,
            self,
        )

    def show_streamers(self, classname=None, version="max", stream=sys.stdout):
        """
        Args:
            classname (None or str): If None, all streamers that are
                defined in the file are shown; if a class name, only
                this class and its dependencies are shown.
            version (int, "min", or "max"): Version number of the desired
                class; "min" or "max" returns the minimum or maximum version
                number, respectively.
            stream (object with a ``write(str)`` method): Stream to write the
                output to.

        Interactively display a file's ``TStreamerInfo``.

        Example with ``classname="TLorentzVector"``:

        .. code-block:: raw

            TVector3 (v3): TObject (v1)
                fX: double (TStreamerBasicType)
                fY: double (TStreamerBasicType)
                fZ: double (TStreamerBasicType)

            TObject (v1)
                fUniqueID: unsigned int (TStreamerBasicType)
                fBits: unsigned int (TStreamerBasicType)

            TLorentzVector (v4): TObject (v1)
                fP: TVector3 (TStreamerObject)
                fE: double (TStreamerBasicType)
        """
        if classname is None:
            names = []
            for name, streamer_versions in self.streamers.items():
                for version in streamer_versions:
                    names.append((name, version))
        else:
            names = self.streamer_dependencies(classname, version=version)
        first = True
        for name, version in names:
            for v, streamer in self.streamers[name].items():
                if v == version:
                    if not first:
                        stream.write(u"\n")
                    streamer.show(stream=stream)
                    first = False

    @property
    def streamers(self):
        """
        A list of :doc:`uproot4.streamers.Model_TStreamerInfo` objects
        representing the ``TStreamerInfos`` in the ROOT file.

        A file's ``TStreamerInfos`` are only read the first time they are needed.
        Uproot has a suite of predefined models in ``uproot4.models`` to reduce
        the probability that ``TStreamerInfos`` will need to be read (depending
        on the choice of classes or versions of the classes that are accessed).

        See also :doc:`uproot4.reading.ReadOnlyFile.streamer_rules`, which are
        read in the same pass with ``TStreamerInfos``.
        """
        import uproot4.streamers
        import uproot4.models.TList
        import uproot4.models.TObjArray
        import uproot4.models.TObjString

        if self._streamers is None:
            if self._fSeekInfo == 0:
                self._streamers = {}

            else:
                key_cursor = uproot4.source.cursor.Cursor(self._fSeekInfo)
                key_start = self._fSeekInfo
                key_stop = min(self._fSeekInfo + _key_format_big.size, self._fEND)
                key_chunk = self.chunk(key_start, key_stop)

                self.hook_before_read_streamer_key(
                    key_chunk=key_chunk, key_cursor=key_cursor,
                )

                streamer_key = ReadOnlyKey(key_chunk, key_cursor, {}, self, self)

                self.hook_before_read_decompress_streamers(
                    key_chunk=key_chunk,
                    key_cursor=key_cursor,
                    streamer_key=streamer_key,
                )

                (
                    streamer_chunk,
                    streamer_cursor,
                ) = streamer_key.get_uncompressed_chunk_cursor()

                self.hook_before_interpret_streamers(
                    key_chunk=key_chunk,
                    key_cursor=key_cursor,
                    streamer_key=streamer_key,
                    streamer_cursor=streamer_cursor,
                    streamer_chunk=streamer_chunk,
                )

                classes = uproot4.model.maybe_custom_classes(self._custom_classes)
                tlist = classes["TList"].read(
                    streamer_chunk, streamer_cursor, {}, self, self.detached, None
                )

                self._streamers = {}
                self._streamer_rules = []

                for x in tlist:
                    if isinstance(x, uproot4.streamers.Model_TStreamerInfo):
                        if x.name not in self._streamers:
                            self._streamers[x.name] = {}
                        self._streamers[x.name][x.class_version] = x

                    elif isinstance(x, uproot4.models.TList.Model_TList) and all(
                        isinstance(y, uproot4.models.TObjString.Model_TObjString)
                        for y in x
                    ):
                        self._streamer_rules.extend([str(y) for y in x])

                    else:
                        raise ValueError(
                            """unexpected type in TList of streamers and streamer rules: {0}
in file {1}""".format(
                                type(x), self._file_path
                            )
                        )

                self.hook_after_interpret_streamers(
                    key_chunk=key_chunk,
                    key_cursor=key_cursor,
                    streamer_key=streamer_key,
                    streamer_cursor=streamer_cursor,
                    streamer_chunk=streamer_chunk,
                )

        return self._streamers

    @property
    def streamer_rules(self):
        """
        A list of strings of C++ code that help schema evolution of
        ``TStreamerInfo`` by providing rules to evaluate when new objects are
        accessed by old ROOT versions.

        Uproot does not evaluate these rules because they are written in C++ and
        Uproot does not have access to a C++ compiler.

        These rules are read in the same pass that produces
        :doc:`uproot4.reading.ReadOnlyFile.streamers`.
        """
        if self._streamer_rules is None:
            self.streamers
        return self._streamer_rules

    def streamers_named(self, classname):
        """
        Returns a list of :doc:`uproot4.streamers.Model_TStreamerInfo` objects
        that match C++ (decoded) ``classname``.

        More that one streamer matching a given name is unlikely, but possible
        because there may be different versions of the same class. (Perhaps such
        files can be created by merging data from different ROOT versions with
        hadd?)

        See also :doc:`uproot4.reading.ReadOnlyFile.streamer_named` (singular).
        """
        streamer_versions = self.streamers.get(classname)
        if streamer_versions is None:
            return []
        else:
            return list(streamer_versions.values())

    def streamer_named(self, classname, version="max"):
        """
        Returns a single :doc:`uproot4.streamers.Model_TStreamerInfo` object
        that matches C++ (decoded) ``classname`` and ``version``.

        The ``version`` can be an integer or ``"min"`` or ``"max"`` for the
        minimum and maximum version numbers available in the file. The default
        is ``"max"`` because there's usually only one.

        See also :doc:`uproot4.reading.ReadOnlyFile.streamers_named` (plural).
        """
        streamer_versions = self.streamers.get(classname)
        if streamer_versions is None or len(streamer_versions) == 0:
            return None
        elif version == "min":
            return streamer_versions[min(streamer_versions)]
        elif version == "max":
            return streamer_versions[max(streamer_versions)]
        else:
            return streamer_versions.get(version)

    def streamer_dependencies(self, classname, version="max"):
        """
        Returns a list of :doc:`uproot4.streamers.Model_TStreamerInfo` objects
        that depend on the one that matches C++ (decoded) ``classname`` and
        ``version``.

        The ``classname`` and ``version`` are interpreted the same way as
        :doc:`uproot4.reading.ReadOnlyFile.streamer_named`.
        """
        streamer = self.streamer_named(classname, version=version)
        out = []
        streamer._dependencies(self.streamers, out)
        return out[::-1]

    @property
    def custom_classes(self):
        """
        Either a dict of class objects specific to this file or None if it uses
        the common ``uproot4.classes`` pool.
        """
        return self._custom_classes

    @custom_classes.setter
    def custom_classes(self, value):
        if value is None or isinstance(value, MutableMapping):
            self._custom_classes = value
        else:
            raise TypeError("custom_classes must be None or a MutableMapping")

    def remove_class_definition(self, classname):
        """
        Removes all versions of a class, specified by C++ (decoded)
        ``classname``, from the :doc:`uproot4.reading.ReadOnlyFile.custom_classes`.

        If the file doesn't have a
        :doc:`uproot4.reading.ReadOnlyFile.custom_classes`, this function adds
        one, so it does not remove the class from the common pool.

        If you want to remove a class from the common pool, you can do so with

        .. code-block:: python

            del uproot4.classes[classname]
        """
        if self._custom_classes is None:
            self._custom_classes = dict(uproot4.classes)
        if classname in self._custom_classes:
            del self._custom_classes[classname]

    def class_named(self, classname, version=None):
        """
        Returns or creates a class with a given C++ (decoded) ``classname``
        and possible ``version``.

        * If the ``version`` is None, this function may return a
          :doc:`uproot4.model.DispatchByVersion`.
        * If the ``version`` is an integer, ``"min"`` or ``"max"``, then it
          returns a :doc:`uproot4.model.VersionedModel`. Using ``"min"`` or
          ``"max"`` specifies the minium or maximum version ``TStreamerInfo``
          defined by the file; most files define only one so ``"max"`` is
          usually safe.

        If this file has :doc:`uproot4.reading.ReadOnlyFile.custom_classes`,
        the new class is added to that dict; otherwise, it is added to the
        global ``uproot4.classes``.
        """
        classes = uproot4.model.maybe_custom_classes(self._custom_classes)
        cls = classes.get(classname)

        if cls is None:
            streamers = self.streamers_named(classname)

            if len(streamers) == 0:
                unknown_cls = uproot4.unknown_classes.get(classname)
                if unknown_cls is None:
                    unknown_cls = uproot4._util.new_class(
                        uproot4.model.classname_encode(classname, unknown=True),
                        (uproot4.model.UnknownClass,),
                        {},
                    )
                    uproot4.unknown_classes[classname] = unknown_cls
                return unknown_cls

            else:
                cls = uproot4._util.new_class(
                    uproot4._util.ensure_str(uproot4.model.classname_encode(classname)),
                    (uproot4.model.DispatchByVersion,),
                    {"known_versions": {}},
                )
                classes[classname] = cls

        if version is not None and issubclass(cls, uproot4.model.DispatchByVersion):
            if not uproot4._util.isint(version):
                streamer = self.streamer_named(classname, version)
                if streamer is not None:
                    version = streamer.class_version
                elif version == "max" and len(cls.known_versions) != 0:
                    version = max(cls.known_versions)
                elif version == "min" and len(cls.known_versions) != 0:
                    version = min(cls.known_versions)
                else:
                    unknown_cls = uproot4.unknown_classes.get(classname)
                    if unknown_cls is None:
                        unknown_cls = uproot4._util.new_class(
                            uproot4.model.classname_encode(
                                classname, version, unknown=True
                            ),
                            (uproot4.model.UnknownClassVersion,),
                            {},
                        )
                        uproot4.unknown_classes[classname] = unknown_cls
                    return unknown_cls

            versioned_cls = cls.class_of_version(version)
            if versioned_cls is None:
                cls = cls.new_class(self, version)
            else:
                cls = versioned_cls

        return cls

    def chunk(self, start, stop):
        """
        Returns a :doc:`uproot4.source.chunk.Chunk` from the
        :doc:`uproot4.source.chunk.Source` that is guaranteed to include bytes
        from ``start`` up to ``stop`` seek points in the file.

        If the desired range is satisfied by a previously saved chunk, such as
        :doc:`uproot4.reading.ReadOnlyFile.begin_chunk`, then that is returned.
        Hence, the returned chunk may include more data than the range from
        ``start`` up to ``stop``.
        """
        if self.closed:
            raise OSError("file {0} is closed".format(repr(self._file_path)))
        elif (start, stop) in self._begin_chunk:
            return self._begin_chunk
        else:
            return self._source.chunk(start, stop)

    @property
    def begin_chunk(self):
        """
        A special :doc:`uproot4.source.chunk.Chunk` corresponding to the
        beginning of the file, from seek point ``0`` up to
        ``options["begin_chunk_size"]``.
        """
        return self._begin_chunk

    def hook_before_create_source(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyFile` constructor before the
        :doc:`uproot4.source.chunk.Source` is created.

        This is the first hook called in the :doc:`uproot4.reading.ReadOnlyFile`
        constructor.
        """
        pass

    def hook_before_get_chunks(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyFile` constructor after the
        :doc:`uproot4.source.chunk.Source` is created but before attempting to
        get any :doc:`uproot4.source.chunk.Chunk`, specifically the
        :doc:`uproot4.reading.ReadOnlyFile.begin_chunk`.
        """
        pass

    def hook_before_interpret(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyFile` constructor after
        loading the :doc:`uproot4.reading.ReadOnlyFile.begin_chunk` and before
        interpreting its ``TFile`` header.
        """
        pass

    def hook_after_interpret(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyFile` constructor after
        interpreting the ``TFile`` header and before raising an error if
        the first four bytes are not ``b"root"``.

        This is the last hook called in the :doc:`uproot4.reading.ReadOnlyFile`
        constructor.
        """
        pass

    def hook_before_read_streamer_key(self, **kwargs):
        """
        Called in :doc:`uproot4.reading.ReadOnlyFile.streamers` before reading
        the ``TKey`` associated with the ``TStreamerInfo``.

        This is the first hook called in
        :doc:`uproot4.reading.ReadOnlyFile.streamers`.
        """
        pass

    def hook_before_read_decompress_streamers(self, **kwargs):
        """
        Called in :doc:`uproot4.reading.ReadOnlyFile.streamers` after reading
        the ``TKey`` associated with the ``TStreamerInfo`` and before reading
        and decompressing the ``TStreamerInfo`` data.
        """
        pass

    def hook_before_interpret_streamers(self, **kwargs):
        """
        Called in :doc:`uproot4.reading.ReadOnlyFile.streamers` after reading
        and decompressing the ``TStreamerInfo`` data, but before interpreting
        it.
        """
        pass

    def hook_after_interpret_streamers(self, **kwargs):
        """
        Called in :doc:`uproot4.reading.ReadOnlyFile.streamers` after
        interpreting the ``TStreamerInfo`` data.

        This is the last hook called in
        :doc:`uproot4.reading.ReadOnlyFile.streamers`.
        """
        pass


_directory_format_small = struct.Struct(">hIIiiiii")
_directory_format_big = struct.Struct(">hIIiiqqq")
_directory_format_num_keys = struct.Struct(">i")


class ReadOnlyDirectory(Mapping):
    """
    Args:
        path (tuple of str): Object path of the ``TDirectory`` as a tuple of
            nested ``TDirectory`` names.
        cursor (:doc:`uproot4.source.cursor.Cursor`): Current position in
            the :doc:`uproot4.reading.ReadOnlyFile`.
        context (dict): Auxiliary data used in deserialization.
        file (:doc:`uproot4.reading.ReadOnlyFile`): The open file object.
        parent (None or calling object): The previous ``read`` in the
            recursive descent.

    Represents a ``TDirectory`` from a ROOT file, most notably, the root
    directory (:doc:`uproot4.reading.ReadOnlyFile.root_directory`).

    Be careful not to confuse :doc:`uproot4.reading.ReadOnlyFile` and
    :doc:`uproot4.reading.ReadOnlyDirectory`: files are for accessing global
    information such as :doc:`uproot4.reading.ReadOnlyFile.streamers` and
    directories are for data in local hierarchies.

    A :doc:`uproot4.reading.ReadOnlyDirectory` is a Python ``Mapping``, which
    uses square bracket syntax to extract objects:

    .. code-block:: python

        my_directory["histogram"]
        my_directory["tree"]
        my_directory["directory"]["another_tree"]

    Objects in ROOT files also have "cycle numbers," which allow multiple
    versions of an object to be retrievable using the same name. A cycle number
    may be specified after a semicolon:

    .. code-block:: python

        my_directory["histogram;2"]

    but without one, the directory defaults to the latest (highest cycle number).

    It's also possible to navigate through nested directories with a slash (``/``)
    instead of sequences of square brackets. The following are equivalent:

    .. code-block:: python

        my_directory["directory"]["another_tree"]["branch_in_tree"]  # long form
        my_directory["directory/another_tree"]["branch_in_tree"]     # / for dir
        my_directory["directory/another_tree/branch_in_tree"]        # / for branch
        my_directory["/directory/another_tree/branch_in_tree"]       # absolute
        my_directory["/directory////another_tree/branch_in_tree"]    # extra ///

    As a Python ``Mapping``, :doc:`uproot4.reading.ReadOnlyDirectory` also has

    * :doc:`uproot4.reading.ReadOnlyDirectory.keys`: names of objects in the
      ``TDirectory``
    * :doc:`uproot4.reading.ReadOnlyDirectory.values`: objects in the
      ``TDirectory``
    * :doc:`uproot4.reading.ReadOnlyDirectory.items`: 2-tuple (name, object)
      pairs.

    However, the :doc:`uproot4.reading.ReadOnlyDirectory` versions of these
    methods have extra parameters for navigating a complex ROOT file. In addition,
    there is a

    * :doc:`uproot4.reading.ReadOnlyDirectory.classnames`: returns a dict of
      (name, classname) pairs.

    with the same parameters.

    See the `ROOT TDirectoryFile documentation <https://root.cern.ch/doc/master/classTDirectoryFile.html>`__
    for a specification of ``TDirectory`` header fields (in an image).
    """

    def __init__(self, path, cursor, context, file, parent):
        self._path = path
        self._cursor = cursor.copy()
        self._file = file
        self._parent = parent

        self.hook_before_read(cursor=cursor)

        directory_start = cursor.index
        directory_stop = min(directory_start + _directory_format_big.size, file.fEND)
        chunk = file.chunk(directory_start, directory_stop)

        self.hook_before_interpret(chunk=chunk, cursor=cursor)

        (
            self._fVersion,
            self._fDatimeC,
            self._fDatimeM,
            self._fNbytesKeys,
            self._fNbytesName,
            self._fSeekDir,
            self._fSeekParent,
            self._fSeekKeys,
        ) = cursor.fields(chunk, _directory_format_small, context, move=False)

        if self.is_64bit:
            (
                self._fVersion,
                self._fDatimeC,
                self._fDatimeM,
                self._fNbytesKeys,
                self._fNbytesName,
                self._fSeekDir,
                self._fSeekParent,
                self._fSeekKeys,
            ) = cursor.fields(chunk, _directory_format_big, context)

        else:
            cursor.skip(_directory_format_small.size)

        if self._fSeekKeys == 0:
            self._header_key = None
            self._keys = []

        else:
            keys_start = self._fSeekKeys
            keys_stop = min(keys_start + self._fNbytesKeys + 8, file.fEND)
            keys_cursor = uproot4.source.cursor.Cursor(self._fSeekKeys)

            self.hook_before_read_keys(
                chunk=chunk, cursor=cursor, keys_cursor=keys_cursor
            )

            if (keys_start, keys_stop) in chunk:
                keys_chunk = chunk
            else:
                keys_chunk = file.chunk(keys_start, keys_stop)

            self.hook_before_header_key(
                chunk=chunk,
                cursor=cursor,
                keys_chunk=keys_chunk,
                keys_cursor=keys_cursor,
            )

            # header_key is never used, but we do need to seek past it
            ReadOnlyKey(keys_chunk, keys_cursor, {}, file, self, read_strings=True)

            num_keys = keys_cursor.field(
                keys_chunk, _directory_format_num_keys, context
            )

            self.hook_before_keys(
                chunk=chunk,
                cursor=cursor,
                keys_chunk=keys_chunk,
                keys_cursor=keys_cursor,
                num_keys=num_keys,
            )

            self._keys = []
            for i in uproot4._util.range(num_keys):
                key = ReadOnlyKey(
                    keys_chunk, keys_cursor, {}, file, self, read_strings=True
                )
                self._keys.append(key)

            self.hook_after_keys(
                chunk=chunk,
                cursor=cursor,
                keys_chunk=keys_chunk,
                keys_cursor=keys_cursor,
                num_keys=num_keys,
            )

    def __repr__(self):
        return "<ReadOnlyDirectory {0} at 0x{1:012x}>".format(
            repr("/" + "/".join(self._path)), id(self)
        )

    @property
    def path(self):
        """
        Object path of the ``TDirectory`` as a tuple of nested ``TDirectory``
        names. The root directory is an empty tuple, ``()``.

        See :doc:`uproot4.reading.ReadOnlyDirectory.object_path` for the path
        as a string.
        """
        return self._path

    @property
    def object_path(self):
        """
        Object path of the ``TDirectory`` as a single string, beginning and
        ending with ``/``. The root directory is a single slash, ``"/"``.

        See :doc:`uproot4.reading.ReadOnlyDirectory.path` for the path as a
        tuple of strings.
        """
        return "/".join(("",) + self._path + ("",)).replace("//", "/")

    @property
    def file(self):
        """
        The :doc:`uproot4.reading.ReadOnlyFile` in which this ``TDirectory``
        resides.

        This property is useful for getting global information, in idioms like

        .. code-block:: python

            with uproot4.open("/path/to/file.root") as handle:
                handle.file.show_streamers()
        """
        return self._file

    def close(self):
        """
        Close the :doc:`uproot4.reading.ReadOnlyFile` in which this ``TDirectory``
        resides.
        """
        self._file.close()

    @property
    def closed(self):
        """
        True if the :doc:`uproot4.reading.ReadOnlyDirectory.file` is closed;
        False otherwise.
        """
        return self._file.closed

    def __enter__(self):
        self._file.source.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.source.__exit__(exception_type, exception_value, traceback)

    @property
    def cursor(self):
        """
        A :doc:`uproot4.source.cursor.Cursor` pointing to the seek point in the
        file where this ``TDirectory`` is defined (at the start of the
        ``TDirectory`` header).
        """
        return self._cursor

    @property
    def parent(self):
        """
        The object that was deserialized before this one in recursive descent,
        usually the containing object (or the container's container).
        """
        return self._parent

    @property
    def is_64bit(self):
        """
        True if the ``TDirectory`` is 64-bit ready; False otherwise.

        This refers to seek points like
        :doc:`uproot4.reading.ReadOnlyDirectory.fSeekDir` being 64-bit integers,
        rather than 32-bit.

        Note that a file being 64-bit is distinct from a ``TDirectory`` being
        64-bit; see :doc:`uproot4.reading.ReadOnlyFile.is_64bit`.
        """
        return self._fVersion > 1000

    @property
    def cache_key(self):
        """
        String that uniquely specifies this ``TDirectory`` path, to use as part
        of object and array cache keys.
        """
        return self.file.hex_uuid + ":" + self.object_path

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
            self.iterkeys(
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def values(
        self, recursive=True, filter_name=no_filter, filter_classname=no_filter,
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
        :doc:`uproot4.model.Model`.

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
        list of 2-tuples of (str, :doc:`uproot4.model.Model`).

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
        recursive=True,
        cycle=False,
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
        filter_name = uproot4._util.regularize_filter(filter_name)
        filter_classname = uproot4._util.regularize_filter(filter_classname)
        for key in self._keys:
            if (filter_name is no_filter or filter_name(key.fName)) and (
                filter_classname is no_filter or filter_classname(key.fClassName)
            ):
                yield key.name(cycle=cycle)

            if recursive and key.fClassName in ("TDirectory", "TDirectoryFile"):
                for k1 in key.get().iterkeys(
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=no_filter,
                    filter_classname=filter_classname,
                ):
                    k2 = "{0}/{1}".format(key.name(cycle=False), k1)
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2

    def itervalues(
        self, recursive=True, filter_name=no_filter, filter_classname=no_filter,
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
        :doc:`uproot4.model.Model`.

        Note that this reads all objects that are selected by ``filter_name``
        and ``filter_classname``.
        """
        for k, v in self.iteritems(
            recursive=recursive,
            cycle=False,
            filter_name=filter_name,
            filter_classname=filter_classname,
        ):
            yield v

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
        iterator over 2-tuples of (str, :doc:`uproot4.model.Model`).

        Note that this reads all objects that are selected by ``filter_name``
        and ``filter_classname``.
        """
        filter_name = uproot4._util.regularize_filter(filter_name)
        filter_classname = uproot4._util.regularize_filter(filter_classname)
        for key in self._keys:
            if (filter_name is no_filter or filter_name(key.fName)) and (
                filter_classname is no_filter or filter_classname(key.fClassName)
            ):
                yield key.name(cycle=cycle), key.get()

            if recursive and key.fClassName in ("TDirectory", "TDirectoryFile"):
                for k1, v in key.get().iteritems(
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=no_filter,
                    filter_classname=filter_classname,
                ):
                    k2 = "{0}/{1}".format(key.name(cycle=False), k1)
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2, v

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
        filter_name = uproot4._util.regularize_filter(filter_name)
        filter_classname = uproot4._util.regularize_filter(filter_classname)
        for key in self._keys:
            if (filter_name is no_filter or filter_name(key.fName)) and (
                filter_classname is no_filter or filter_classname(key.fClassName)
            ):
                yield key.name(cycle=cycle), key.fClassName

            if recursive and key.fClassName in ("TDirectory", "TDirectoryFile"):
                for k1, v in key.get().iterclassnames(
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=no_filter,
                    filter_classname=filter_classname,
                ):
                    k2 = "{0}/{1}".format(key.name(cycle=False), k1)
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2, v

    def _ipython_key_completions_(self):
        """
        Supports key-completion in an IPython or Jupyter kernel.
        """
        return self.iterkeys()

    def __len__(self):
        return len(self._keys) + sum(
            len(x.get())
            for x in self._keys
            if x.fClassName in ("TDirectory", "TDirectoryFile")
        )

    def __contains__(self, where):
        try:
            self.key(where)
        except KeyError:
            return False
        else:
            return True

    def __iter__(self):
        return self.iterkeys()

    def classname_of(self, where, encoded=False, version=None):
        """
        Returns the classname of the object selected by ``where``. If
        ``encoded`` with a possible ``version``, return a Python classname;
        otherwise, return a C++ (decoded) classname.

        The syntax for ``where`` is the same as in square brakets, namely that
        cycle numbers can be specified after semicolons (``;``) and nested
        ``TDirectories`` can be specified with slashes (``/``).

        Unlike the square bracket syntax, this method cannot descend into the
        ``TBranches`` of a ``TTree``.

        Note that this does not read any data from the file.
        """
        key = self.key(where)
        return key.classname(encoded=encoded, version=version)

    def class_of(self, where, version=None):
        """
        Returns a class object for the ROOT object selected by ``where``. If
        ``version`` is specified, get a :doc:`uproot4.model.VersionedModel`;
        otherwise, get a :doc:`uproot4.model.DispatchByVersion` or a versionless
        :doc:`uproot4.model.Model`.

        The syntax for ``where`` is the same as in square brakets, namely that
        cycle numbers can be specified after semicolons (``;``) and nested
        ``TDirectories`` can be specified with slashes (``/``).

        Unlike the square bracket syntax, this method cannot descend into the
        ``TBranches`` of a ``TTree``.

        Note that this does not read any data from the file.
        """
        key = self.key(where)
        return self._file.class_named(key.fClassName, version=version)

    def streamer_of(self, where, version):
        """
        Returns a ``TStreamerInfo`` (:doc:`uproot4.streamers.Model_TStreamerInfo`)
        for the object selected by ``where`` and ``version``.

        The syntax for ``where`` is the same as in square brakets, namely that
        cycle numbers can be specified after semicolons (``;``) and nested
        ``TDirectories`` can be specified with slashes (``/``).

        Unlike the square bracket syntax, this method cannot descend into the
        ``TBranches`` of a ``TTree``.

        Note that this does not read any data from the file.
        """
        key = self.key(where)
        return self._file.streamer_named(key.fClassName, version)

    def key(self, where):
        """
        Returns a ``TKey`` (:doc:`uproot4.reading.ReadOnlyKey`) for the object
        selected by ``where``.

        The syntax for ``where`` is the same as in square brakets, namely that
        cycle numbers can be specified after semicolons (``;``) and nested
        ``TDirectories`` can be specified with slashes (``/``).

        Unlike the square bracket syntax, this method cannot descend into the
        ``TBranches`` of a ``TTree`` (since they have no ``TKeys``).

        Note that this does not read any data from the file.
        """
        where = uproot4._util.ensure_str(where)

        if "/" in where:
            items = where.split("/")
            step = last = self
            for item in items[:-1]:
                if item != "":
                    if isinstance(step, ReadOnlyDirectory):
                        last = step
                        step = step[item]
                    else:
                        raise uproot4.KeyInFileError(
                            where,
                            repr(item) + " is not a TDirectory",
                            keys=[key.fName for key in last._keys],
                            file_path=self._file.file_path,
                        )
            return step.key(items[-1])

        if ";" in where:
            at = where.rindex(";")
            item, cycle = where[:at], where[at + 1 :]
            try:
                cycle = int(cycle)
            except ValueError:
                item, cycle = where, None
        else:
            item, cycle = where, None

        last = None
        for key in self._keys:
            if key.fName == item:
                if cycle == key.fCycle:
                    return key
                elif cycle is None and last is None:
                    last = key
                elif cycle is None and last.fCycle < key.fCycle:
                    last = key

        if last is not None:
            return last
        elif cycle is None:
            raise uproot4.KeyInFileError(
                item, cycle="any", keys=self.keys(), file_path=self._file.file_path
            )
        else:
            raise uproot4.KeyInFileError(
                item, cycle=cycle, keys=self.keys(), file_path=self._file.file_path
            )

    def __getitem__(self, where):
        if "/" in where or ":" in where:
            items = where.split("/")
            step = last = self

            for i, item in enumerate(items):
                if item != "":
                    if isinstance(step, ReadOnlyDirectory):
                        if ":" in item and item not in step:
                            index = item.index(":")
                            head, tail = item[:index], item[index + 1 :]
                            last = step
                            step = step[head]
                            if isinstance(step, uproot4.behaviors.TBranch.HasBranches):
                                return step["/".join([tail] + items[i + 1 :])]
                            else:
                                raise uproot4.KeyInFileError(
                                    where,
                                    repr(head)
                                    + " is not a TDirectory, TTree, or TBranch",
                                    keys=[key.fName for key in last._keys],
                                    file_path=self._file.file_path,
                                )
                        else:
                            last = step
                            step = step[item]

                    elif isinstance(step, uproot4.behaviors.TBranch.HasBranches):
                        return step["/".join(items[i:])]

                    else:
                        raise uproot4.KeyInFileError(
                            where,
                            repr(item) + " is not a TDirectory, TTree, or TBranch",
                            keys=[key.fName for key in last._keys],
                            file_path=self._file.file_path,
                        )

            return step

        else:
            return self.key(where).get()

    @property
    def fVersion(self):
        """
        Raw integer version of the ``TDirectory`` class.
        """
        return self._fVersion

    @property
    def fDatimeC(self):
        """
        Raw integer creation date/time.

        FIXME: include a property that converts this to a Python ``datetime``.
        """
        return self._fDatimeC

    @property
    def fDatimeM(self):
        """
        Raw integer date/time of last modification.

        FIXME: include a property that converts this to a Python ``datetime``.
        """
        return self._fDatimeM

    @property
    def fNbytesKeys(self):
        """
        Number of bytes in the collection of ``TKeys`` (header key, number of
        directory keys, and directory keys).
        """
        return self._fNbytesKeys

    @property
    def fNbytesName(self):
        """
        Number of bytes in the header up to its title.
        """
        return self._fNbytesName

    @property
    def fSeekDir(self):
        """
        File seek position (int) of the ``TDirectory``.
        """
        return self._fSeekDir

    @property
    def fSeekParent(self):
        """
        File seek position (int) of the parent object (``TDirectory`` or ``TFile``).
        """
        return self._fSeekParent

    @property
    def fSeekKeys(self):
        """
        File seek position (int) to the collection of ``TKeys`` (header key,
        number of directory keys, and directory keys).
        """
        return self._fSeekKeys

    def hook_before_read(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyDirectory` constructor before
        reading the ``TDirectory`` header fields.

        This is the first hook called in the
        :doc:`uproot4.reading.ReadOnlyDirecotry` constructor.
        """
        pass

    def hook_before_interpret(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyDirectory` constructor after
        reading the ``TDirectory`` header fields and before interpreting them.
        """
        pass

    def hook_before_read_keys(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyDirectory` constructor after
        interpreting the ``TDirectory`` header fields and before reading the
        chunk of ``TKeys``.
        """
        pass

    def hook_before_header_key(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyDirectory` constructor after
        reading the chunk of ``TKeys`` and before interpreting the header ``TKey``.
        """
        pass

    def hook_before_keys(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyDirectory` constructor after
        interpreting the header ``TKey`` and number of keys, and before
        interpeting the object ``TKeys``.
        """
        pass

    def hook_after_keys(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyDirectory` constructor after
        interpeting the object ``TKeys``.

        This is the last hook called in the
        :doc:`uproot4.reading.ReadOnlyDirecotry` constructor.
        """
        pass


_key_format_small = struct.Struct(">ihiIhhii")
_key_format_big = struct.Struct(">ihiIhhqq")


class ReadOnlyKey(object):
    """
    Args:
        chunk (:doc:`uproot4.source.chunk.Chunk`): Buffer of contiguous data
            from the file :doc:`uproot4.source.chunk.Source`.
        cursor (:doc:`uproot4.source.cursor.Cursor`): Current position in
            the :doc:`uproot4.reading.ReadOnlyFile`.
        context (dict): Auxiliary data used in deserialization.
        file (:doc:`uproot4.reading.ReadOnlyFile`): The open file object.
        parent (None or calling object): The previous ``read`` in the
            recursive descent.
        read_strings (bool): If True, interpret the `fClassName`, `fName`, and
            `fTitle` attributes; otherwise, ignore them.

    Represents a ``TKey`` in a ROOT file, which aren't often accessed by users.

    See the `ROOT TKey documentation <https://root.cern.ch/doc/master/classTKey.html>`__
    for a specification of ``TKey`` header fields.
    """

    def __init__(self, chunk, cursor, context, file, parent, read_strings=False):
        self._cursor = cursor.copy()
        self._file = file
        self._parent = parent

        self.hook_before_interpret(
            chunk=chunk,
            cursor=cursor,
            context=context,
            file=file,
            parent=parent,
            read_strings=read_strings,
        )

        (
            self._fNbytes,
            self._fVersion,
            self._fObjlen,
            self._fDatime,
            self._fKeylen,
            self._fCycle,
            self._fSeekKey,
            self._fSeekPdir,
        ) = cursor.fields(chunk, _key_format_small, context, move=False)

        if self.is_64bit:
            (
                self._fNbytes,
                self._fVersion,
                self._fObjlen,
                self._fDatime,
                self._fKeylen,
                self._fCycle,
                self._fSeekKey,
                self._fSeekPdir,
            ) = cursor.fields(chunk, _key_format_big, context)

        else:
            cursor.skip(_key_format_small.size)

        if read_strings:
            self.hook_before_strings(
                chunk=chunk,
                cursor=cursor,
                context=context,
                file=file,
                parent=parent,
                read_strings=read_strings,
            )

            self._fClassName = cursor.string(chunk, context)
            self._fName = cursor.string(chunk, context)
            self._fTitle = cursor.string(chunk, context)

        else:
            self._fClassName = None
            self._fName = None
            self._fTitle = None

        self.hook_after_interpret(
            chunk=chunk,
            cursor=cursor,
            context=context,
            file=file,
            parent=parent,
            read_strings=read_strings,
        )

    def __repr__(self):
        if self._fName is None or self._fClassName is None:
            nameclass = ""
        else:
            nameclass = " {0}: {1}".format(self.name(cycle=True), self.classname())
        return "<ReadOnlyKey{0} (seek pos {1}) at 0x{2:012x}>".format(
            nameclass, self.data_cursor.index, id(self)
        )

    @property
    def cursor(self):
        """
        A :doc:`uproot4.source.cursor.Cursor` pointing to the seek point in the
        file where this ``TKey`` starts (before its header fields).
        """
        return self._cursor

    @property
    def data_cursor(self):
        """
        A :doc:`uproot4.source.cursor.Cursor` pointing to the seek point in the
        file where the data begins (the object to be read, after its copy of the
        ``TKey`` and before the object's number of bytes/version header).
        """
        return uproot4.source.cursor.Cursor(self._fSeekKey + self._fKeylen)

    @property
    def file(self):
        """
        The :doc:`uproot4.reading.ReadOnlyFile` in which this ``TKey`` resides.
        """
        return self._file

    @property
    def parent(self):
        """
        The object that was deserialized before this one in recursive descent,
        usually the containing object (or the container's container).
        """
        return self._parent

    def name(self, cycle=False):
        """
        The name of this ``TKey``, with semicolon (``;``) and cycle number if
        ``cycle`` is True.
        """
        if cycle:
            return "{0};{1}".format(self.fName, self.fCycle)
        else:
            return self.fName

    def classname(self, encoded=False, version=None):
        """
        The classname of the object pointed to by this ``TKey``.

        If ``encoded`` is True, the classname is a Python (encoded) classname
        (optionally with ``version`` number).

        If ``encoded`` is False, the classname is a C++ (decoded) classname.
        """
        if encoded:
            return uproot4.model.classname_encode(self.fClassName, version=version)
        else:
            return self.fClassName

    @property
    def object_path(self):
        """
        Object path of the object pointed to by this ``TKey``.

        If an object path is not known (e.g. the object does not reside in a
        ``TDirectory``), this returns a message with the raw seek position.
        """
        if isinstance(self._parent, ReadOnlyDirectory):
            return "{0}{1};{2}".format(
                self._parent.object_path, self.name(False), self._fCycle
            )
        else:
            return "(seek pos {0})/{1}".format(self.data_cursor.index, self.name(False))

    @property
    def cache_key(self):
        """
        String that uniquely specifies this ``TKey``, to use as part
        of object and array cache keys.
        """
        return "{0}:{1}".format(self._file.hex_uuid, self._fSeekKey)

    @property
    def is_64bit(self):
        """
        True if the ``TKey`` is 64-bit ready; False otherwise.

        This refers to seek points like
        :doc:`uproot4.reading.ReadOnlyKey.fSeekKey` being 64-bit integers,
        rather than 32-bit.
        """
        return self._fVersion > 1000

    @property
    def is_compressed(self):
        """
        True if the object is compressed; False otherwise.

        Note: compression is determined by comparing ``fObjlen`` and
        ``fNbytes - fKeylen``; if they are the same, the object is uncompressed.
        """
        return self.data_compressed_bytes != self.data_uncompressed_bytes

    @property
    def data_uncompressed_bytes(self):
        """
        Number of bytes in the uncompressed object (excluding any keys)

        This is equal to :doc:`uproot4.reading.ReadOnlyKey.fObjlen``.
        """
        return self._fObjlen

    @property
    def data_compressed_bytes(self):
        """
        Number of bytes in the compressed object (excluding any keys)

        This is equal to :doc:`uproot4.reading.ReadOnlyKey.fNbytes``
        minus :doc:`uproot4.reading.ReadOnlyKey.fKeylen``.
        """
        return self._fNbytes - self._fKeylen

    def get(self):
        """
        Returns the object pointed to by this ``TKey``, decompressing it if
        necessary.

        If the first attempt to deserialize the object fails with
        :doc:`uproot4.deserialization.DeserializationError` and any of the
        models used in that attempt were predefined (not from
        :doc:`uproot4.reading.ReadOnlyFile.streamers`), this method will
        try again with the file's own
        :doc:`uproot4.reading.ReadOnlyFile.streamers`.

        (Some ROOT files do have classes that don't match the standard
        ``TStreamerInfo``; they may have been produced from private builds of
        ROOT between official releases.)
        """
        if self._file.object_cache is not None:
            out = self._file.object_cache.get(self.cache_key)
            if out is not None:
                if isinstance(out.file, ReadOnlyFile) and out.file.closed:
                    del self._file.object_cache[self.cache_key]
                else:
                    return out

        if self._fClassName in must_be_attached:
            selffile = self._file
            parent = self
        else:
            selffile = self._file.detached
            parent = None

        if isinstance(self._parent, ReadOnlyDirectory) and self._fClassName in (
            "TDirectory",
            "TDirectoryFile",
        ):
            out = ReadOnlyDirectory(
                self._parent.path + (self.fName,),
                self.data_cursor,
                {},
                self._file,
                self,
            )

        else:
            chunk, cursor = self.get_uncompressed_chunk_cursor()
            start_cursor = cursor.copy()
            cls = self._file.class_named(self._fClassName)
            context = {"breadcrumbs": (), "TKey": self}

            try:
                out = cls.read(chunk, cursor, context, self._file, selffile, parent)

            except uproot4.deserialization.DeserializationError:
                breadcrumbs = context.get("breadcrumbs")

                if breadcrumbs is None or all(
                    breadcrumb_cls.classname in uproot4.model.bootstrap_classnames
                    or isinstance(breadcrumb_cls, uproot4.containers.AsContainer)
                    or getattr(breadcrumb_cls.class_streamer, "file_uuid", None)
                    == self._file.uuid
                    for breadcrumb_cls in breadcrumbs
                ):
                    # we're already using the most specialized versions of each class
                    raise

                for breadcrumb_cls in breadcrumbs:
                    if (
                        breadcrumb_cls.classname
                        not in uproot4.model.bootstrap_classnames
                    ):
                        self._file.remove_class_definition(breadcrumb_cls.classname)

                cursor = start_cursor
                cls = self._file.class_named(self._fClassName)
                context = {"breadcrumbs": (), "TKey": self}

                out = cls.read(chunk, cursor, context, self._file, selffile, parent)

        if self._fClassName not in must_be_attached:
            out._file = self._file.detached
            out._parent = None

        if self._file.object_cache is not None:
            self._file.object_cache[self.cache_key] = out
        return out

    def get_uncompressed_chunk_cursor(self):
        """
        Returns an uncompressed :doc:`uproot4.source.chunk.Chunk` and
        :doc:`uproot4.source.cursor.Cursor` for the object pointed to by this
        ``TKey`` as a 2-tuple.
        """
        cursor = uproot4.source.cursor.Cursor(0, origin=-self._fKeylen)

        data_start = self.data_cursor.index
        data_stop = data_start + self.data_compressed_bytes
        chunk = self._file.chunk(data_start, data_stop)

        if self.is_compressed:
            uncompressed_chunk = uproot4.compression.decompress(
                chunk,
                self.data_cursor,
                {},
                self.data_compressed_bytes,
                self.data_uncompressed_bytes,
            )
        else:
            uncompressed_chunk = uproot4.source.chunk.Chunk.wrap(
                chunk.source,
                chunk.get(
                    data_start,
                    data_stop,
                    self.data_cursor,
                    {"breadcrumbs": (), "TKey": self},
                ),
            )

        return uncompressed_chunk, cursor

    @property
    def fNbytes(self):
        """
        The total number of bytes in the compressed object and ``TKey``.
        """
        return self._fNbytes

    @property
    def fVersion(self):
        """
        Raw integer version of the ``TKey`` class.
        """
        return self._fVersion

    @property
    def fObjlen(self):
        """
        The number of bytes in the uncompressed object, not including its
        ``TKey``.
        """
        return self._fObjlen

    @property
    def fDatime(self):
        """
        Raw integer date/time when the object was written.

        FIXME: include a property that converts this to a Python ``datetime``.
        """
        return self._fDatime

    @property
    def fKeylen(self):
        """
        The number of bytes in the ``TKey``, not including the object.
        """
        return self._fKeylen

    @property
    def fCycle(self):
        """
        The cycle number of the object, which disambiguates different versions
        of an object with the same name.
        """
        return self._fCycle

    @property
    def fSeekKey(self):
        """
        File seek position (int) pointing to a second copy of this ``TKey``, just
        before the object itself.
        """
        return self._fSeekKey

    @property
    def fSeekPdir(self):
        """
        File seek position (int) to the ``TDirectory`` in which this object
        resides.
        """
        return self._fSeekPdir

    @property
    def fClassName(self):
        """
        The C++ (decoded) classname of the object or None if the
        :doc:`uproot4.reading.ReadOnlyKey` was constructed with
        ``read_strings=False``.
        """
        return self._fClassName

    @property
    def fName(self):
        """
        The name of the object or None if the :doc:`uproot4.reading.ReadOnlyKey`
        was constructed with ``read_strings=False``.
        """
        return self._fName

    @property
    def fTitle(self):
        """
        The title of the object or None if the :doc:`uproot4.reading.ReadOnlyKey`
        was constructed with ``read_strings=False``.
        """
        return self._fTitle

    def hook_before_interpret(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyKey` constructor before
        interpeting anything.

        This is the first hook called in the
        :doc:`uproot4.reading.ReadOnlyKey` constructor.
        """
        pass

    def hook_before_strings(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyKey` constructor after
        interpeting the header and before interpreting
        :doc:`uproot4.reading.ReadOnlyKey.fClassName`,
        :doc:`uproot4.reading.ReadOnlyKey.fName`, and
        :doc:`uproot4.reading.ReadOnlyKey.fTitle`.

        Only called if ``read_strings=True`` is passed to the constructor.
        """
        pass

    def hook_after_interpret(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyKey` constructor after
        interpeting everything.

        This is the last hook called in the
        :doc:`uproot4.reading.ReadOnlyKey` constructor.
        """
        pass
