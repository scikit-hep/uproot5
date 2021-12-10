# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module provides an interface between Uproot and PyROOT.

PyROOT is a *complete* set of ROOT bindings generated automatically from ROOT's
reflected C++ code. Uproot is a pure Python reimplementation of ROOT I/O, with
incomplete coverage of ROOT's suite of classes, but without an install-time
dependence on ROOT.

The only point of contact between Uproot and PyROOT is that they both recognize
the same *serialized* form of ROOT objects, so conversions in this module proceed
by serializing and deserializing the objects through a TMessage. This makes it
possible to

- convert any PyROOT object into its Uproot :doc:`uproot.model.Model` *if Uproot can read it* (which is possible for most classes)
- convert any Uproot :doc:`uproot.model.Model` into the corresponding PyROOT object *if Uproot can write it* (which is considerably more constrained; mostly just histograms).

This module also makes it possible for PyROOT objects to be added to ROOT files
that Uproot is writing (regardless of whether Uproot could read such objects).
"""


import threading
import uuid

import numpy

import uproot


def to_pyroot(obj, name=None):
    """
    Args:
        obj (:doc:`uproot.model.Model`): The Uproot model to convert.
        name (str or None): A name for the new PyROOT object.

    Converts an :doc:`uproot.model.Model` into a PyROOT object *if it is writable by Uproot*.
    A minority of Uproot models are writable, mostly just histograms. Writability
    is necessary for conversion to PyROOT because it is serialized through a
    ROOT TMessage.
    """
    import ROOT

    if to_pyroot._Uproot_FromTMessage is None:
        ROOT.gInterpreter.Declare(
            """
class _Uproot_FromTMessage : public TMessage {
public:
    _Uproot_FromTMessage(void* buffer, Int_t size): TMessage(buffer, size) { }
};
"""
        )
        to_pyroot._Uproot_FromTMessage = ROOT._Uproot_FromTMessage

    serialized = uproot.serialization.serialize_object_any(obj, name)
    buffer = numpy.empty(len(serialized) + 8, numpy.uint8)
    buffer[:8].view(numpy.uint64)[0] = ROOT.kMESS_OBJECT
    buffer[8:] = numpy.frombuffer(serialized, numpy.uint8)

    message = to_pyroot._Uproot_FromTMessage(buffer, len(buffer))
    out = message.ReadObject(message.GetClass())
    message.DetachBuffer()
    return out


to_pyroot._Uproot_FromTMessage = None


def pyroot_to_buffer(obj):
    """
    Args:
        obj (PyROOT object inheriting from TObject): PyROOT object to serialize.

    Serializes a PyROOT object into a NumPy array that is owned by this function.

    This function is not thread-safe and the output buffer gets overwritten by
    the next call to this function. It is essential for callers to copy the data
    out of the returned buffer, perhaps by calling :doc:`uproot._util.tobytes` on
    it or by assigning it into another array.

    A lock is provided for safety: callers should always call this function within
    the lock's context:

    .. code-block:: python

        with pyroot_to_buffer.lock:
            return uproot._util.tobytes(pyroot_to_buffer(obj))
    """
    import ROOT

    if pyroot_to_buffer.sizer is None:
        ROOT.gInterpreter.Declare(
            """
class _Uproot_buffer_sizer : public TObject {
public:
  size_t buffer;
  size_t newsize;
  size_t oldsize;
};

char* _uproot_TMessage_reallocate(char* buffer, size_t newsize, size_t oldsize) {
    _Uproot_buffer_sizer* ptr = reinterpret_cast<_Uproot_buffer_sizer*>(
        (void*)TPython::Eval("uproot.pyroot.pyroot_to_buffer.sizer")
    );
    ptr->buffer = reinterpret_cast<size_t>(buffer);
    ptr->newsize = newsize;
    ptr->oldsize = oldsize;

    TPython::Exec("uproot.pyroot.pyroot_to_buffer.reallocate()");

    TPyReturn out = TPython::Eval("uproot.pyroot.pyroot_to_buffer.buffer.ctypes.data");
    return reinterpret_cast<char*>((size_t)out);
}

void _uproot_TMessage_SetBuffer(TMessage& message, void* buffer, UInt_t newsize) {
    message.SetBuffer(buffer, newsize, false, _uproot_TMessage_reallocate);
}
"""
        )
        pyroot_to_buffer.sizer = ROOT._Uproot_buffer_sizer()
        pyroot_to_buffer.buffer = numpy.empty(1024, numpy.uint8)

        def reallocate():
            newbuf = numpy.empty(pyroot_to_buffer.sizer.newsize, numpy.uint8)
            newbuf[: len(pyroot_to_buffer.buffer)] = pyroot_to_buffer.buffer
            pyroot_to_buffer.buffer = newbuf

        pyroot_to_buffer.reallocate = reallocate

    message = ROOT.TMessage(ROOT.kMESS_OBJECT)
    message.SetCompressionLevel(0)
    ROOT._uproot_TMessage_SetBuffer(
        message, pyroot_to_buffer.buffer, len(pyroot_to_buffer.buffer)
    )
    message.WriteObject(obj)
    return pyroot_to_buffer.buffer[: message.Length()]


pyroot_to_buffer.lock = threading.Lock()
pyroot_to_buffer.sizer = None
pyroot_to_buffer.buffer = None


class _GetStreamersOnce:
    _custom_classes = {}
    _streamers = {}
    _streamer_dependencies = {}

    def __init__(self, obj):
        self._obj = obj

    def class_named(self, classname, version=None):
        return uproot.reading.ReadOnlyFile.class_named(self, classname, version)

    def streamers_named(self, classname):
        return uproot.reading.ReadOnlyFile.streamers_named(self, classname)

    def streamer_named(self, classname, version):
        return uproot.reading.ReadOnlyFile.streamer_named(
            self, classname, version=version
        )

    @property
    def custom_classes(self):
        return self._custom_classes

    @property
    def file_path(self):
        return None

    class ArrayFile:
        def __init__(self, array):
            self.array = array
            self.current = 0

        def seek(self, position):
            self.current = position

        def read(self, num_bytes):
            position = self.current + num_bytes
            out = self.array[self.current : position]
            self.current = position
            return out

    @property
    def streamers(self):
        tclass = self._obj.IsA()
        obj_classname = tclass.GetName()
        obj_version = tclass.GetClassVersion()
        if self._streamers.get(obj_classname, {}).get(obj_version, None) is None:
            import ROOT

            memfile = ROOT.TMemFile("noname.root", "new")
            memfile.SetCompressionLevel(0)
            memfile.WriteObjectAny(self._obj, self._obj.IsA(), "noname")
            memfile.WriteStreamerInfo()
            memfile.Close()

            buffer = numpy.empty(memfile.GetEND(), numpy.uint8)
            memfile.CopyTo(buffer, len(buffer))

            file = uproot.open(_GetStreamersOnce.ArrayFile(buffer))

            dependencies = self._streamer_dependencies[obj_classname, obj_version] = []

            for classname, versions in file.file.streamers.items():
                if classname not in self._streamers:
                    self._streamers[classname] = {}
                for version, streamerinfo in versions.items():
                    self._streamers[classname][version] = streamerinfo
                    dependencies.append(streamerinfo)

        return self._streamers


class _NoFile:
    def __init__(self):
        import ROOT

        self._file_path = ""
        self._options = {}
        self._fVersion = ROOT.gROOT.GetVersionInt()
        self._fBEGIN = 0
        self._fEND = 0
        self._fSeekFree = 0
        self._fNbytesFree = 0
        self._nfree = 0
        self._fNbytesName = 0
        self._fUnits = 0
        self._fCompress = 0
        self._fSeekInfo = 0
        self._fNbytesInfo = 0
        self._fUUID_version = 1
        self._fUUID = uuid.UUID("00000000-0000-0000-0000-000000000000")

    @property
    def streamers(self):
        return _GetStreamersOnce._streamers

    @property
    def custom_classes(self):
        return _GetStreamersOnce._custom_classes


def from_pyroot(obj):
    """
    Args:
        obj (PyROOT object inheriting from TObject): PyROOT object to convert to
            an Uproot :doc:`uproot.model.Model`.

    Converts a PyROOT object into its corresponding Uproot :doc:`uproot.model.Model`
    *if it is readable by Uproot*. Most ROOT classes are readable by Uproot. Readability
    is necessary for conversion from PyROOT because the object is serialized through a
    ROOT TMessage.
    """
    with pyroot_to_buffer.lock:
        buffer = pyroot_to_buffer(obj)
        chunk = uproot.source.chunk.Chunk.wrap(None, buffer)
        cursor = uproot.source.cursor.Cursor(0)
        maybestreamers = _GetStreamersOnce(obj)
        detatched = _NoFile()
        return uproot.deserialization.read_object_any(
            chunk, cursor, {}, maybestreamers, detatched, None
        )


class _PyROOTWritable:
    def __init__(self, obj):
        self._obj = obj

    @property
    def class_rawstreamers(self):
        tclass = self._obj.IsA()
        key = (tclass.GetName(), tclass.GetClassVersion())
        return _GetStreamersOnce._streamer_dependencies.get(key, [])

    @property
    def classname(self):
        tclass = self._obj.IsA()
        return tclass.GetName()

    @property
    def fTitle(self):
        return self._obj.GetTitle()

    def serialize(self, name=None):
        if name is None or name == self._obj.GetName():
            obj = self._obj
        else:
            obj = self._obj.Clone(name)

        with pyroot_to_buffer.lock:
            return uproot._util.tobytes(
                pyroot_to_buffer(obj)[len(self.classname) + 9 :]
            )
