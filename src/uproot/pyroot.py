# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import threading

import numpy

import uproot


def pyroot_to_buffer(obj):
    """
    FIXME: docstring
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
    ROOT._uproot_TMessage_SetBuffer(
        message, pyroot_to_buffer.buffer, len(pyroot_to_buffer.buffer)
    )
    message.WriteObject(obj)
    return pyroot_to_buffer.buffer[: message.Length()]


pyroot_to_buffer.lock = threading.Lock()
pyroot_to_buffer.sizer = None
pyroot_to_buffer.buffer = None


class _ReadFromTMessage(object):
    def class_named(self, classname, version=None):
        return uproot.class_named(classname, version=version)


def from_pyroot(obj):
    """
    FIXME: docstring
    """
    with pyroot_to_buffer.lock:
        buffer = pyroot_to_buffer(obj)
        chunk = uproot.source.chunk.Chunk.wrap(None, buffer)
        cursor = uproot.source.cursor.Cursor(0)
        fakefile = _ReadFromTMessage()
        return uproot.deserialization.read_object_any(
            chunk, cursor, {}, fakefile, fakefile, None
        )
