# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.interpretation
import uproot4.interpretation.strings
import uproot4.interpretation.jagged
import uproot4.interpretation.numerical
import uproot4.stl_containers
import uproot4.model
import uproot4.source.chunk
import uproot4.source.cursor
import uproot4._util


class ObjectArray(object):
    def __init__(self, model, branch, context, byte_offsets, byte_content):
        self._model = model
        self._branch = branch
        self._context = context
        self._byte_offsets = byte_offsets
        self._byte_content = byte_content

    def __repr__(self):
        return "ObjectArray({0}, {1}, {2}, {3}, {4})".format(
            self._model,
            self._branch,
            self._context,
            self._byte_offsets,
            self._byte_content,
        )

    @property
    def model(self):
        return self._model

    @property
    def branch(self):
        return self._branch

    @property
    def context(self):
        return self._context

    @property
    def byte_offsets(self):
        return self._byte_offsets

    @property
    def byte_content(self):
        return self._byte_content

    def __len__(self):
        return len(self._byte_offsets) - 1

    def chunk(self, entry_num):
        byte_start = self._byte_offsets[entry_num]
        byte_stop = self._byte_offsets[entry_num + 1]
        data = self._byte_content[byte_start:byte_stop]
        return uproot4.source.chunk.Chunk.wrap(self._branch.file.source, data)

    def __getitem__(self, where):
        if uproot4._util.isint(where):
            chunk = self.chunk(where)
            cursor = uproot4.source.cursor.Cursor(0)
            return self._model.read(
                chunk, cursor, self._context, self._branch.file, self._branch
            )

        elif isinstance(where, slice):
            return ObjectArray(
                self._model,
                self._branch,
                self._context,
                self._byte_offsets[where],
                self._byte_content,
            )

        else:
            raise NotImplementedError(repr(where))


class AsObjects(uproot4.interpretation.Interpretation):
    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    def __repr__(self):
        if isinstance(self._model, type):
            model = self._model.__name__
        else:
            model = repr(self._model)
        return "AsObjects({0})".format(model)

    def __eq__(self, other):
        return isinstance(other, AsObjects) and self._model == other._model

    @property
    def numpy_dtype(self):
        return numpy.dtype(numpy.object)

    @property
    def awkward_form(self):
        raise NotImplementedError

    @property
    def cache_key(self):
        content_key = uproot4.stl_containers._content_cache_key(self._model)
        return "{0}({1})".format(type(self).__name__, content_key)

    @property
    def typename(self):
        if isinstance(self._model, uproot4.stl_containers.AsSTLContainer):
            return self._model.typename
        else:
            return uproot4.model.classname_decode(self._model.__name__)[0]

    def basket_array(self, data, byte_offsets, basket, branch, context):
        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
        )

        assert basket.byte_offsets is not None

        output = ObjectArray(self._model, branch, context, byte_offsets, data)

        self.hook_after_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            output=output,
        )

        return output

    def final_array(
        self, basket_arrays, entry_start, entry_stop, entry_offsets, library, branch
    ):
        self.hook_before_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
        )

        output = numpy.empty(entry_stop - entry_start, dtype=numpy.dtype(numpy.object))

        start = entry_offsets[0]
        for basket_num, stop in enumerate(entry_offsets[1:]):
            if start <= entry_start and entry_stop <= stop:
                basket_array = basket_arrays[basket_num]
                for global_i in range(entry_start, entry_stop):
                    local_i = global_i - start
                    output[global_i - entry_start] = basket_array[local_i]

            elif start <= entry_start < stop:
                basket_array = basket_arrays[basket_num]
                for global_i in range(entry_start, stop):
                    local_i = global_i - start
                    output[global_i - entry_start] = basket_array[local_i]

            elif start <= entry_stop <= stop:
                basket_array = basket_arrays[basket_num]
                for global_i in range(start, entry_stop):
                    local_i = global_i - start
                    output[global_i - entry_start] = basket_array[local_i]

            elif entry_start < stop and start <= entry_stop:
                for global_i in range(start, stop):
                    local_i = global_i - start
                    output[global_i - entry_start] = basket_array[local_i]

            start = stop

        self.hook_before_library_finalize(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
        )

        output = library.finalize(output, branch, self)

        self.hook_after_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
        )

        return output

    def simplify(self):
        if isinstance(self._model, uproot4.stl_containers.AsString):
            header_bytes = 0
            if self._model.header:
                header_bytes = 6
            return uproot4.interpretation.strings.AsStrings(
                header_bytes, self._model.length_bytes, self._model.typename
            )

        if isinstance(self._model, uproot4.stl_containers.AsVector):
            if isinstance(self._model.values, numpy.dtype):
                header_bytes = 0
                if self._model.header:
                    header_bytes = 10
                content = uproot4.interpretation.numerical.AsDtype(self._model.values)
                return uproot4.interpretation.jagged.AsJagged(
                    content, header_bytes, self._model.typename
                )

        return self


def _strided_object(path, interpretation, data):
    out = interpretation._model.empty()
    for name, member in interpretation._members:
        p = name
        if len(path) != 0:
            p = path + "/" + name
        if isinstance(member, AsStridedObjects):
            out._members[name] = _strided_object(p, member, data)
        else:
            out._members[name] = data[p]
    return out


class StridedObjectArray(object):
    def __init__(self, interpretation, array):
        self._interpretation = interpretation
        self._array = array

    @property
    def interpretation(self):
        return self._interpretation

    @property
    def array(self):
        return self._array

    def __repr__(self):
        return "StridedObjectArray({0}, {1})".format(self._interpretation, self._array)

    def __len__(self):
        return len(self._array)

    def __getitem__(self, where):
        if uproot4._util.isint(where):
            return _strided_object("", self._interpretation, self._array[where])

        else:
            return StridedObjectArray(self._interpretation, self._array[where])


class CannotBeStrided(Exception):
    pass


def _unravel_members(members):
    out = []
    for name, member in members:
        if isinstance(member, AsStridedObjects):
            for n, m in _unravel_members(member.members):
                out.append((name + "/" + n, m))
        else:
            out.append((name, member))
    return out


class AsStridedObjects(uproot4.interpretation.numerical.AsDtype):
    __slots__ = ["_model", "_members"]

    def __init__(self, model, members):
        self._model = model
        self._members = members
        super(AsStridedObjects, self).__init__(_unravel_members(members))

    @property
    def model(self):
        return self._model

    @property
    def members(self):
        return self._members

    def __repr__(self):
        return "AsStridedObjects({0})".format(self._model.__name__)

    def __eq__(self, other):
        return isinstance(other, AsStridedObjects) and self._model == other._model

    @property
    def numpy_dtype(self):
        return numpy.dtype(numpy.object)

    @property
    def awkward_form(self):
        raise NotImplementedError

    @property
    def cache_key(self):
        return "{0}({1})".format(type(self).__name__, self._model.__name__)

    @property
    def typename(self):
        return uproot4.model.classname_decode(self._model.__name__)[0]

    def _wrap_almost_finalized(self, array):
        return StridedObjectArray(self, array)
