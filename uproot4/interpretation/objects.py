# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.interpretation
import uproot4.interpretation.strings
import uproot4.interpretation.jagged
import uproot4.interpretation.numerical
import uproot4.containers
import uproot4.model
import uproot4.source.chunk
import uproot4.source.cursor
import uproot4._util


class ObjectArray(object):
    def __init__(
        self, model, branch, context, byte_offsets, byte_content, cursor_offset
    ):
        self._model = model
        self._branch = branch
        self._context = context
        self._byte_offsets = byte_offsets
        self._byte_content = byte_content
        self._cursor_offset = cursor_offset

    def __repr__(self):
        return "ObjectArray({0}, {1}, {2}, {3}, {4}, {5})".format(
            self._model,
            self._branch,
            self._context,
            self._byte_offsets,
            self._byte_content,
            self._cursor_offset,
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

    @property
    def cursor_offset(self):
        return self._cursor_offset

    def __len__(self):
        return len(self._byte_offsets) - 1

    def __getitem__(self, where):
        if uproot4._util.isint(where):
            byte_start = self._byte_offsets[where]
            byte_stop = self._byte_offsets[where + 1]
            data = self._byte_content[byte_start:byte_stop]
            chunk = uproot4.source.chunk.Chunk.wrap(self._branch.file.source, data)
            cursor = uproot4.source.cursor.Cursor(
                0, origin=-(byte_start + self._cursor_offset)
            )
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
                self._cursor_offset,
            )

        else:
            raise NotImplementedError(repr(where))

    def __iter__(self):
        source = self._branch.file.source
        context = self._context
        file = self._branch.file
        branch = self._branch
        byte_start = self._byte_offsets[0]
        for byte_stop in self._byte_offsets[1:]:
            data = self._byte_content[byte_start:byte_stop]
            chunk = uproot4.source.chunk.Chunk.wrap(source, data)
            cursor = uproot4.source.cursor.Cursor(0, origin=-self._cursor_offset)
            yield self._model.read(chunk, cursor, context, file, branch)
            byte_start = byte_stop


class AsObjects(uproot4.interpretation.Interpretation):
    def __init__(self, model, branch=None):
        self._model = model
        self._branch = branch

    @property
    def model(self):
        return self._model

    @property
    def branch(self):
        return self._branch

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
    def cache_key(self):
        content_key = uproot4.containers._content_cache_key(self._model)
        return "{0}({1})".format(type(self).__name__, content_key)

    @property
    def typename(self):
        if isinstance(self._model, uproot4.containers.AsContainer):
            return self._model.typename
        else:
            return uproot4.model.classname_decode(self._model.__name__)[0]

    def awkward_form(self, file, index_format="i64", header=False, tobject_header=True):
        if isinstance(self._model, type):
            return self._model.awkward_form(
                self._branch.file, index_format, header, tobject_header
            )
        else:
            return self._model.awkward_form(
                self._branch.file, index_format, header, tobject_header
            )

    def basket_array(self, data, byte_offsets, basket, branch, context, cursor_offset):
        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
        )

        assert basket.byte_offsets is not None

        output = ObjectArray(
            self._model, branch, context, byte_offsets, data, cursor_offset
        )

        self.hook_after_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            output=output,
            cursor_offset=cursor_offset,
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
                basket_array = basket_arrays[basket_num]
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

        output = library.finalize(output, branch, self, entry_start, entry_stop)

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
        if self._branch is not None:
            try:
                return self._model.strided_interpretation(
                    self._branch.file,
                    header=False,
                    tobject_header=True,
                    original=self._model,
                )
            except CannotBeStrided:
                pass

        if isinstance(self._model, uproot4.containers.AsString):
            header_bytes = 0
            if self._model.header:
                header_bytes = 6
            return uproot4.interpretation.strings.AsStrings(
                header_bytes,
                self._model.length_bytes,
                self._model.typename,
                original=self._model,
            )

        if isinstance(
            self._model, (uproot4.containers.AsArray, uproot4.containers.AsVector),
        ):
            if not self._model.header:
                header_bytes = 0
            elif isinstance(self._model, uproot4.containers.AsArray):
                header_bytes = 1
            else:
                header_bytes = 10

            if isinstance(self._model.values, numpy.dtype):
                content = uproot4.interpretation.numerical.AsDtype(self._model.values)
                return uproot4.interpretation.jagged.AsJagged(
                    content, header_bytes, self._model.typename, original=self._model
                )

            if self._branch is not None:
                try:
                    content = self._model.values.strided_interpretation(
                        self._branch.file,
                        header=False,
                        tobject_header=True,
                        original=self._model.values,
                    )
                    return uproot4.interpretation.jagged.AsJagged(
                        content,
                        header_bytes,
                        self._model.typename,
                        original=self._model,
                    )
                except CannotBeStrided:
                    pass

        return self


class CannotBeStrided(Exception):
    pass


class CannotBeAwkward(Exception):
    def __init__(self, because):
        self.because = because


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

    def __iter__(self):
        interpretation = self._interpretation
        for x in self._array:
            yield _strided_object("", interpretation, x)


def _unravel_members(members):
    out = []
    for name, member in members:
        if isinstance(member, AsStridedObjects):
            for n, m in _unravel_members(member.members):
                out.append((name + "/" + n, m))
        else:
            out.append((name, member))
    return out


def _strided_awkward_form(
    awkward1, classname, members, file, index_format, header, tobject_header
):
    contents = {}
    for name, member in members:
        if isinstance(member, AsStridedObjects):
            cname = uproot4.model.classname_decode(member._model.__name__)[0]
            contents[name] = _strided_awkward_form(
                awkward1,
                cname,
                member._members,
                file,
                index_format,
                header,
                tobject_header,
            )
        else:
            contents[name] = uproot4._util.awkward_form(
                member, file, index_format, header, tobject_header
            )
    return awkward1.forms.RecordForm(contents, parameters={"__record__": classname})


class AsStridedObjects(uproot4.interpretation.numerical.AsDtype):
    def __init__(self, model, members, original=None):
        self._model = model
        self._members = members
        self._original = original
        super(AsStridedObjects, self).__init__(_unravel_members(members))

    @property
    def model(self):
        return self._model

    @property
    def members(self):
        return self._members

    @property
    def original(self):
        return self._original

    def __repr__(self):
        return "AsStridedObjects({0})".format(self._model.__name__)

    def __eq__(self, other):
        return isinstance(other, AsStridedObjects) and self._model == other._model

    @property
    def numpy_dtype(self):
        return numpy.dtype(numpy.object)

    def awkward_form(self, file, index_format="i64", header=False, tobject_header=True):
        import awkward1

        cname = uproot4.model.classname_decode(self._model.__name__)[0]
        return _strided_awkward_form(
            awkward1, cname, self._members, file, index_format, header, tobject_header
        )

    @property
    def cache_key(self):
        return "{0}({1})".format(type(self).__name__, self._model.__name__)

    @property
    def typename(self):
        return uproot4.model.classname_decode(self._model.__name__)[0]

    def _wrap_almost_finalized(self, array):
        return StridedObjectArray(self, array)
