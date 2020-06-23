# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.interpretation
import uproot4.source.chunk
import uproot4.source.cursor
import uproot4._util


class ObjectArray(uproot4.interpretation.Interpretation):
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
        return len(self._offsets) - 1

    def chunk(self, entry_num):
        byte_start = self._byte_offsets[entry_num]
        byte_stop = self._byte_offsets[entry_num + 1]
        data = self._byte_content[byte_start, byte_stop]
        return uproot4.source.chunk.Chunk.wrap(self._branch.file.source, data)

    def __getitem__(self, where):
        if uproot4._util.isint(where):
            chunk = self.chunk(where)
            cursor = uproot4.source.cursor.Cursor(0)
            return self._model.read(
                chunk, cursor, self._context, self._file, self._branch
            )

        elif isinstance(where, slice):
            wheres = range(*where.indicies(len(self)))
            out = numpy.empty(len(wheres), dtype=numpy.object)
            for i in wheres:
                out[i] = self[i]
            return out

        else:
            raise NotImplementedError(repr(where))


class AsObjects(uproot4.interpretation.Interpretation):
    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    def __repr__(self):
        return "AsObjects({0})".format(repr(self._model))

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
