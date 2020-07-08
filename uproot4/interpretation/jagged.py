# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.interpretation


class JaggedArray(object):
    def __init__(self, offsets, content):
        self._offsets = offsets
        self._content = content

    def __repr__(self):
        return "JaggedArray({0}, {1})".format(self._offsets, self._content)

    @property
    def offsets(self):
        return self._offsets

    @property
    def content(self):
        return self._content

    def __getitem__(self, where):
        return self._content[self._offsets[where] : self._offsets[where + 1]]

    def __len__(self):
        return len(self._offsets) - 1

    def __iter__(self):
        start = self._offsets[0]
        content = self._content
        for stop in self._offsets[1:]:
            yield content[start:stop]
            start = stop

    def parents_localindex(self, entry_start, entry_stop):
        counts = self._offsets[1:] - self._offsets[:-1]
        if uproot4._util.win:
            counts = counts.astype(numpy.int32)

        assert entry_stop - entry_start == len(self._offsets) - 1
        indexes = numpy.arange(len(self._offsets) - 1, dtype=numpy.int64)

        parents = numpy.repeat(indexes, counts)

        localindex = numpy.arange(
            self._offsets[0], self._offsets[-1], dtype=numpy.int64
        )
        localindex -= self._offsets[parents]

        return parents + entry_start, localindex


def fast_divide(array, divisor):
    if divisor == 1:
        return array
    elif divisor == 2:
        return numpy.right_shift(array, 1)
    elif divisor == 4:
        return numpy.right_shift(array, 2)
    elif divisor == 8:
        return numpy.right_shift(array, 3)
    else:
        return numpy.floor_divide(array, divisor)


class AsJagged(uproot4.interpretation.Interpretation):
    def __init__(self, content, header_bytes=0, typename=None, original=None):
        if not isinstance(content, uproot4.interpretation.numerical.Numerical):
            raise TypeError("AsJagged content can only be Numerical")
        self._content = content
        self._header_bytes = header_bytes
        self._typename = typename
        self._original = original

    @property
    def content(self):
        return self._content

    @property
    def header_bytes(self):
        return self._header_bytes

    @property
    def original(self):
        return self._original

    def __repr__(self):
        if self._header_bytes == 0:
            return "AsJagged({0})".format(repr(self._content))
        else:
            return "AsJagged({0}, header_bytes={1})".format(
                repr(self._content), self._header_bytes
            )

    def __eq__(self, other):
        return (
            isinstance(other, AsJagged)
            and self._content == other._content
            and self._header_bytes == other._header_bytes
        )

    @property
    def numpy_dtype(self):
        return numpy.dtype(numpy.object)

    def awkward_form(self, file, index_format="i64", header=False, tobject_header=True):
        import awkward1

        return awkward1.forms.ListOffsetForm(
            index_format,
            uproot4._util.awkward_form(
                self._content, file, index_format, header, tobject_header
            ),
            parameters={"uproot": {"as": "jagged", "header_bytes": self._header_bytes}},
        )

    @property
    def cache_key(self):
        return "{0}({1},{2})".format(
            type(self).__name__, self._content.cache_key, self._header_bytes
        )

    @property
    def typename(self):
        if self._typename is None:
            content = self._content.typename
            try:
                i = content.index("[")
                return content[:i] + "[]" + content[i:]
            except ValueError:
                return content + "[]"
        else:
            return self._typename

    def basket_array(self, data, byte_offsets, basket, branch, context, cursor_offset):
        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
        )

        if byte_offsets is None:
            counts = basket.counts
            if counts is None:
                raise uproot4.deserialization.DeserializationError(
                    "missing offsets (and missing count branch) for jagged array",
                    None,
                    None,
                    context,
                    branch.file.file_path,
                )
            else:
                itemsize = self._content.from_dtype.itemsize
                counts = numpy.multiply(counts, itemsize)
                byte_offsets = numpy.empty(len(counts) + 1, dtype=numpy.int32)
                byte_offsets[0] = 0
                numpy.cumsum(counts, out=byte_offsets[1:])

        if self._header_bytes == 0:
            offsets = fast_divide(byte_offsets, self._content.itemsize)
            content = self._content.basket_array(
                data, None, basket, branch, context, cursor_offset
            )
            output = JaggedArray(offsets, content)

        else:
            byte_starts = byte_offsets[:-1] + self._header_bytes
            byte_stops = byte_offsets[1:]

            mask = numpy.zeros(len(data), dtype=numpy.int8)
            mask[byte_starts[byte_starts < len(data)]] = 1
            numpy.add.at(mask, byte_stops[byte_stops < len(data)], -1)
            numpy.cumsum(mask, out=mask)
            data = data[mask.view(numpy.bool_)]

            content = self._content.basket_array(
                data, None, basket, branch, context, cursor_offset
            )

            byte_counts = byte_stops - byte_starts
            counts = fast_divide(byte_counts, self._content.itemsize)

            offsets = numpy.empty(len(counts) + 1, dtype=numpy.int32)
            offsets[0] = 0
            numpy.cumsum(counts, out=offsets[1:])
            output = JaggedArray(offsets, content)

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

        basket_offsets = {}
        basket_content = {}
        for k, v in basket_arrays.items():
            basket_offsets[k] = v.offsets
            basket_content[k] = v.content

        if entry_start >= entry_stop:
            offsets = library.zeros((1,), numpy.int64)
            content = self._content.final_array(
                basket_content, entry_start, entry_stop, entry_offsets, library, branch
            )
            return JaggedArray(offsets, content)

        else:
            length = 0
            start = entry_offsets[0]
            for basket_num, stop in enumerate(entry_offsets[1:]):
                if start <= entry_start and entry_stop <= stop:
                    length += entry_stop - entry_start
                elif start <= entry_start < stop:
                    length += stop - entry_start
                elif start <= entry_stop <= stop:
                    length += entry_stop - start
                elif entry_start < stop and start <= entry_stop:
                    length += stop - start
                start = stop

            offsets = numpy.empty((length + 1,), numpy.int64)

            before = 0
            start = entry_offsets[0]
            contents = []
            for basket_num, stop in enumerate(entry_offsets[1:]):
                if start <= entry_start and entry_stop <= stop:
                    local_start = entry_start - start
                    local_stop = entry_stop - start
                    off, cnt = basket_offsets[basket_num], basket_content[basket_num]
                    offsets[:] = (
                        before - off[local_start] + off[local_start : local_stop + 1]
                    )
                    before += off[local_stop] - off[local_start]
                    contents.append(cnt[off[local_start] : off[local_stop]])

                elif start <= entry_start < stop:
                    local_start = entry_start - start
                    local_stop = stop - start
                    off, cnt = basket_offsets[basket_num], basket_content[basket_num]
                    offsets[: stop - entry_start + 1] = (
                        before - off[local_start] + off[local_start : local_stop + 1]
                    )
                    before += off[local_stop] - off[local_start]
                    contents.append(cnt[off[local_start] : off[local_stop]])

                elif start <= entry_stop <= stop:
                    local_start = 0
                    local_stop = entry_stop - start
                    off, cnt = basket_offsets[basket_num], basket_content[basket_num]
                    offsets[start - entry_start :] = (
                        before - off[local_start] + off[local_start : local_stop + 1]
                    )
                    before += off[local_stop] - off[local_start]
                    contents.append(cnt[off[local_start] : off[local_stop]])

                elif entry_start < stop and start <= entry_stop:
                    off, cnt = basket_offsets[basket_num], basket_content[basket_num]
                    offsets[start - entry_start : stop - entry_start + 1] = (
                        before - off[0] + off
                    )
                    before += off[-1] - off[0]
                    contents.append(cnt[off[0] : off[-1]])

                start = stop

            content = numpy.empty((before,), self.content.to_dtype)
            before = 0
            for cnt in contents:
                content[before : before + len(cnt)] = cnt
                before += len(cnt)

            content = self._content._wrap_almost_finalized(content)

            output = JaggedArray(offsets, content)

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
