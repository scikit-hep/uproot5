# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.interpretation


class StringArray(uproot4.interpretation.Interpretation):
    def __init__(self, offsets, content):
        self._offsets = offsets
        self._content = content

    def __repr__(self):
        if len(self._content) > 100:
            left, right = self._content[:45], self._content[-45:]
            content = repr(left) + " ... " + repr(right)
        else:
            content = repr(self._content)
        return "StringArray({0}, {1})".format(self._offsets, content)

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


class AsStrings(uproot4.interpretation.Interpretation):
    def __init__(self, header_bytes=0, size_1to5_bytes=False):
        self._header_bytes = header_bytes
        self._size_1to5_bytes = size_1to5_bytes

    @property
    def header_bytes(self):
        return self._header_bytes

    @property
    def size_1to5_bytes(self):
        return self._size_1to5_bytes

    def __repr__(self):
        args = []
        if self._header_bytes != 0:
            args.append("header_bytes={0}".format(self._header_bytes))
        if self._size_1to5_bytes is not False:
            args.append("size_1to5_bytes={0}".format(self._size_1to5_bytes))
        return "AsStrings({0})".format(", ".join(args))

    def __eq__(self, other):
        return (
            isinstance(other, AsStrings)
            and self._header_bytes == other._header_bytes
            and self._size_1to5_bytes == other._size_1to5_bytes
        )

    @property
    def numpy_dtype(self):
        return numpy.dtype(numpy.object)

    @property
    def awkward_form(self):
        raise NotImplementedError

    @property
    def cache_key(self):
        return "{0}({1},{2})".format(
            type(self).__name__, self._header_bytes, self._size_1to5_bytes
        )

    def basket_array(self, data, byte_offsets, basket, branch, context):
        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
        )

        assert basket.byte_offsets is not None

        byte_starts = byte_offsets[:-1] + self._header_bytes
        byte_stops = byte_offsets[1:]

        if self._size_1to5_bytes:
            length_header_size = numpy.ones(len(byte_starts), dtype=numpy.int32)
            length_header_size[data[byte_starts] == 255] += 4
            byte_starts += length_header_size

        mask = numpy.zeros(len(data), dtype=numpy.int8)
        mask[byte_starts[byte_starts < len(data)]] = 1
        numpy.add.at(mask, byte_stops[byte_stops < len(data)], -1)
        numpy.cumsum(mask, out=mask)
        data = data[mask.view(numpy.bool_)]

        counts = byte_stops - byte_starts
        offsets = numpy.empty(len(counts) + 1, dtype=numpy.int32)
        offsets[0] = 0
        numpy.cumsum(counts, out=offsets[1:])

        output = StringArray(offsets, uproot4._util.ensure_str(data.tostring()))

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

        basket_offsets = {}
        basket_content = {}
        for k, v in basket_arrays.items():
            basket_offsets[k] = v.offsets
            basket_content[k] = v.content

        if entry_start >= entry_stop:
            return StringArray(library.zeros((1,), numpy.int64), "")

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
                    offsets[:] = before + off[local_start : local_stop + 1]
                    before += off[local_stop] - off[local_start]
                    contents.append(cnt[off[local_start] : off[local_stop]])

                elif start <= entry_start < stop:
                    local_start = entry_start - start
                    local_stop = stop - start
                    off, cnt = basket_offsets[basket_num], basket_content[basket_num]
                    offsets[: stop - entry_start + 1] = (
                        before + off[local_start : local_stop + 1]
                    )
                    before += off[local_stop] - off[local_start]
                    contents.append(cnt[off[local_start] : off[local_stop]])

                elif start <= entry_stop <= stop:
                    local_start = 0
                    local_stop = entry_stop - start
                    off, cnt = basket_offsets[basket_num], basket_content[basket_num]
                    offsets[start - entry_start :] = (
                        before + off[local_start : local_stop + 1]
                    )
                    before += off[local_stop] - off[local_start]
                    contents.append(cnt[off[local_start] : off[local_stop]])

                elif entry_start < stop and start <= entry_stop:
                    off, cnt = basket_offsets[basket_num], basket_content[basket_num]
                    offsets[start - entry_start : stop - entry_start + 1] = before + off
                    before += off[-1] - off[0]
                    contents.append(cnt[off[0] : off[-1]])

                start = stop

            output = StringArray(offsets, "".join(contents))

            self.hook_before_library_finalize(
                basket_arrays=basket_arrays,
                entry_start=entry_start,
                entry_stop=entry_stop,
                entry_offsets=entry_offsets,
                library=library,
                branch=branch,
                output=output,
            )

        output = library.finalize(output, branch)

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
