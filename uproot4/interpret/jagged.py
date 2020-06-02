# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.interpret


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

    def parents_localindex(self):
        counts = self._offsets[1:] - self._offsets[:-1]
        if uproot4._util.win:
            counts = counts.astype(numpy.int32)
        indexes = numpy.arange(len(self._offsets) - 1, dtype=numpy.int64)

        parents = numpy.repeat(indexes, counts)

        localindex = numpy.arange(
            self._offsets[0], self._offsets[-1], dtype=numpy.int64
        )
        localindex -= self._offsets[parents]

        return parents, localindex


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


class AsJagged(uproot4.interpret.Interpretation):
    def __init__(self, content, header_bytes=0):
        if not isinstance(content, uproot4.interpret.numerical.Numerical):
            raise TypeError("AsJagged content can only be Numerical")
        self._content = content
        self._header_bytes = header_bytes

    @property
    def content(self):
        return self._content

    @property
    def header_bytes(self):
        return self._header_bytes

    def __repr__(self):
        if self._header_bytes == 0:
            return "AsJagged({0})".format(repr(self._content))
        else:
            return "AsJagged({0}, header_bytes={1})".format(
                repr(self._content), self._header_bytes
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
            type(self).__name__, self._content.cache_key, self._header_bytes
        )

    def basket_array(self, data, byte_offsets, basket, branch):
        self.hook_before_basket_array(
            data=data, byte_offsets=byte_offsets, basket=basket, branch=branch
        )

        assert basket.byte_offsets is not None

        if self._header_bytes == 0:
            offsets = fast_divide(basket.byte_offsets, self._content.itemsize)
            content = self._content.basket_array(data, None, basket, branch)
            output = JaggedArray(offsets, content)

        else:
            byte_starts = byte_offsets[:-1] + self._header_bytes
            byte_stops = byte_offsets[1:]

            mask = numpy.zeros(len(data), dtype=numpy.int8)
            mask[byte_starts < len(data)] = 1
            numpy.add.at(mask, byte_stops[byte_stops < len(data)], -1)
            numpy.cumsum(mask, out=mask)
            data = data[mask.view(numpy.bool_)]

            content = self._content.basket_array(data, None, basket, branch)

            byte_counts = byte_stops - byte_starts
            counts = fast_divide(byte_counts, self._content.itemsize)

            offsets = numpy.empty(len(counts) + 1, dtype=numpy.int32)
            offsets[0] = 0
            numpy.cumsum(counts, out=offsets[1:])

            output = JaggedArray(offsets, content)

        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
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

            content = numpy.empty((before,), contents[0].dtype)
            before = 0
            for cnt in contents:
                content[before : before + len(cnt)] = cnt
                before += len(cnt)

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
