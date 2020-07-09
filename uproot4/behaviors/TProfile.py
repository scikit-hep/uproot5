# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.models.TArray


class TProfile(object):
    @property
    def np(self):
        bin_entries = self.member("fBinEntries")
        bin_entries = numpy.array(
            bin_entries, dtype=bin_entries.dtype.newbyteorder("=")
        )

        (values,) = self.base(uproot4.models.TArray.Model_TArray)
        values = numpy.array(values, dtype=values.dtype.newbyteorder("="))

        print(self.all_members)

        return values / bin_entries

    @property
    def bh(self):
        raise NotImplementedError(repr(self))
