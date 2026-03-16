# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Regression test for a bug where FreeSegments.write() used a stale cached
``_allocation`` value instead of the actual serialized size (``num_bytes``).

When FreeSegments.allocate() consumed a free slice exactly (slice size ==
requested bytes), it updated ``_data.slices`` but did not reset the
``_allocation`` cache.  FreeSegments.write() then recorded an inflated
``free_num_bytes`` in the FileHeader.  On the next uproot.update() call,
FreeSegmentsData.deserialize() read that inflated byte count, parsed fewer
entries than expected, and hit ``assert position == num_bytes``.

The fix: use ``self._data.num_bytes`` (always freshly computed) in
FreeSegments.write() instead of ``self._data.allocation`` (cached).
"""

import os

import numpy as np

import uproot

import pytest

hist = pytest.importorskip("hist")


def test_freesegments_consistent_after_repeated_updates(tmp_path):
    n_histograms = 30
    n_events = 1_000
    n_tries = 100

    np.random.seed(27)  # Chosen to trigger with a low number of histograms

    filename = os.path.join(tmp_path, "histograms.root")

    uproot.recreate(filename)

    for i in range(n_histograms):
        h = hist.Hist(hist.axis.Regular(50, 0, 5, name=f"x{i}", label=f"Variable {i}"))

        data = np.random.normal(loc=2.5, scale=0.7, size=n_events)
        h.fill(**{f"x{i}": data})

        with uproot.update(filename) as f:
            f[f"hist_{i}"] = h

    with uproot.open(filename) as f:
        assert set(f.keys(cycle=False)) == {f"hist_{i}" for i in range(n_histograms)}
