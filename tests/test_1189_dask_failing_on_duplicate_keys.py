# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

pytest.importorskip("pandas")


def test_dask_duplicated_keys():

    lazy = uproot.dask(
        skhep_testdata.data_path("uproot-metadata-performance.root") + ":Events"
    )
    materialized = lazy.FatJet_btagDDBvLV2.compute()

    lazy = uproot.dask(skhep_testdata.data_path("uproot-issue513.root") + ":Delphes")
    materialized = lazy.Particle.compute()

    lazy = uproot.dask(
        skhep_testdata.data_path("uproot-issue443.root") + ":muonDataTree"
    )
    materialized = lazy.hitEnd.compute()
