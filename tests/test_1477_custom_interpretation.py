# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import re

import numpy
import pytest
import skhep_testdata

import uproot
import uproot.interpretation.custom

awkward = uproot.extras.awkward()
pandas = uproot.extras.pandas()


class AsUint32(uproot.interpretation.custom.CustomInterpretation):
    @classmethod
    def match_branch(cls, branch, context, simplify):
        if branch.object_path == "/Events;1:Info/evtNum":
            return True

    def basket_array(
        self,
        data,
        byte_offsets,
        basket,
        branch,
        context,
        cursor_offset,
        library,
        interp_options,
    ):
        np_data = data.view(">u4").astype("uint32")

        if library.name == "np":
            return np_data
        elif library.name == "ak":
            return awkward.from_numpy(np_data)
        elif library.name == "pd":
            return pandas.Series(np_data)
        else:
            raise ValueError(f"Unsupported library: {library.name}")


# During the test, registration can only be done once, since the
# uproot will not be reloaded across tests.
uproot.register_interpretation(AsUint32)


def test_registration_and_use():
    with uproot.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        br2 = f["Events/Info/evtNum"]
        assert isinstance(br2.interpretation, AsUint32)

        evtNum_ak = br2.array()
        assert evtNum_ak.tolist() == [
            135353219,
            135353222,
            135353225,
            135353230,
            135353239,
            135353242,
            135353244,
            135353247,
            135353252,
            135353256,
        ]

        evtNum_np = br2.array(library="np")
        assert isinstance(evtNum_np, numpy.ndarray)
        assert numpy.all(evtNum_np == evtNum_ak)

        evtNum_pd = br2.array(library="pd")
        assert isinstance(evtNum_pd, pandas.Series)
        assert numpy.all(evtNum_pd.values == evtNum_np)


def test_repeated_register():
    with pytest.warns(
        UserWarning,
        match="Overwriting existing custom interpretation <class 'tests.test_1477_custom_interpretation.AsUint32'>",
    ):
        uproot.register_interpretation(AsUint32)


def test_entry_range():
    with uproot.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        br = f["Events/Info/runNum"]
        runNo_custom = br.array(entry_start=0, entry_stop=6)
        assert len(runNo_custom) == 6

        runNo_custom = br.array(entry_start=6, entry_stop=10)
        assert len(runNo_custom) == 4

        runNo_custom = br.array(entry_start=0, entry_stop=10)
        assert len(runNo_custom) == 10
