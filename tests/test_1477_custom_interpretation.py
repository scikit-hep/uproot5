# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot
import uproot.interpretation.custom


class AsBinary(uproot.interpretation.custom.CustomInterpretation):
    @classmethod
    def match_branch(cls, branch, context, simplify):
        if branch.object_path == "/Events;1:Info/runNum":
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
        return data.view(">u4")


uproot.register_interpretation(AsBinary)


def test_custom_interpretation():
    with uproot.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        runNo = f["Events/Info/runNum"].array()

    with uproot.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        br = f["Events/Info/runNum"]
        assert isinstance(br.interpretation, AsBinary)

        runNo_custom = br.array()
        assert numpy.all(runNo_custom == runNo)


def test_custom_interpretation_entry_range():
    with uproot.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        br = f["Events/Info/runNum"]
        runNo_custom = br.array(entry_start=0, entry_stop=5)
        assert len(runNo_custom) == 5

        runNo_custom = br.array(entry_start=5, entry_stop=10)
        assert len(runNo_custom) == 5

        runNo_custom = br.array(entry_start=0, entry_stop=10)
        assert len(runNo_custom) == 10


def test_repeated_register():
    with pytest.warns(
        UserWarning,
        match="Overwriting existing custom interpretation <class 'tests.test_1477_custom_interpretation.AsBinary'>",
    ):
        uproot.register_interpretation(AsBinary)
