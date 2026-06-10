# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import skhep_testdata

import uproot
import uproot.interpretation.numerical as N

# First few values of branch "px1" in tree "events" of uproot-Zmumu.root.
EXPECTED = numpy.array([-41.1952876, 35.1180498, 35.1180498])


def test_interpretation_string_spec_not_byteswapped():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root")) as f:
        array = f["events"]["px1"].array(interpretation="f8", library="np")
    numpy.testing.assert_allclose(array[:3], EXPECTED)


def test_interpretation_numpy_dtype_not_byteswapped():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root")) as f:
        array = f["events"]["px1"].array(interpretation=numpy.dtype("f8"), library="np")
    numpy.testing.assert_allclose(array[:3], EXPECTED)


def test_interpretation_bigendian_to_dtype_preserves_values():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root")) as f:
        array = f["events"]["px1"].array(
            interpretation=N.AsDtype(">f8", ">f8"), library="np"
        )
    numpy.testing.assert_allclose(array[:3], EXPECTED)


def test_interpretation_default_matches_explicit_spec():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root")) as f:
        branch = f["events"]["px1"]
        default = branch.array(library="np")
        explicit = branch.array(interpretation="f8", library="np")
    numpy.testing.assert_array_equal(default, explicit)


def test_interpretation_not_equal_does_not_raise():
    same_a = N.AsDtype("f8")
    same_b = N.AsDtype("f8")
    different = N.AsDtype("f4")

    assert (same_a != same_b) is False
    assert (same_a != different) is True
    assert (same_a == same_b) is True
