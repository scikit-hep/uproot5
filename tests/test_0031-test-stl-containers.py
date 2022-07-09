# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import json
import sys

import numpy
import pytest
import skhep_testdata

import uproot
from uproot.containers import AsMap, AsSet, AsString, AsVector
from uproot.interpretation.jagged import AsJagged
from uproot.interpretation.numerical import AsDtype
from uproot.interpretation.objects import AsObjects


@pytest.mark.skip(
    reason="Implement non-memberwise std::map; we have a sample (map<string,double>)"
)
def test_nonmemberwise_asmap():
    with uproot.open(skhep_testdata.data_path("uproot-issue243.root")) as file:
        branch = file["triggerList/triggerMap"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        py = branch.array(library="np", entry_stop=2)

    with uproot.open(skhep_testdata.data_path("uproot-issue-268.root")) as file:
        branch = file["aTree/VtxTracks/channels_"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        py = branch.array(library="np", entry_stop=2)

    with uproot.open(skhep_testdata.data_path("uproot-issue-268.root")) as file:
        branch = file["aTree/VtxTracks2TofHits/match_"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        py = branch.array(library="np", entry_stop=2)

    with uproot.open(skhep_testdata.data_path("uproot-issue-268.root")) as file:
        branch = file["aTree/VtxTracks2TofHits/match_inverted_"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        py = branch.array(library="np", entry_stop=2)


def test_typename():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert tree["vector_int32"].interpretation == AsJagged(AsDtype(">i4"), 10)
        assert tree["vector_string"].interpretation == AsObjects(
            AsVector(True, AsString(False))
        )
        assert tree["vector_vector_int32"].interpretation == AsObjects(
            AsVector(True, AsVector(False, numpy.dtype(">i4")))
        )
        assert tree["vector_vector_string"].interpretation == AsObjects(
            AsVector(True, AsVector(False, AsString(False)))
        )
        assert tree["vector_set_int32"].interpretation == AsObjects(
            AsVector(True, AsSet(False, numpy.dtype(">i4")))
        )
        assert tree["vector_set_string"].interpretation == AsObjects(
            AsVector(True, AsSet(False, AsString(False)))
        )
        assert tree["set_int32"].interpretation == AsObjects(
            AsSet(True, numpy.dtype(">i4"))
        )
        assert tree["set_string"].interpretation == AsObjects(
            AsSet(True, AsString(False))
        )
        assert tree["map_int32_int16"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), numpy.dtype(">i2"))
        )
        assert tree["map_int32_vector_int16"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), AsVector(True, numpy.dtype(">i2")))
        )
        assert tree["map_int32_vector_string"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), AsVector(True, AsString(False)))
        )
        assert tree["map_int32_set_int16"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), AsSet(True, numpy.dtype(">i2")))
        )
        assert tree["map_int32_set_string"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), AsSet(True, AsString(False)))
        )
        assert tree["map_string_int16"].interpretation == AsObjects(
            AsMap(True, AsString(True), numpy.dtype(">i2"))
        )
        assert tree["map_string_vector_int16"].interpretation == AsObjects(
            AsMap(True, AsString(True), AsVector(True, numpy.dtype(">i2")))
        )
        assert tree["map_string_vector_string"].interpretation == AsObjects(
            AsMap(True, AsString(True), AsVector(True, AsString(False)))
        )
        assert tree["map_string_set_int16"].interpretation == AsObjects(
            AsMap(True, AsString(True), AsSet(True, numpy.dtype(">i2")))
        )
        assert tree["map_string_set_string"].interpretation == AsObjects(
            AsMap(True, AsString(True), AsSet(True, AsString(False)))
        )
        assert tree["map_int32_vector_vector_int16"].interpretation == AsObjects(
            AsMap(
                True,
                numpy.dtype(">i4"),
                AsVector(True, AsVector(False, numpy.dtype(">i2"))),
            )
        )
        assert tree["map_int32_vector_set_int16"].interpretation == AsObjects(
            AsMap(
                True,
                numpy.dtype(">i4"),
                AsVector(True, AsSet(False, numpy.dtype(">i2"))),
            )
        )
        assert tree["map_string_string"].interpretation == AsObjects(
            AsMap(True, AsString(True), AsString(True))
        )


def test_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert tree["string"].array(library="np").tolist() == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_tstring():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert tree["tstring"].array(library="np").tolist() == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_vector_int32():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["vector_int32"].array(library="np")] == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


def test_vector_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["vector_string"].array(library="np")] == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


def test_vector_tstring():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["vector_tstring"].array(library="np")] == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


def test_vector_vector_int32():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["vector_vector_int32"].array(library="np")
        ] == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_vector_vector_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["vector_vector_string"].array(library="np")
        ] == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two"], ["one", "two", "three"]],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
            ],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
                ["one", "two", "three", "four", "five"],
            ],
        ]


def test_vector_set_int32():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["vector_set_int32"].array(library="np")] == [
            [{1}],
            [{1}, {1, 2}],
            [{1}, {1, 2}, {1, 2, 3}],
            [{1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}],
            [
                {1},
                {1, 2},
                {1, 2, 3},
                {1, 2, 3, 4},
                {1, 2, 3, 4, 5},
            ],
        ]


def test_vector_set_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["vector_set_string"].array(library="np")] == [
            [{"one"}],
            [{"one"}, {"one", "two"}],
            [{"one"}, {"one", "two"}, {"one", "two", "three"}],
            [
                {"one"},
                {"one", "two"},
                {"one", "two", "three"},
                {"one", "two", "three", "four"},
            ],
            [
                {"one"},
                {"one", "two"},
                {"one", "two", "three"},
                {"one", "two", "three", "four"},
                {"one", "two", "three", "four", "five"},
            ],
        ]


def test_set_int32():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["set_int32"].array(library="np")] == [
            {1},
            {1, 2},
            {1, 2, 3},
            {1, 2, 3, 4},
            {1, 2, 3, 4, 5},
        ]


def test_set_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["set_string"].array(library="np")] == [
            {"one"},
            {"one", "two"},
            {"one", "two", "three"},
            {"one", "two", "three", "four"},
            {"one", "two", "three", "four", "five"},
        ]


def test_map_int32_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["map_int32_int16"].array(library="np")] == [
            {1: 1},
            {1: 1, 2: 2},
            {1: 1, 2: 2, 3: 3},
            {1: 1, 2: 2, 3: 3, 4: 4},
            {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        ]


def test_map_int32_vector_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_int32_vector_int16"].array(library="np")
        ] == [
            {1: [1]},
            {1: [1], 2: [1, 2]},
            {1: [1], 2: [1, 2], 3: [1, 2, 3]},
            {1: [1], 2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 3, 4]},
            {1: [1], 2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4, 5]},
        ]


def test_map_int32_vector_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_int32_vector_string"].array(library="np")
        ] == [
            {1: ["one"]},
            {1: ["one"], 2: ["one", "two"]},
            {1: ["one"], 2: ["one", "two"], 3: ["one", "two", "three"]},
            {
                1: ["one"],
                2: ["one", "two"],
                3: ["one", "two", "three"],
                4: ["one", "two", "three", "four"],
            },
            {
                1: ["one"],
                2: ["one", "two"],
                3: ["one", "two", "three"],
                4: ["one", "two", "three", "four"],
                5: ["one", "two", "three", "four", "five"],
            },
        ]


def test_map_int32_set_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_int32_set_int16"].array(library="np")
        ] == [
            {1: {1}},
            {1: {1}, 2: {1, 2}},
            {1: {1}, 2: {1, 2}, 3: {1, 2, 3}},
            {1: {1}, 2: {1, 2}, 3: {1, 2, 3}, 4: {1, 2, 3, 4}},
            {
                1: {1},
                2: {1, 2},
                3: {1, 2, 3},
                4: {1, 2, 3, 4},
                5: {1, 2, 3, 4, 5},
            },
        ]


def test_map_int32_set_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_int32_set_string"].array(library="np")
        ] == [
            {1: {"one"}},
            {1: {"one"}, 2: {"one", "two"}},
            {1: {"one"}, 2: {"one", "two"}, 3: {"one", "two", "three"}},
            {
                1: {"one"},
                2: {"one", "two"},
                3: {"one", "two", "three"},
                4: {"one", "two", "three", "four"},
            },
            {
                1: {"one"},
                2: {"one", "two"},
                3: {"one", "two", "three"},
                4: {"one", "two", "three", "four"},
                5: {"one", "two", "three", "four", "five"},
            },
        ]


def test_map_string_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["map_string_int16"].array(library="np")] == [
            {"one": 1},
            {"one": 1, "two": 2},
            {"one": 1, "two": 2, "three": 3},
            {"one": 1, "two": 2, "three": 3, "four": 4},
            {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5},
        ]


def test_map_string_vector_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_string_vector_int16"].array(library="np")
        ] == [
            {"one": [1]},
            {"one": [1], "two": [1, 2]},
            {"one": [1], "two": [1, 2], "three": [1, 2, 3]},
            {"one": [1], "two": [1, 2], "three": [1, 2, 3], "four": [1, 2, 3, 4]},
            {
                "one": [1],
                "two": [1, 2],
                "three": [1, 2, 3],
                "four": [1, 2, 3, 4],
                "five": [1, 2, 3, 4, 5],
            },
        ]


def test_map_string_vector_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_string_vector_string"].array(library="np")
        ] == [
            {"one": ["one"]},
            {"one": ["one"], "two": ["one", "two"]},
            {"one": ["one"], "two": ["one", "two"], "three": ["one", "two", "three"]},
            {
                "one": ["one"],
                "two": ["one", "two"],
                "three": ["one", "two", "three"],
                "four": ["one", "two", "three", "four"],
            },
            {
                "one": ["one"],
                "two": ["one", "two"],
                "three": ["one", "two", "three"],
                "four": ["one", "two", "three", "four"],
                "five": ["one", "two", "three", "four", "five"],
            },
        ]


def test_map_string_set_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_string_set_int16"].array(library="np")
        ] == [
            {"one": {1}},
            {"one": {1}, "two": {1, 2}},
            {"one": {1}, "two": {1, 2}, "three": {1, 2, 3}},
            {
                "one": {1},
                "two": {1, 2},
                "three": {1, 2, 3},
                "four": {1, 2, 3, 4},
            },
            {
                "one": {1},
                "two": {1, 2},
                "three": {1, 2, 3},
                "four": {1, 2, 3, 4},
                "five": {1, 2, 3, 4, 5},
            },
        ]


def test_map_string_set_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_string_set_string"].array(library="np")
        ] == [
            {"one": {"one"}},
            {"one": {"one"}, "two": {"one", "two"}},
            {
                "one": {"one"},
                "two": {"one", "two"},
                "three": {"one", "two", "three"},
            },
            {
                "one": {"one"},
                "two": {"one", "two"},
                "three": {"one", "two", "three"},
                "four": {"one", "two", "three", "four"},
            },
            {
                "one": {"one"},
                "two": {"one", "two"},
                "three": {"one", "two", "three"},
                "four": {"one", "two", "three", "four"},
                "five": {"one", "two", "three", "four", "five"},
            },
        ]


def test_map_int32_vector_vector_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist()
            for x in tree["map_int32_vector_vector_int16"].array(library="np")
        ] == [
            {1: [[1]]},
            {1: [[1]], 2: [[1], [1, 2]]},
            {1: [[1]], 2: [[1], [1, 2]], 3: [[1], [1, 2], [1, 2, 3]]},
            {
                1: [[1]],
                2: [[1], [1, 2]],
                3: [[1], [1, 2], [1, 2, 3]],
                4: [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            },
            {
                1: [[1]],
                2: [[1], [1, 2]],
                3: [[1], [1, 2], [1, 2, 3]],
                4: [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                5: [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
            },
        ]


def test_map_int32_vector_set_int16():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_int32_vector_set_int16"].array(library="np")
        ] == [
            {1: [{1}]},
            {1: [{1}], 2: [{1}, {1, 2}]},
            {
                1: [{1}],
                2: [{1}, {1, 2}],
                3: [{1}, {1, 2}, {1, 2, 3}],
            },
            {
                1: [{1}],
                2: [{1}, {1, 2}],
                3: [{1}, {1, 2}, {1, 2, 3}],
                4: [{1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}],
            },
            {
                1: [{1}],
                2: [{1}, {1, 2}],
                3: [{1}, {1, 2}, {1, 2, 3}],
                4: [{1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}],
                5: [
                    {1},
                    {1, 2},
                    {1, 2, 3},
                    {1, 2, 3, 4},
                    {1, 2, 3, 4, 5},
                ],
            },
        ]


def test_map_string_string():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["map_string_string"].array(library="np")] == [
            {"one": "ONE"},
            {"one": "ONE", "two": "TWO"},
            {"one": "ONE", "two": "TWO", "three": "THREE"},
            {"one": "ONE", "two": "TWO", "three": "THREE", "four": "FOUR"},
            {
                "one": "ONE",
                "two": "TWO",
                "three": "THREE",
                "four": "FOUR",
                "five": "FIVE",
            },
        ]


def test_map_string_tstring():
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["map_string_tstring"].array(library="np")] == [
            {"one": "ONE"},
            {"one": "ONE", "two": "TWO"},
            {"one": "ONE", "two": "TWO", "three": "THREE"},
            {"one": "ONE", "two": "TWO", "three": "THREE", "four": "FOUR"},
            {
                "one": "ONE",
                "two": "TWO",
                "three": "THREE",
                "four": "FOUR",
                "five": "FIVE",
            },
        ]


def test_map_int_struct():
    with uproot.open(skhep_testdata.data_path("uproot-issue468.root"))[
        "Geant4Data/Geant4Data./Geant4Data.particles"
    ] as branch:
        assert (
            repr(branch.interpretation) == "AsObjects(AsMap(True, dtype('>i4'), "
            "Model_BDSOutputROOTGeant4Data_3a3a_ParticleInfo))"
        )
        result = branch.array(library="np")[0]
        assert result.keys().tolist() == [
            -1000020040,
            -1000020030,
            -1000010030,
            -1000010020,
            -2212,
            -2112,
            -321,
            -211,
            -16,
            -15,
            -14,
            -13,
            -12,
            -11,
            0,
            11,
            12,
            13,
            14,
            15,
            16,
            22,
            211,
            321,
            2112,
            2212,
            1000010020,
            1000010030,
            1000020030,
            1000020040,
        ]
        assert [x.member("name") for x in result.values()] == [
            "anti_alpha",
            "anti_He3",
            "anti_triton",
            "anti_deuteron",
            "anti_proton",
            "anti_neutron",
            "kaon-",
            "pi-",
            "anti_nu_tau",
            "tau+",
            "anti_nu_mu",
            "mu+",
            "anti_nu_e",
            "e+",
            "geantino",
            "e-",
            "nu_e",
            "mu-",
            "nu_mu",
            "tau-",
            "nu_tau",
            "gamma",
            "pi+",
            "kaon+",
            "neutron",
            "proton",
            "deuteron",
            "triton",
            "He3",
            "alpha",
        ]
        assert [x.member("charge") for x in result.values()] == [
            -2,
            -2,
            -1,
            -1,
            -1,
            0,
            -1,
            -1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            -1,
            0,
            -1,
            0,
            -1,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            2,
            2,
        ]
        assert [x.member("mass") for x in result.values()] == [
            3.727379,
            2.808391,
            2.808921,
            1.875613,
            0.938272013,
            0.93956536,
            0.493677,
            0.1395701,
            0.0,
            1.77686,
            0.0,
            0.1056583715,
            0.0,
            0.00051099891,
            0.0,
            0.00051099891,
            0.0,
            0.1056583715,
            0.0,
            1.77686,
            0.0,
            0.0,
            0.1395701,
            0.493677,
            0.93956536,
            0.938272013,
            1.875613,
            2.808921,
            2.808391,
            3.727379,
        ]
