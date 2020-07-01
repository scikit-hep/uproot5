# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4
from uproot4.interpretation.numerical import AsDtype
from uproot4.interpretation.jagged import AsJagged


def test_histograms_outside_of_ttrees():
    with uproot4.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        contents = numpy.asarray(f["hpx"].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 2417.0)

        contents = numpy.asarray(f["hpxpy"].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 497.0)

        contents = numpy.asarray(f["hprof"].bases[-1].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 3054.7299575805664)

        numpy.asarray(f["ntuple"])


def test_gohep_nosplit_file():
    with uproot4.open(skhep_testdata.data_path("uproot-small-evnt-tree-nosplit.root"))[
        "tree/evt"
    ] as branch:
        result = branch.array(library="np", entry_start=5, entry_stop=6)[0]
        assert result.member("Beg") == "beg-005"
        assert result.member("I16") == 5
        assert result.member("I32") == 5
        assert result.member("I64") == 5
        assert result.member("U16") == 5
        assert result.member("U32") == 5
        assert result.member("U64") == 5
        assert result.member("F32") == 5.0
        assert result.member("F64") == 5.0
        assert result.member("Str") == "evt-005"
        assert result.member("P3").member("Px") == 4
        assert result.member("P3").member("Py") == 5.0
        assert result.member("P3").member("Pz") == 4
        assert result.member("ArrayI16").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayU16").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayI32").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayU32").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayI64").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayU64").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayF32").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("ArrayF32").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("ArrayF64").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("ArrayF64").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("StdStr") == "std-005"
        assert result.member("SliceI16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceI32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceI64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceU16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceU32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceU64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecI16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecI32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecI64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecU16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecU32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecU64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecF32").tolist() == [5.0, 5.0, 5.0, 5.0, 5.0]
        assert result.member("StlVecF64").tolist() == [5.0, 5.0, 5.0, 5.0, 5.0]
        assert result.member("StlVecStr").tolist() == [
            "vec-005",
            "vec-005",
            "vec-005",
            "vec-005",
            "vec-005",
        ]
        assert result.member("End") == "end-005"


def test_TLorentzVectors_show():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        tree.show()


def test_TVector2():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["MET"].array(library="np", entry_stop=1)[0]
        assert (result.member("fX"), result.member("fY")) == (
            5.912771224975586,
            2.5636332035064697,
        )


def test_vector_TLorentzVector():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["muonp4"].array(library="np", entry_stop=1)[0]
        assert len(result) == 2
        assert (
            result[0].member("fE"),
            result[0].member("fP").member("fX"),
            result[0].member("fP").member("fY"),
            result[0].member("fP").member("fZ"),
        ) == (
            54.77949905395508,
            -52.89945602416992,
            -11.654671669006348,
            -8.16079330444336,
        )


def test_strided():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        assert tree.file.class_named("TLorentzVector", "max").strided_interpretation(
            tree.file
        ).from_dtype == numpy.dtype(
            [
                ("@instance_version", numpy.dtype(">u2")),
                ("@num_bytes", numpy.dtype(">u4")),
                ("@fUniqueID", numpy.dtype(">u4")),
                ("@fBits", numpy.dtype(">u4")),
                ("@pidf", numpy.dtype(">u2")),
                ("fP/@instance_version", numpy.dtype(">u2")),
                ("fP/@num_bytes", numpy.dtype(">u4")),
                ("fP/@fUniqueID", numpy.dtype(">u4")),
                ("fP/@fBits", numpy.dtype(">u4")),
                ("fP/@pidf", numpy.dtype(">u2")),
                ("fP/fX", ">f8"),
                ("fP/fY", ">f8"),
                ("fP/fZ", ">f8"),
                ("fE", ">f8"),
            ]
        )


def test_read_strided_TVector2():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        interp = tree.file.class_named("TVector2", "max").strided_interpretation(
            tree.file
        )

        assert tree["MET"].interpretation == interp

        result = tree["MET"].array(entry_stop=1, library="np")[0]
        assert (result.member("fX"), result.member("fY")) == (
            5.912771224975586,
            2.5636332035064697,
        )


def test_read_strided_TLorentzVector():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        interp = tree.file.class_named("TLorentzVector", "max").strided_interpretation(
            tree.file
        )
        interp = uproot4.interpretation.jagged.AsJagged(interp, header_bytes=10)

        assert tree["muonp4"].interpretation == interp

        result = tree["muonp4"].array(library="np", entry_stop=1)[0]
        assert len(result) == 2
        assert (
            result[0].member("fE"),
            result[0].member("fP").member("fX"),
            result[0].member("fP").member("fY"),
            result[0].member("fP").member("fZ"),
        ) == (
            54.77949905395508,
            -52.89945602416992,
            -11.654671669006348,
            -8.16079330444336,
        )


def test_strided_awkward():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["MET"].array(library="ak")

        assert (
            repr(awkward1.type(result))
            == '2421 * TVector2["fX": float64, "fY": float64]'
        )

        assert awkward1.to_list(result["fX"][:10]) == [
            5.912771224975586,
            24.76520347595215,
            -25.78508758544922,
            8.619895935058594,
            5.393138885498047,
            -3.7594752311706543,
            23.962148666381836,
            -57.533348083496094,
            42.416194915771484,
            -1.9144694805145264,
        ]
        assert awkward1.to_list(result["fY"][:10]) == [
            2.5636332035064697,
            -16.349109649658203,
            16.237131118774414,
            -22.78654670715332,
            -1.3100523948669434,
            -19.417020797729492,
            -9.049156188964844,
            -20.48767852783203,
            -94.35086059570312,
            -23.96303367614746,
        ]


def test_jagged_strided_awkward():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["muonp4"].array(library="ak")

        assert (
            repr(awkward1.type(result))
            == '2421 * var * TLorentzVector["fP": TVector3["fX": float64, '
            '"fY": float64, "fZ": float64], "fE": float64]'
        )

        assert result[0, 0, "fE"] == 54.77949905395508
        assert result[0, 0, "fP", "fX"] == -52.89945602416992
        assert result[0, 0, "fP", "fY"] == -11.654671669006348
        assert result[0, 0, "fP", "fZ"] == -8.16079330444336


def test_jagged_awkward_1():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree/evt/SliceU64"] as branch:
        assert branch.interpretation == AsJagged(AsDtype(">u8"), header_bytes=1)
        result = branch.array(library="ak", entry_stop=6)
        assert awkward1.to_list(result) == [
            [],
            [1],
            [2, 2],
            [3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5, 5],
        ]


def test_jagged_awkward_2():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree/evt/StlVecF64"] as branch:
        assert branch.interpretation == AsJagged(AsDtype(">f8"), header_bytes=10)
        result = branch.array(library="ak", entry_stop=6)
        assert awkward1.to_list(result) == [
            [],
            [1],
            [2, 2],
            [3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5, 5],
        ]


def test_general_awkward_form():
    with uproot4.open(skhep_testdata.data_path("uproot-small-evnt-tree-nosplit.root"))[
        "tree/evt"
    ] as branch:
        assert json.loads(
            branch.interpretation.awkward_form.tojson(verbose=False)
        ) == json.loads(
            """{
    "class": "RecordArray",
    "contents": {
        "ArrayF32": {
            "class": "RegularArray",
            "content": "float32",
            "size": 10
        },
        "ArrayF64": {
            "class": "RegularArray",
            "content": "float64",
            "size": 10
        },
        "ArrayI16": {
            "class": "RegularArray",
            "content": "int16",
            "size": 10
        },
        "ArrayI32": {
            "class": "RegularArray",
            "content": "int32",
            "size": 10
        },
        "ArrayI64": {
            "class": "RegularArray",
            "content": "int64",
            "size": 10
        },
        "ArrayU16": {
            "class": "RegularArray",
            "content": "uint16",
            "size": 10
        },
        "ArrayU32": {
            "class": "RegularArray",
            "content": "uint32",
            "size": 10
        },
        "ArrayU64": {
            "class": "RegularArray",
            "content": "uint64",
            "size": 10
        },
        "Beg": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "uint8",
            "parameters": {
                "__array__": "string",
                "uproot": {
                    "as": "string",
                    "header": false,
                    "length_bytes": "1-5"
                }
            }
        },
        "End": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "uint8",
            "parameters": {
                "__array__": "string",
                "uproot": {
                    "as": "string",
                    "header": false,
                    "length_bytes": "1-5"
                }
            }
        },
        "F32": "float32",
        "F64": "float64",
        "I16": "int16",
        "I32": "int32",
        "I64": "int64",
        "N": "uint32",
        "P3": {
            "class": "RecordArray",
            "contents": {
                "Px": "int32",
                "Py": "float64",
                "Pz": "int32"
            },
            "parameters": {
                "__hidden_prefix__": "@",
                "__record__": "P3"
            }
        },
        "SliceF32": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "float32",
            "parameters": {
                "uproot": {
                    "as": "TStreamerBasicPointer",
                    "count_name": "N"
                }
            }
        },
        "SliceF64": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "float64",
            "parameters": {
                "uproot": {
                    "as": "TStreamerBasicPointer",
                    "count_name": "N"
                }
            }
        },
        "SliceI16": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "int16",
            "parameters": {
                "uproot": {
                    "as": "TStreamerBasicPointer",
                    "count_name": "N"
                }
            }
        },
        "SliceI32": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "int32",
            "parameters": {
                "uproot": {
                    "as": "TStreamerBasicPointer",
                    "count_name": "N"
                }
            }
        },
        "SliceI64": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "int64",
            "parameters": {
                "uproot": {
                    "as": "TStreamerBasicPointer",
                    "count_name": "N"
                }
            }
        },
        "SliceU16": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "uint16",
            "parameters": {
                "uproot": {
                    "as": "TStreamerBasicPointer",
                    "count_name": "N"
                }
            }
        },
        "SliceU32": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "uint32",
            "parameters": {
                "uproot": {
                    "as": "TStreamerBasicPointer",
                    "count_name": "N"
                }
            }
        },
        "SliceU64": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "uint64",
            "parameters": {
                "uproot": {
                    "as": "TStreamerBasicPointer",
                    "count_name": "N"
                }
            }
        },
        "StdStr": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "uint8",
            "parameters": {
                "__array__": "string",
                "uproot": {
                    "as": "string",
                    "header": true,
                    "length_bytes": "1-5"
                }
            }
        },
        "StlVecF32": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "float32",
            "parameters": {
                "uproot": {
                    "as": "vector",
                    "header": true
                }
            }
        },
        "StlVecF64": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "float64",
            "parameters": {
                "uproot": {
                    "as": "vector",
                    "header": true
                }
            }
        },
        "StlVecI16": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "int16",
            "parameters": {
                "uproot": {
                    "as": "vector",
                    "header": true
                }
            }
        },
        "StlVecI32": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "int32",
            "parameters": {
                "uproot": {
                    "as": "vector",
                    "header": true
                }
            }
        },
        "StlVecI64": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "int64",
            "parameters": {
                "uproot": {
                    "as": "vector",
                    "header": true
                }
            }
        },
        "StlVecStr": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": {
                "class": "ListOffsetArray32",
                "offsets": "i32",
                "content": "uint8",
                "parameters": {
                    "__array__": "string",
                    "uproot": {
                        "as": "string",
                        "header": false,
                        "length_bytes": "1-5"
                    }
                }
            },
            "parameters": {
                "uproot": {
                    "as": "vector",
                    "header": true
                }
            }
        },
        "StlVecU16": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "uint16",
            "parameters": {
                "uproot": {
                    "as": "vector",
                    "header": true
                }
            }
        },
        "StlVecU32": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "uint32",
            "parameters": {
                "uproot": {
                    "as": "vector",
                    "header": true
                }
            }
        },
        "StlVecU64": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "uint64",
            "parameters": {
                "uproot": {
                    "as": "vector",
                    "header": true
                }
            }
        },
        "Str": {
            "class": "ListOffsetArray32",
            "offsets": "i32",
            "content": "uint8",
            "parameters": {
                "__array__": "string",
                "uproot": {
                    "as": "string",
                    "header": false,
                    "length_bytes": "1-5"
                }
            }
        },
        "U16": "uint16",
        "U32": "uint32",
        "U64": "uint64"
    },
    "parameters": {
        "__hidden_prefix__": "@",
        "__record__": "Event"
    }
}"""
        )


def test_jagged_awkward_3():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree/evt/StlVecStr"] as branch:
        assert awkward1.to_list(branch.array(library="ak")[:6, :3]) == [
            [],
            ["vec-001"],
            ["vec-002", "vec-002"],
            ["vec-003", "vec-003", "vec-003"],
            ["vec-004", "vec-004", "vec-004"],
            ["vec-005", "vec-005", "vec-005"],
        ]


def test_awkward_string():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward1.to_list(tree["string"].array(library="ak")) == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_awkward_tstring():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward1.to_list(tree["tstring"].array(library="ak")) == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_awkward_vector_int32():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward1.to_list(tree["vector_int32"].array(library="ak")) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


def test_awkward_vector_string():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward1.to_list(tree["vector_string"].array(library="ak")) == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


def test_awkward_vector_tstring():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward1.to_list(tree["vector_tstring"].array(library="ak")) == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


def test_awkward_vector_vector_int32():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward1.to_list(tree["vector_vector_int32"].array(library="ak")) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_awkward_vector_vector_string():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward1.to_list(tree["vector_vector_string"].array(library="ak")) == [
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


def test_awkward_map_int32_int16():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward1.unzip(tree["map_int32_int16"].array(library="ak"))
        assert awkward1.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward1.to_list(values) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


def test_awkward_map_int32_vector_int16():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward1.unzip(
            tree["map_int32_vector_int16"].array(library="ak")
        )
        assert awkward1.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward1.to_list(values) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_awkward_map_int32_vector_string():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward1.unzip(
            tree["map_int32_vector_string"].array(library="ak")
        )
        assert awkward1.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward1.to_list(values) == [
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


def test_awkward_map_string_int16():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward1.unzip(tree["map_string_int16"].array(library="ak"))
        assert awkward1.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward1.to_list(values) == [
            [1],
            [1, 2],
            [1, 3, 2],
            [4, 1, 3, 2],
            [5, 4, 1, 3, 2],
        ]


def test_awkward_map_string_vector_string():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward1.unzip(
            tree["map_string_vector_string"].array(library="ak")
        )
        assert awkward1.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward1.to_list(values) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two", "three"], ["one", "two"]],
            [
                ["one", "two", "three", "four"],
                ["one"],
                ["one", "two", "three"],
                ["one", "two"],
            ],
            [
                ["one", "two", "three", "four", "five"],
                ["one", "two", "three", "four"],
                ["one"],
                ["one", "two", "three"],
                ["one", "two"],
            ],
        ]


def test_awkward_map_int32_vector_vector_int16():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward1.unzip(
            tree["map_int32_vector_vector_int16"].array(library="ak")
        )
        assert awkward1.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward1.to_list(values) == [
            [[[1]]],
            [[[1]], [[1], [1, 2]]],
            [[[1]], [[1], [1, 2]], [[1], [1, 2], [1, 2, 3]]],
            [
                [[1]],
                [[1], [1, 2]],
                [[1], [1, 2], [1, 2, 3]],
                [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            ],
            [
                [[1]],
                [[1], [1, 2]],
                [[1], [1, 2], [1, 2, 3]],
                [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
            ],
        ]


def test_awkward_map_string_string():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward1.unzip(tree["map_string_string"].array(library="ak"))
        assert awkward1.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward1.to_list(values) == [
            ["ONE"],
            ["ONE", "TWO"],
            ["ONE", "THREE", "TWO"],
            ["FOUR", "ONE", "THREE", "TWO"],
            ["FIVE", "FOUR", "ONE", "THREE", "TWO"],
        ]


def test_awkward_map_string_tstring():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward1.unzip(tree["map_string_tstring"].array(library="ak"))
        assert awkward1.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward1.to_list(values) == [
            ["ONE"],
            ["ONE", "TWO"],
            ["ONE", "THREE", "TWO"],
            ["FOUR", "ONE", "THREE", "TWO"],
            ["FIVE", "FOUR", "ONE", "THREE", "TWO"],
        ]
