# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
import os
import struct

import numpy
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")

# ranges shared by the lookup tests: two overlapping, one empty, one disjoint
RANGES = [(0, 10), (5, 0), (5, 7), (12, 8)]
TAGS = [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}, {"tag": "d"}]


def _attribute_set_record(key, name, schema_version=(1, 0)):
    """
    Builds the footer record that ROOT writes for a linked attribute set.

    ROOT points the locator at the payload of the anchor key of the attribute
    ``RNTuple``, which is what a ``ReadOnlyKey`` exposes through its data cursor and
    its compressed and uncompressed sizes.
    """
    record = uproot.models.RNTuple.MetaData("LinkedAttributeSetRecord")
    record.schema_version_major, record.schema_version_minor = schema_version
    record.anchor_uncompressed_size = key.data_uncompressed_bytes
    record.locator = uproot.models.RNTuple.MetaData("Locator")
    record.locator.offset = key.data_cursor.index
    record.locator.num_bytes = key.data_compressed_bytes
    record.attribute_set_name = name
    return record


def _attribute_data(ranges, user_data):
    # an attribute set with no records still needs a schema, so build one row and drop it
    starts = [start for start, _ in ranges] or [0]
    lengths = [length for _, length in ranges] or [0]
    data = ak.Array(
        {
            "_rangeStart": numpy.array(starts, dtype=numpy.uint64),
            "_rangeLen": numpy.array(lengths, dtype=numpy.uint64),
            "_userData": ak.Array(user_data[: len(starts)]),
        }
    )
    return data[: len(ranges)]


def _ntuple_with_attributes(path, sets, schema_versions=None):
    """
    Writes an ``RNTuple`` named ``Events`` together with one ``RNTuple`` per named
    attribute set in ``sets``, and links them the way ROOT does.

    No test file with attributes is available yet and ROOT only started writing them
    in 6.40, so the links are made here instead of coming from the file itself.
    """
    events = ak.Array({"x": numpy.arange(20, dtype=numpy.int64)})
    schema_versions = {} if schema_versions is None else schema_versions

    with uproot.recreate(path) as file:
        file.mkrntuple("Events", events.layout.form).extend(events)
        for name, (ranges, user_data) in sets.items():
            data = _attribute_data(ranges, user_data)
            writable = file.mkrntuple(name, data.layout.form)
            if len(data) > 0:
                writable.extend(data)

    file = uproot.open(path)
    ntuple = file["Events"]
    ntuple.footer.linked_attribute_sets = [
        _attribute_set_record(file.key(name), name, schema_versions.get(name, (1, 0)))
        for name in sets
    ]
    return ntuple


def _one_set(path, ranges=RANGES, user_data=TAGS, schema_version=(1, 0)):
    ntuple = _ntuple_with_attributes(
        path,
        {"Calibration": (ranges, user_data)},
        {"Calibration": schema_version},
    )
    return ntuple, ntuple.attributes["Calibration"]


def test_no_attributes(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as file:
        file.mkrntuple(
            "ntuple", ak.forms.RecordForm([ak.forms.NumpyForm("int32")], ["one"])
        )

    assert uproot.open(filepath)["ntuple"].attributes == {}

    # v1.0.0.0 predates attributes entirely, v1.0.1.0 has an empty list of them
    for filename in (
        "ntpl001_staff_rntuple_v1-0-0-0.root",
        "ntpl001_staff_rntuple_v1-0-1-0.root",
    ):
        with uproot.open(skhep_testdata.data_path(filename)) as file:
            assert file["Staff"].attributes == {}


def test_attribute_set_record():
    payload = b"".join(
        [
            struct.pack("<HHI", 1, 0, 78),  # schema version and anchor size
            struct.pack("<iQ", 42, 1024),  # locator
            struct.pack("<I", 11),  # name
            b"Calibration",
        ]
    )
    chunk = uproot.source.chunk.Chunk.wrap(None, payload)

    record = uproot.models.RNTuple.LinkedAttributeSetRecordReader().read(
        chunk, uproot.source.cursor.Cursor(0), {}
    )

    assert record.schema_version_major == 1
    assert record.schema_version_minor == 0
    assert record.anchor_uncompressed_size == 78
    assert record.locator.num_bytes == 42
    assert record.locator.offset == 1024
    assert record.attribute_set_name == "Calibration"


def test_attribute_ntuple(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"), [(0, 20)], TAGS[:1])

    assert attributes.name == "Calibration"
    assert attributes.schema_version == (1, 0)

    # the escape hatch exposes the internal layout of the attribute set
    linked = attributes.ntuple
    assert linked.name == "Calibration"
    assert linked.keys(recursive=False) == ["_rangeStart", "_rangeLen", "_userData"]


def test_attribute_ntuple_compressed_anchor():
    # the anchor written by ROOT is compressed, unlike the one Uproot writes
    with uproot.open(
        skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-1-0.root")
    ) as file:
        staff = file["Staff"]
        key = file.key("Staff")
        assert key.data_compressed_bytes < key.data_uncompressed_bytes

        linked = staff.read_attribute_ntuple(_attribute_set_record(key, "Whatever"))
        assert linked.name == "Staff"
        assert linked.num_entries == staff.num_entries


def test_multiple_attribute_sets(tmp_path):
    ntuple = _ntuple_with_attributes(
        os.path.join(tmp_path, "test.root"),
        {
            "Calibration": ([(0, 20)], [{"tag": "v1"}]),
            "Alignment": ([(0, 10), (10, 10)], [{"pass": 1}, {"pass": 2}]),
        },
    )

    # the dict follows the order of the records in the footer
    assert list(ntuple.attributes) == ["Calibration", "Alignment"]
    assert ntuple.attributes["Calibration"].keys() == ["tag"]
    assert ntuple.attributes["Alignment"].keys() == ["pass"]
    assert ntuple.attributes["Alignment"].arrays()["pass"].tolist() == [1, 2]
    assert ntuple.attributes["Alignment"].ranges.tolist() == [(0, 10), (10, 10)]


def test_user_fields(tmp_path):
    _, attributes = _one_set(
        os.path.join(tmp_path, "test.root"),
        [(0, 10), (10, 10)],
        [{"tag": "v1", "index": 1}, {"tag": "v2", "index": 2}],
    )

    assert attributes.keys() == ["tag", "index"]
    assert attributes.typenames() == {"tag": "std::string", "index": "std::int64_t"}
    assert attributes.arrays().tolist() == [
        {"tag": "v1", "index": 1},
        {"tag": "v2", "index": 2},
    ]

    # the meta fields of the attribute schema stay out of the attribute interface
    assert "_rangeStart" not in attributes
    assert len(attributes) == 2
    assert [field.name for field in attributes] == ["tag", "index"]
    assert attributes["tag"].array().tolist() == ["v1", "v2"]


def test_filters(tmp_path):
    _, attributes = _one_set(
        os.path.join(tmp_path, "test.root"),
        [(0, 20)],
        [{"tag": "v1", "index": 1}],
    )

    assert attributes.keys(filter_name="tag") == ["tag"]
    assert attributes.typenames(filter_name="index") == {"index": "std::int64_t"}
    assert attributes.arrays(filter_name="tag").tolist() == [{"tag": "v1"}]
    assert attributes.arrays(filter_typename="std::int64_t").tolist() == [{"index": 1}]
    assert attributes.arrays(library="np").tolist() == [("v1", 1)]


def test_ranges(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"))

    assert attributes.ranges.dtype.names == ("start", "length")
    assert attributes.ranges.tolist() == RANGES


def test_no_records(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"), [], TAGS[:1])

    assert attributes.keys() == ["tag"]
    assert attributes.ranges.tolist() == []
    assert attributes.arrays().tolist() == []
    assert attributes.for_entry(0).tolist() == []


def test_for_entry(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"))

    assert attributes.for_entry(0).tolist() == [0]
    assert attributes.for_entry(9).tolist() == [0, 2]
    assert attributes.for_entry(11).tolist() == [2]
    assert attributes.for_entry(19).tolist() == [3]
    # the empty range starting at 5 never matches, and neither does an unused entry
    assert attributes.for_entry(5).tolist() == [0, 2]
    assert attributes.for_entry(100).tolist() == []

    assert attributes.arrays()[attributes.for_entry(11)].tag.tolist() == ["c"]


def test_for_entries(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"))

    assert attributes.for_entries(0, 20).tolist() == [0, 2, 3]
    assert attributes.for_entries(0, 1).tolist() == [0]
    assert attributes.for_entries(10, 12).tolist() == [2]
    assert attributes.for_entries(20, 30).tolist() == []
    # an empty range of entries matches nothing, not even the ranges around it
    assert attributes.for_entries(5, 5).tolist() == []


def test_cache_key(tmp_path):
    ntuple, attributes = _one_set(
        os.path.join(tmp_path, "test.root"), [(0, 20)], TAGS[:1]
    )
    assert attributes.ntuple.cache_key != ntuple.cache_key

    # the two RNTuples share the file's array cache, so colliding cache keys would
    # make them read each other's pages
    assert ntuple.arrays().x.tolist() == list(range(20))
    assert attributes.ranges.tolist() == [(0, 20)]
    assert attributes.arrays().tag.tolist() == ["a"]
    assert ntuple.arrays().x.tolist() == list(range(20))


def test_unsupported_schema_version(tmp_path):
    ntuple = _ntuple_with_attributes(
        os.path.join(tmp_path, "test.root"),
        {"FromTheFuture": ([(0, 20)], TAGS[:1]), "Calibration": ([(0, 20)], TAGS[:1])},
        {"FromTheFuture": (2, 0)},
    )

    # listing the attribute sets works; only using the unsupported one raises, and
    # the other attribute sets of the same RNTuple stay readable
    assert list(ntuple.attributes) == ["FromTheFuture", "Calibration"]
    assert ntuple.attributes["FromTheFuture"].schema_version == (2, 0)
    with pytest.raises(NotImplementedError, match="schema version"):
        ntuple.attributes["FromTheFuture"].keys()
    assert ntuple.attributes["Calibration"].keys() == ["tag"]


def test_unknown_minor_schema_version(tmp_path):
    _, attributes = _one_set(
        os.path.join(tmp_path, "test.root"), [(0, 20)], TAGS[:1], schema_version=(1, 7)
    )

    assert attributes.schema_version == (1, 7)
    assert attributes.keys() == ["tag"]


def test_invalid_attribute_set(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")
    data = ak.Array({"x": numpy.arange(5, dtype=numpy.int64)})
    with uproot.recreate(filepath) as file:
        file.mkrntuple("Events", data.layout.form).extend(data)

    file = uproot.open(filepath)
    ntuple = file["Events"]
    ntuple.footer.linked_attribute_sets = [
        _attribute_set_record(file.key("Events"), "NotAnAttributeSet")
    ]

    with pytest.raises(ValueError, match="_rangeStart"):
        ntuple.attributes["NotAnAttributeSet"].keys()


def test_attributes_are_lazy(tmp_path):
    ntuple = _ntuple_with_attributes(
        os.path.join(tmp_path, "test.root"), {"Calibration": ([(0, 20)], TAGS[:1])}
    )
    attributes = ntuple.attributes["Calibration"]

    # name and schema version come from the footer record, with nothing else read
    assert attributes.name == "Calibration"
    assert attributes.schema_version == (1, 0)
    assert attributes._ntuple is None

    assert attributes.keys() == ["tag"]
    assert attributes._ntuple is not None


def test_arrays_are_unaffected(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")
    ntuple, _ = _one_set(filepath, [(0, 20)], TAGS[:1])

    # attribute sets are not fields, so they stay out of the field interface
    assert ntuple.keys() == ["x"]
    assert ntuple.arrays().fields == ["x"]
    assert ntuple.arrays().x.tolist() == list(range(20))


def test_ROOT(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    if ROOT.gROOT.GetVersionInt() < 64000:
        pytest.skip("ROOT version does not support RNTuple attributes")

    filepath = os.path.join(tmp_path, "test.root")

    assert ROOT.gInterpreter.Declare("""
        #include <ROOT/RNTupleAttrWriting.hxx>
        #include <ROOT/RNTupleModel.hxx>
        #include <ROOT/RNTupleWriter.hxx>
        #include <TFile.h>
        #include <cstdint>
        #include <memory>
        #include <string>
        #include <utility>

        void uproot_write_rntuple_attributes(const char *path)
        {
            auto file = std::unique_ptr<TFile>(TFile::Open(path, "RECREATE"));

            auto model = ROOT::RNTupleModel::Create();
            auto pX = model->MakeField<std::int64_t>("x");
            auto writer = ROOT::RNTupleWriter::Append(std::move(model), "Events", *file);

            auto attrModel = ROOT::RNTupleModel::Create();
            auto pTag = attrModel->MakeField<std::string>("tag");
            auto attrs = writer->CreateAttributeSet(std::move(attrModel), "Calibration");

            auto first = attrs->BeginRange();
            for (std::int64_t i = 0; i < 5; ++i) { *pX = i; writer->Fill(); }
            *pTag = "first";
            attrs->CommitRange(std::move(first));

            auto second = attrs->BeginRange();
            for (std::int64_t i = 5; i < 8; ++i) { *pX = i; writer->Fill(); }
            *pTag = "second";
            attrs->CommitRange(std::move(second));

            writer.reset();
            file->Close();
        }
        """)
    ROOT.uproot_write_rntuple_attributes(filepath)

    ntuple = uproot.open(filepath)["Events"]
    attributes = ntuple.attributes["Calibration"]

    assert attributes.schema_version == (1, 0)
    assert attributes.keys() == ["tag"]
    assert attributes.ranges.tolist() == [(0, 5), (5, 3)]
    assert attributes.arrays().tag.tolist() == ["first", "second"]
    assert attributes.for_entry(2).tolist() == [0]
    assert attributes.for_entry(6).tolist() == [1]
    assert attributes.for_entries(4, 6).tolist() == [0, 1]
    assert ntuple.arrays().x.tolist() == list(range(8))
