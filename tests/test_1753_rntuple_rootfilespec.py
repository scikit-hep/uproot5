# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import io
import os
import pickle

import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("awkward")

RNTUPLE_FILES = sorted(
    name
    for name in skhep_testdata.known_files
    if "rntuple" in name and name.endswith(".root")
)


def _rntuples(filename):
    with uproot.open(skhep_testdata.data_path(filename)) as f:
        for key, classname in f.classnames().items():
            if classname == "ROOT::RNTuple":
                yield f[key]


@pytest.mark.parametrize("filename", RNTUPLE_FILES)
def test_metadata_views(filename):
    for ntuple in _rntuples(filename):
        assert ntuple.num_entries == sum(c.num_entries for c in ntuple.clusters)
        entry = 0
        for cluster_id, cluster in enumerate(ntuple.clusters):
            assert cluster.cluster_id == cluster_id
            assert cluster.num_first_entry == entry
            entry += cluster.num_entries
            assert len(cluster.columns) == len(ntuple.column_records)
            for column_id, column in enumerate(cluster.columns):
                assert column.column_id == column_id
                if column.suppressed:
                    assert column.pages == ()
                else:
                    stored = sum(page.num_elements for page in column.pages)
                    assert stored == column.num_elements - column.num_zero_elements
        for field_id, record in enumerate(ntuple.field_records):
            assert record.field_id == field_id
            assert isinstance(record.field_name, str)
        assert isinstance(ntuple.name, str)
        assert len(ntuple.keys()) > 0


def test_pickle_after_reading_metadata():
    filename = skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        ntuple = f["Staff"]
        ntuple.keys()
        expected = ntuple.arrays()
        unpickled = pickle.loads(pickle.dumps(ntuple))
        assert unpickled.arrays().tolist() == expected.tolist()


def _corrupted(data, position):
    data = bytearray(data)
    data[position] ^= 0xFF
    return io.BytesIO(bytes(data))


def test_corrupt_anchor():
    filename = skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-0-0.root")
    with open(filename, "rb") as file:
        data = file.read()
    with uproot.open(filename) as f:
        key = f.key("Staff")
        assert not key.is_compressed
        position = key.data_cursor.index + 10

    # A FormatError is not a DeserializationError, so Uproot does not retry with
    # a class generated from the file's streamers
    assert issubclass(uproot.exceptions.FormatError, ValueError)
    assert not issubclass(
        uproot.exceptions.FormatError, uproot.deserialization.DeserializationError
    )
    with uproot.open(_corrupted(data, position)) as f:
        with pytest.raises(uproot.exceptions.FormatError, match="checksum"):
            f["Staff"]


def test_corrupt_page_list(tmp_path):
    filename = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filename) as f:
        f["ntuple"] = {"x": np.arange(5, dtype=np.int32)}
    with uproot.open(filename) as f:
        group = f["ntuple"]._metadata_schema.cluster_groups[0]
        # Uproot writes page lists uncompressed
        assert group.page_list_num_bytes == group.page_list_length
        position = group.page_list_offset + 10
    with open(filename, "rb") as file:
        data = file.read()

    with uproot.open(_corrupted(data, position)) as f:
        ntuple = f["ntuple"]
        assert ntuple.keys() == ["x"]
        with pytest.raises(uproot.exceptions.FormatError, match="checksum") as err:
            ntuple.num_entries
        assert err.value.object_path is not None


def test_deprecated_metadata_attributes():
    filename = skhep_testdata.data_path(
        "test_multiple_cluster_groups_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        ntuple = f["ntuple"]
        with pytest.warns(DeprecationWarning):
            header = ntuple.header
        assert header.ntuple_name == ntuple.name
        assert [r.field_name for r in header.field_records] == [
            r.field_name for r in ntuple.field_records
        ]
        with pytest.warns(DeprecationWarning):
            footer = ntuple.footer
        assert footer.header_checksum == header.checksum
        assert [g.num_clusters for g in footer.cluster_group_records] == [5, 4, 3]
        with pytest.warns(DeprecationWarning):
            page_list_envelopes = ntuple.page_list_envelopes
        assert len(page_list_envelopes) == 3
        with pytest.warns(DeprecationWarning):
            page_link_list = ntuple.page_link_list
        assert len(page_link_list) == len(ntuple.clusters)
        page = page_link_list[0][0].pages[0]
        assert page.locator.offset == ntuple.clusters[0].columns[0].pages[0].offset
        with pytest.warns(DeprecationWarning):
            chunk, _ = ntuple.read_locator(page.locator, page.locator.num_bytes)
        assert len(chunk.raw_data) == page.locator.num_bytes


def test_element_offset_of_field_added_by_update(tmp_path):
    filename = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filename) as f:
        f["ntuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}
    with uproot.update(filename) as f:
        f["ntuple"].add_fields({"z": np.int32})
    with uproot.update(filename) as f:
        f["ntuple"].extend(
            {
                "x": np.array([4, 5, 6], dtype=np.float32),
                "z": np.array([40, 50, 60], dtype=np.int32),
            }
        )

    with uproot.open(filename) as f:
        ntuple = f["ntuple"]
        z_column = ntuple.column_records[1]
        assert z_column.first_element_index == 3
        first, second = ntuple.clusters
        # the cluster written before add_fields does not list the column
        assert first.columns[1].element_offset is None
        assert first.columns[1].num_zero_elements == 3
        # the deferred column's elements start at its first element index
        assert second.columns[1].element_offset == 3
        assert ntuple["z"].array().tolist() == [0, 0, 0, 40, 50, 60]
