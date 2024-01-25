import pytest
import skhep_testdata

import uproot


dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")


@pytest.mark.parametrize(
    "allowed", [True, (OSError,), (FileNotFoundError, RuntimeError)]
)
def test_with_report(allowed):
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    test_path3 = "/some/file/that/doesnt/exist"
    files = [test_path1, test_path2, test_path3]
    collection, report = uproot.dask(
        files,
        library="ak",
        open_files=False,
        allow_read_errors_with_report=allowed,
    )
    _, creport = dask.compute(collection, report)
    assert creport[0].exception is None  # test_path1 is good
    assert creport[1].exception is None  # test_path2 is good
    assert creport[2].exception == "FileNotFoundError"  # test_path3 is a bad file


def test_with_report_exception_missed():
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    test_path3 = "/some/file/that/doesnt/exist"
    files = [test_path1, test_path2, test_path3]
    collection, report = uproot.dask(
        files,
        library="ak",
        open_files=False,
        allow_read_errors_with_report=(
            ValueError,  # allow ValueErrors, but that's not what happens!
        ),
    )
    with pytest.raises(FileNotFoundError):
        _, creport = dask.compute(collection, report)
