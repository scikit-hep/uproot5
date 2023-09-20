# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata
import awkward as ak

import uproot

dask_awkward = pytest.importorskip("dask_awkward")


@pytest.mark.parametrize("library", ["np", "ak"])
@pytest.mark.parametrize("step_size", ["100MB", uproot._util.unset])
@pytest.mark.parametrize("steps_per_file", [1, 2, 5, 10, 15, uproot._util.unset])
@pytest.mark.parametrize("open_files", [False, True])
def test_uproot_dask_steps(library, step_size, steps_per_file, open_files):
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ttree = uproot.open(test_path)

    arrays = ttree.arrays(library=library)
    if not isinstance(step_size, uproot._util._Unset) and not open_files:
        with pytest.raises(TypeError):
            uproot.dask(
                test_path,
                library=library,
                step_size=step_size,
                steps_per_file=steps_per_file,
                open_files=open_files,
            )
    elif not isinstance(step_size, uproot._util._Unset) and not isinstance(
        steps_per_file, uproot._util._Unset
    ):
        with pytest.raises(TypeError):
            uproot.dask(
                test_path,
                library=library,
                step_size=step_size,
                steps_per_file=steps_per_file,
                open_files=open_files,
            )
    else:
        dask_arrays = uproot.dask(
            test_path,
            library=library,
            step_size=step_size,
            steps_per_file=steps_per_file,
            open_files=open_files,
        )

        if library == "np":
            assert list(dask_arrays.keys()) == list(
                arrays.keys()
            ), "Different keys detected in dictionary of dask arrays and dictionary of numpy arrays"

            comp = []
            for key in arrays.keys():
                comp.append(numpy.array_equal(dask_arrays[key].compute(), arrays[key]))
            assert all(comp), f"Incorrect array at key {key}"

        else:
            assert ak.almost_equal(
                dask_arrays[["px1", "px2", "py1", "py2"]].compute(
                    scheduler="synchronous"
                ),
                arrays[["px1", "px2", "py1", "py2"]],
            )
