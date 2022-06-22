# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest

import uproot


@pytest.fixture(scope="function", autouse=False)
def reset_classes():
    uproot.model.reset_classes()
    return
