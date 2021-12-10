# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines all of the classes and functions that are needed for writing ROOT
files. Uproot has a strong asymmetry between reading and writing, with writing defined
as a distinct task.

The :doc:`uproot.writing.writable` submodule defines the entry-points for user
interaction: :doc:`uproot.writing.writable.create`, :doc:`uproot.writing.writable.recreate`,
and :doc:`uproot.writing.writable.update`.
"""


from uproot.writing.identify import (
    to_TArray,
    to_TH1x,
    to_TH2x,
    to_TH3x,
    to_TList,
    to_TObjString,
    to_TProfile,
    to_TProfile2D,
    to_TProfile3D,
    to_writable,
)
from uproot.writing.writable import (
    WritableBranch,
    WritableDirectory,
    WritableFile,
    WritableTree,
    create,
    recreate,
    update,
)

__all__ = [
    "to_TArray",
    "to_TH1x",
    "to_TH2x",
    "to_TH3x",
    "to_TList",
    "to_TObjString",
    "to_TProfile",
    "to_TProfile2D",
    "to_TProfile3D",
    "to_writable",
    "WritableDirectory",
    "WritableFile",
    "WritableTree",
    "WritableBranch",
    "create",
    "recreate",
    "update",
]
