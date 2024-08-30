# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines known forth code for some ElementLink data types in ATLAS (D)AODs
"""

# TODO: delay import?
from __future__ import annotations

from awkward.forms import ListOffsetForm, NumpyForm, RecordForm

vector_vector_element_link = (
    """
input stream
input byteoffsets
input bytestops
output node1-offsets int64
output node2-offsets int64
output node3-data uint32
output node4-data uint32

0 node1-offsets <- stack
0 node2-offsets <- stack

0 do
    byteoffsets I-> stack
    stream seek
    6 stream skip
    stream !I-> stack
    dup node1-offsets +<- stack
    0 do
        stream !I-> stack
        dup node2-offsets +<- stack
        0 do
            20 stream skip
            stream !I-> node3-data
            stream !I-> node4-data
        loop
    loop
loop
""",
    ListOffsetForm(
        "i64",
        ListOffsetForm(
            "i64",
            RecordForm(
                [
                    NumpyForm("uint32", form_key="node3"),
                    NumpyForm("uint32", form_key="node4"),
                ],
                ["m_persKey", "m_persIndex"],
            ),
            form_key="node2",
        ),
        form_key="node1",
    ),
)
