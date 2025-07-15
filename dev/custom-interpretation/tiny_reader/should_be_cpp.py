from __future__ import annotations

from array import array

import numpy as np
from AsCustom.should_be_cpp import Cpp_BaseReader, Cpp_ObjectHeaderReader


class Cpp_MyTObjArrayReader(Cpp_BaseReader):
    """
    This class reads a TObjArray from a binary parser.

    I know that there is only 1 kind of class in the TObjArray I will read,
    so I can use only 1 reader to read all elements in TObjArray.
    """

    def __init__(self, name: str, element_reader: Cpp_ObjectHeaderReader):
        """
        Args:
            element_reader (BaseReader): The reader for the elements in the array.
        """
        self.name = name
        self.element_reader = element_reader
        self.counts = array("Q")

    def read(self, parser):
        _ = parser.read_fNBytes()
        _ = parser.read_fVersion()
        _ = parser.read_fVersion()
        _ = parser.read_number("u4")  # fUniqueID
        _ = parser.read_number("u4")  # fBits

        # Just directly read data
        _ = parser.read_number("u1")  # fName
        fSize = parser.read_number("u4")
        _ = parser.read_number("u4")  # fLowerBound

        for _ in range(fSize):
            self.element_reader.read(parser)

        # Update offsets
        self.counts.append(fSize)

    def get_data(self):
        return (
            np.asarray(self.counts, dtype="Q"),
            self.element_reader.get_data(),
        )
