from __future__ import annotations

from array import array
from typing import Literal

import numpy as np

ctype_hints = Literal["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f", "d"]
type_np2array = {
    "u1": "B",
    "u2": "H",
    "u4": "I",
    "u8": "Q",
    "i1": "b",
    "i2": "h",
    "i4": "i",
    "i8": "q",
    "f": "f",
    "d": "d",
}


class Cpp_BinaryParser:
    nbytes_dict = {
        "u1": 1,
        "u2": 2,
        "u4": 4,
        "u8": 8,
        "i1": 1,
        "i2": 2,
        "i4": 4,
        "i8": 8,
        "f": 4,
        "d": 8,
    }

    def __init__(
        self,
        data: np.ndarray[np.uint8],
        offsets: np.ndarray,
    ):
        """
        Args:
            data (np.ndarray): The binary data to parse.
        """
        self.data = data
        self.offsets = offsets
        self.cursor = 0

    @property
    def n_entries(self) -> int:
        """
        Returns:
            The number of entries in the binary data.
        """
        return len(self.offsets) - 1

    def read_number(
        self,
        ctype: Literal["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f", "d"],
    ) -> np.number:
        nbytes = self.nbytes_dict[ctype]
        value = self.data[self.cursor : self.cursor + nbytes].view(f">{ctype}")[0]
        self.cursor += nbytes
        return value

    def read_fNBytes(self) -> np.uint32:
        nbytes = self.read_number("u4")
        assert nbytes & 0x40000000 != 0, f"Invalid fNBytes: {nbytes:#x}"
        return nbytes & ~np.uint32(0x40000000)

    def read_fVersion(self) -> np.uint16:
        return self.read_number("u2")

    def read_null_terminated_string(self) -> str:
        """
        Reads a null-terminated string from the binary data.

        Returns:
            The null-terminated string.
        """
        start = self.cursor
        while self.data[self.cursor] != 0:
            self.cursor += 1
        end = self.cursor
        self.cursor += 1

        return self.data[start:end].tobytes().decode("utf-8")

    def __repr__(self):
        cur_data_str = str(self.data[self.cursor :])
        return f"BinaryParser({cur_data_str})"


class Cpp_BaseReader:
    def read(self, parser: Cpp_BinaryParser) -> None:
        raise AssertionError

    def get_data(self) -> np.ndarray | tuple:
        """
        Returns:
            The data read by the reader.
        """
        raise AssertionError


class Cpp_CtypeReader(Cpp_BaseReader):
    """
    This class reads C++ primitive types from a binary parser.
    """

    def __init__(
        self,
        name: str,
        ctype: ctype_hints,
    ):
        self.name = name
        self.ctype = ctype
        self.data = array(type_np2array[ctype])

    def read(self, parser: Cpp_BinaryParser):
        self.data.append(parser.read_number(self.ctype))

    def get_data(self):
        return np.array(self.data, dtype=self.ctype, copy=True)


class Cpp_STLSequenceReader(Cpp_BaseReader):
    """
    This class reads STL sequence (vector, array) from a binary parser.
    """

    def __init__(self, name: str, is_top: bool, element_reader: Cpp_BaseReader):
        self.name = name
        self.element_reader = element_reader
        self.counts = array("Q")
        self.is_top = is_top

    def read(self, parser):
        # Read fNBytes and fVersion when it is top level
        if self.is_top:
            _ = parser.read_fNBytes()
            _ = parser.read_fVersion()

        # Read data
        fSize = parser.read_number("u4")
        for _ in range(fSize):
            self.element_reader.read(parser)

        # Update counts
        self.counts.append(fSize)

    def get_data(self):
        return (
            np.asarray(self.counts, dtype="Q"),
            self.element_reader.get_data(),
        )


class Cpp_STLMapReader(Cpp_BaseReader):
    """
    This class reads std::map, unordered_map, multimap from a binary parser.
    """

    def __init__(
        self,
        name: str,
        is_top: bool,
        key_reader: Cpp_BaseReader,
        val_reader: Cpp_BaseReader,
    ):
        self.name = name
        self.key_reader = key_reader
        self.val_reader = val_reader
        self.counts = array("Q")
        self.is_top = is_top

    def read(self, parser):
        # Read fNBytes and fVersion when it is top level
        if self.is_top:
            _ = parser.read_fNBytes()
            _ = parser.read_number("u8")  # I don't know what this is

        # Read data
        fSize = parser.read_number("u4")

        if self.is_top:
            for _ in range(fSize):
                self.key_reader.read(parser)
            for _ in range(fSize):
                self.val_reader.read(parser)
        else:
            for _ in range(fSize):
                self.key_reader.read(parser)
                self.val_reader.read(parser)

        # Update counts
        self.counts.append(fSize)

    def get_data(self):
        return (
            np.asarray(self.counts, dtype="Q"),
            self.key_reader.get_data(),
            self.val_reader.get_data(),
        )


class Cpp_STLStringReader(Cpp_BaseReader):
    """
    This class reads std::string from a binary parser.
    """

    def __init__(self, name: str, is_top: bool):
        self.name = name
        self.data = array("B")
        self.counts = array("Q")
        self.is_top = is_top

    def read(self, parser):
        # Read fNBytes and fVersion when it is top level
        if self.is_top:
            _ = parser.read_fNBytes()
            _ = parser.read_fVersion()

        # Get length of std::string
        fSize = parser.read_number("u1")
        if fSize == 255:
            fSize = parser.read_number("u4")

        # Read data
        for _ in range(fSize):
            self.data.append(parser.read_number("u1"))

        # Update counts
        self.counts.append(fSize)

    def get_data(self):
        return (
            np.asarray(self.counts, dtype="Q"),
            np.asarray(self.data, dtype="B"),
        )


class Cpp_TArrayReader(Cpp_BaseReader):
    """
    This class reads TArray from a binary paerser.

    TArray includes TArrayC, TArrayS, TArrayI, TArrayL, TArrayF, and TArrayD.
    Corresponding ctype is u1, u2, i4, i8, f, and d.
    """

    def __init__(
        self,
        name: str,
        ctype: Literal["i1", "i2", "i4", "i8", "f", "d"],
    ):
        self.name = name
        self.ctype = ctype
        self.data = array(type_np2array[ctype])
        self.counts = array("Q")

    def read(self, parser):
        # Read data
        fSize = parser.read_number("u4")
        for _ in range(fSize):
            self.data.append(parser.read_number(self.ctype))

        # Update counts
        self.counts.append(fSize)

    def get_data(self):
        return (
            np.asarray(self.counts, dtype="Q"),
            np.asarray(self.data, dtype=self.ctype),
        )


class Cpp_TStringReader(Cpp_BaseReader):
    """
    This class reads TString from a binary parser.
    """

    def __init__(self, name: str):
        self.name = name
        self.data = array("B")
        self.counts = array("Q")

    def read(self, parser):
        # Get length of TString
        fSize = parser.read_number("u1")
        if fSize == 255:
            fSize = parser.read_number("u4")

        # Read data
        for _ in range(fSize):
            self.data.append(parser.read_number("u1"))

        # Update counts
        self.counts.append(fSize)

    def get_data(self):
        return (
            np.asarray(self.counts, dtype="Q"),
            np.asarray(self.data, dtype="B"),
        )


class Cpp_TObjectReader(Cpp_BaseReader):
    """
    This class reads TObject from a binary parser.

    It will not record any data.
    """

    def __init__(self, name: str):
        self.name = name

    def read(self, parser):
        _ = parser.read_fVersion()  # fVersion
        _ = parser.read_number("u4")  # fUniqueID
        _ = parser.read_number("u4")  # fBits

    def get_data(self):
        return None  # should I return anything?


class Cpp_CArrayReader(Cpp_BaseReader):
    """
    This class reads a C-array from a binary parser.
    """

    def __init__(
        self,
        name: str,
        is_obj: bool,
        flat_size: int,
        element_reader: Cpp_BaseReader,
    ):
        self.name = name
        self.is_obj = is_obj
        self.flat_size = flat_size
        self.element_reader = element_reader

    def read(self, parser):
        if self.is_obj:
            # Read fNBytes and fVersion
            _ = parser.read_fNBytes()
            _ = parser.read_fVersion()

        # Read data
        for _ in range(self.flat_size):
            self.element_reader.read(parser)

    def get_data(self):
        return self.element_reader.get_data()


class Cpp_BaseObjectReader(Cpp_BaseReader):
    """
    Base class is what a custom class inherits from.
    It has fNBytes(uint32), fVersion(uint16) at the beginning.
    """

    def __init__(self, name: str, sub_readers: list[Cpp_BaseReader]):
        self.name = name
        self.sub_readers = sub_readers

    def read(self, parser):
        # Read fNBytes and fVersion
        _ = parser.read_fNBytes()
        _ = parser.read_fVersion()

        # Read data
        for sub_reader in self.sub_readers:
            sub_reader.read(parser)

    def get_data(self):
        data = []
        for sub_reader in self.sub_readers:
            data.append(sub_reader.get_data())
        return data


class Cpp_ObjectHeaderReader(Cpp_BaseReader):
    """
    This class read an object starting with an object header.
    """

    def __init__(self, name: str, sub_readers: list[Cpp_BaseReader]):
        """
        Args:
            sub_readers (list[BaseReader]): The readers for the elements in the object.
        """
        self.name = name
        self.sub_readers = sub_readers

    def read(self, parser):
        # read object header
        _ = parser.read_fNBytes()
        fTag = parser.read_number("i4")
        _ = parser.read_null_terminated_string() if fTag == -1 else ""  # fClassName

        _ = parser.read_fNBytes()
        _ = parser.read_fVersion()
        for sub_reader in self.sub_readers:
            sub_reader.read(parser)

    def get_data(self):
        data = []
        for sub_reader in self.sub_readers:
            data.append(sub_reader.get_data())
        return data


class Cpp_EmptyReader(Cpp_BaseReader):
    """
    This class does nothing.
    """

    def __init__(self, name: str):
        self.name = name

    def read(self, parser):
        pass

    def get_data(self):
        return None
