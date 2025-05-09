from __future__ import annotations

import re
from enum import Enum
from typing import Literal

import awkward as ak
import numpy as np

import uproot

from .should_be_cpp import (
    Cpp_BaseObjectReader,
    Cpp_BinaryParser,
    Cpp_CArrayReader,
    Cpp_CtypeReader,
    Cpp_EmptyReader,
    Cpp_ObjectHeaderReader,
    Cpp_STLMapReader,
    Cpp_STLSequenceReader,
    Cpp_STLStringReader,
    Cpp_TArrayReader,
    Cpp_TObjectReader,
    Cpp_TStringReader,
)

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

num_typenames = {
    "bool": "i1",
    "char": "i1",
    "short": "i2",
    "int": "i4",
    "long": "i8",
    "unsigned char": "u1",
    "unsigned short": "u2",
    "unsigned int": "u4",
    "unsigned long": "u8",
    "float": "f",
    "double": "d",
    # cstdint
    "int8_t": "i1",
    "int16_t": "i2",
    "int32_t": "i4",
    "int64_t": "i8",
    "uint8_t": "u1",
    "uint16_t": "u2",
    "uint32_t": "u4",
    "uint64_t": "u8",
    # ROOT types
    "Bool_t": "i1",
    "Char_t": "i1",
    "Short_t": "i2",
    "Int_t": "i4",
    "Long_t": "i8",
    "UChar_t": "u1",
    "UShort_t": "u2",
    "UInt_t": "u4",
    "ULong_t": "u8",
    "Float_t": "f",
    "Double_t": "d",
}

stl_typenames = {
    "vector",
    "array",
    "map",
    "unordered_map",
    "string",
}


tarray_typenames = {
    "TArrayC": "i1",
    "TArrayS": "i2",
    "TArrayI": "i4",
    "TArrayL": "i8",
    "TArrayF": "f",
    "TArrayD": "d",
}


ctype_hints = Literal["bool", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f", "d"]


class ReaderType(Enum):
    CType = "CType"
    STLSequence = "STLSequence"
    STLMap = "STLMap"
    STLString = "STLString"
    TArray = "TArray"
    TString = "TString"
    TObject = "TObject"
    CArray = "CArray"
    BaseObject = "BaseObject"
    ObjectHeader = "ObjectHeader"
    Empty = "Empty"


def get_top_type_name(type_name: str) -> str:
    if type_name.endswith("*"):
        type_name = type_name[:-1].strip()
    type_name = type_name.replace("std::", "").strip()
    return type_name.split("<")[0]


def gen_tree_config(
    cls_streamer_info: dict,
    all_streamer_info: dict,
    item_path: str = "",
) -> dict:
    """
    Generate reader configuration for a class streamer information.

    The content it returns should be:

    ```python
    {
        "reader": ReaderType,
        "name": str,
        "ctype": str, # for CTypeReader, TArrayReader
        "element_reader": dict, # reader config of the element, for STLVectorReader, SimpleCArrayReader, TObjectCArrayReader
        "flat_size": int, # for SimpleCArrayReader, TObjectCArrayReader
        "fMaxIndex": list[int], # for SimpleCArrayReader, TObjectCArrayReader
        "fArrayDim": int, # for SimpleCArrayReader, TObjectCArrayReader
        "key_reader": dict, # reader config of the key, for STLMapReader
        "val_reader": dict, # reader config of the value, for STLMapReader
        "sub_readers": list[dict], # for BaseObjectReader, ObjectHeaderReader
        "is_top_level": bool, # for STLVectorReader, STLMapReader, STLStringReader
    }
    ```

    Args:
        cls_streamer_info (dict): Class streamer information.
        all_streamer_info (dict): All streamer information.
        item_path (str): Path to the item.

    Returns:
        dict: Reader configuration.
    """
    fName = cls_streamer_info["fName"]
    item_path = fName if item_path == "" else f"{item_path}.{fName}"

    for reader in sorted(readers, key=lambda x: x.priority(), reverse=True):
        top_type_name = get_top_type_name(cls_streamer_info["fTypeName"])
        tree_config = reader.gen_tree_config(
            top_type_name,
            cls_streamer_info,
            all_streamer_info,
            item_path,
        )
        if tree_config is not None:
            return tree_config

    raise ValueError(f"Unknown type: {cls_streamer_info['fTypeName']} for {item_path}")


def get_reader_instance(tree_config: dict):
    for cls_reader in sorted(readers, key=lambda x: x.priority(), reverse=True):
        reader = cls_reader.get_reader_instance(tree_config)
        if reader is not None:
            return reader

    raise ValueError(
        f"Unknown reader type: {tree_config['reader']} for {tree_config['name']}"
    )


def reconstruct_array(
    raw_data: np.ndarray | tuple | list | None,
    tree_config: dict,
) -> ak.Array | None:
    for reader in sorted(readers, key=lambda x: x.priority(), reverse=True):
        data = reader.reconstruct_array(raw_data, tree_config)
        if data is not None:
            return data

    raise ValueError(
        f"Unknown reader type: {tree_config['reader']} for {tree_config['name']}"
    )


def gen_tree_config_from_type_name(
    type_name: str,
    all_streamer_info: dict,
    item_path: str = "",
):
    return gen_tree_config(
        {
            "fName": type_name,
            "fTypeName": type_name,
        },
        all_streamer_info,
        item_path,
    )


def regularize_object_path(object_path: str) -> str:
    return re.sub(r";[0-9]+", r"", object_path)


class BaseReader:
    """
    Base class for all readers.
    """

    @classmethod
    def priority(cls) -> int:
        """
        The priority of the reader. Higher priority means the reader will be
        used first.
        """
        return 20

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name: str,
        cls_streamer_info: dict,
        all_streamer_info: dict,
        item_path: str = "",
    ) -> dict:
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        """
        Args:
            tree_config (dict): The configuration dictionary for the reader.

        Returns:
            An instance of the appropriate reader class.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    def reconstruct_array(
        cls,
        raw_data: np.ndarray | tuple | list | None,
        tree_config: dict,
    ) -> ak.Array | None:
        """
        Args:
            raw_data (Union[np.ndarray, tuple, list, None]): The raw data to be
                recovered.
            tree_config (dict): The configuration dictionary for the reader.

        Returns:
            awkward.Array: The recovered data as an awkward array.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


readers: set[BaseReader] = set()


class CTypeReader(BaseReader):
    """
    This class reads C++ primitive types from a binary parser.
    """

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name,
        cls_streamer_info,
        all_streamer_info,
        item_path,
    ):
        if top_type_name in num_typenames:
            ctype = num_typenames[top_type_name]
            return {
                "reader": ReaderType.CType,
                "name": cls_streamer_info["fName"],
                "ctype": ctype,
            }
        else:
            return None

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        if tree_config["reader"] != ReaderType.CType:
            return None

        ctype = tree_config["ctype"]
        return Cpp_CtypeReader(tree_config["name"], ctype)

    @classmethod
    def reconstruct_array(cls, raw_data, tree_config):
        if tree_config["reader"] != ReaderType.CType:
            return None

        return ak.Array(raw_data)


class STLSequenceReader(BaseReader):
    """
    This class reads STL sequence (vector, array) from a binary parser.
    """

    @staticmethod
    def get_sequence_element_typename(type_name: str) -> str:
        """
        Get the element type name of a vector type.

        e.g. vector<vector<int>> -> vector<int>
        """
        type_name = (
            type_name.replace("std::", "").replace("< ", "<").replace(" >", ">").strip()
        )
        return re.match(r"^(vector|array)<(.*)>$", type_name).group(2)

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name,
        cls_streamer_info,
        all_streamer_info,
        item_path,
    ):
        if top_type_name not in ["vector", "array"]:
            return None

        fName = cls_streamer_info["fName"]
        fTypeName = cls_streamer_info["fTypeName"]
        element_type = cls.get_sequence_element_typename(fTypeName)
        element_info = {
            "fName": fName,
            "fTypeName": element_type,
        }

        element_tree_config = gen_tree_config(
            element_info,
            all_streamer_info,
            item_path,
        )

        top_element_type = get_top_type_name(element_type)
        if top_element_type in stl_typenames:
            element_tree_config["is_top"] = False

        return {
            "reader": ReaderType.STLSequence,
            "name": fName,
            "element_reader": element_tree_config,
        }

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        if tree_config["reader"] != ReaderType.STLSequence:
            return None

        element_reader = get_reader_instance(tree_config["element_reader"])
        is_top = tree_config.get("is_top", True)
        return Cpp_STLSequenceReader(tree_config["name"], is_top, element_reader)

    @classmethod
    def reconstruct_array(cls, raw_data, tree_config):
        if tree_config["reader"] != ReaderType.STLSequence:
            return None

        counts, element_raw_data = raw_data
        element_data = reconstruct_array(
            element_raw_data,
            tree_config["element_reader"],
        )
        return ak.unflatten(element_data, counts)


class STLMapReader(BaseReader):
    """
    This class reads std::map from a binary parser.
    """

    @staticmethod
    def get_map_key_val_typenames(type_name: str) -> tuple[str, str]:
        """
        Get the key and value type names of a map type.

        e.g. map<int, vector<int>> -> (int, vector<int>)
        """
        type_name = (
            type_name.replace("std::", "").replace("< ", "<").replace(" >", ">").strip()
        )
        return re.match(
            r"^(map|unordered_map|multimap)<(.*),(.*)>$", type_name
        ).groups()[1:3]

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name,
        cls_streamer_info,
        all_streamer_info,
        item_path,
    ):
        if top_type_name not in ["map", "unordered_map", "multimap"]:
            return None

        fTypeName = cls_streamer_info["fTypeName"]
        key_type_name, val_type_name = cls.get_map_key_val_typenames(fTypeName)

        fName = cls_streamer_info["fName"]
        key_info = {
            "fName": "key",
            "fTypeName": key_type_name,
        }

        val_info = {
            "fName": "val",
            "fTypeName": val_type_name,
        }

        key_tree_config = gen_tree_config(key_info, all_streamer_info, item_path)
        if get_top_type_name(key_type_name) in stl_typenames:
            key_tree_config["is_top"] = False

        val_tree_config = gen_tree_config(val_info, all_streamer_info, item_path)
        if get_top_type_name(val_type_name) in stl_typenames:
            val_tree_config["is_top"] = False

        return {
            "reader": ReaderType.STLMap,
            "name": fName,
            "key_reader": key_tree_config,
            "val_reader": val_tree_config,
        }

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        if tree_config["reader"] != ReaderType.STLMap:
            return None

        key_cpp_reader = get_reader_instance(tree_config["key_reader"])
        val_cpp_reader = get_reader_instance(tree_config["val_reader"])
        is_top = tree_config.get("is_top", True)
        return Cpp_STLMapReader(
            tree_config["name"],
            is_top,
            key_cpp_reader,
            val_cpp_reader,
        )

    @classmethod
    def reconstruct_array(cls, raw_data, tree_config):
        if tree_config["reader"] != ReaderType.STLMap:
            return None

        key_tree_config = tree_config["key_reader"]
        val_tree_config = tree_config["val_reader"]
        counts, key_raw_data, val_raw_data = raw_data
        key_data = reconstruct_array(key_raw_data, key_tree_config)
        val_data = reconstruct_array(val_raw_data, val_tree_config)

        return ak.unflatten(
            ak.zip(
                {
                    key_tree_config["name"]: key_data,
                    val_tree_config["name"]: val_data,
                },
                with_name="pair",
            ),
            counts,
        )


class STLStringReader(BaseReader):
    """
    This class reads std::string from a binary parser.
    """

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name,
        cls_streamer_info,
        all_streamer_info,
        item_path,
    ):
        if top_type_name != "string":
            return None

        return {
            "reader": ReaderType.STLString,
            "name": cls_streamer_info["fName"],
        }

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        if tree_config["reader"] != ReaderType.STLString:
            return None

        return Cpp_STLStringReader(
            tree_config["name"],
            tree_config.get("is_top", True),
        )

    @classmethod
    def reconstruct_array(cls, raw_data, tree_config):
        if tree_config["reader"] != ReaderType.STLString:
            return None

        counts, data = raw_data
        return ak.enforce_type(ak.unflatten(data, counts), "string")


class TArrayReader(BaseReader):
    """
    This class reads TArray from a binary paerser.

    TArray includes TArrayC, TArrayS, TArrayI, TArrayL, TArrayF, and TArrayD.
    Corresponding ctype is u1, u2, i4, i8, f, and d.
    """

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name,
        cls_streamer_info,
        all_streamer_info,
        item_path,
    ):
        if top_type_name not in tarray_typenames:
            return None

        ctype = tarray_typenames[top_type_name]
        return {
            "reader": ReaderType.TArray,
            "name": cls_streamer_info["fName"],
            "ctype": ctype,
        }

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        if tree_config["reader"] != ReaderType.TArray:
            return None

        return Cpp_TArrayReader(tree_config["name"], tree_config["ctype"])

    @classmethod
    def reconstruct_array(cls, raw_data, tree_config):
        if tree_config["reader"] != ReaderType.TArray:
            return None

        counts, data = raw_data
        return ak.unflatten(data, counts)


class TStringReader(BaseReader):
    """
    This class reads TString from a binary parser.
    """

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name,
        cls_streamer_info,
        all_streamer_info,
        item_path,
    ):
        if top_type_name != "TString":
            return None

        return {
            "reader": ReaderType.TString,
            "name": cls_streamer_info["fName"],
        }

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        if tree_config["reader"] != ReaderType.TString:
            return None

        return Cpp_TStringReader(tree_config["name"])

    @classmethod
    def reconstruct_array(cls, raw_data, tree_config):
        if tree_config["reader"] != ReaderType.TString:
            return None

        counts, data = raw_data
        return ak.enforce_type(ak.unflatten(data, counts), "string")


class TObjectReader(BaseReader):
    """
    This class reads TObject from a binary parser.

    It will not record any data.
    """

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name,
        cls_streamer_info,
        all_streamer_info,
        item_path,
    ):
        if top_type_name != "BASE":
            return None

        fType = cls_streamer_info["fType"]
        if fType != 66:
            return None

        return {
            "reader": ReaderType.TObject,
            "name": cls_streamer_info["fName"],
        }

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        if tree_config["reader"] != ReaderType.TObject:
            return None

        return Cpp_TObjectReader(tree_config["name"])

    @classmethod
    def reconstruct_array(cls, raw_data, tree_config):
        return None


class CArrayReader(BaseReader):
    """
    This class reads a C-array from a binary parser.
    """

    @classmethod
    def priority(cls):
        return 100  # This reader should be called first

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name,
        cls_streamer_info,
        all_streamer_info,
        item_path,
    ):
        if cls_streamer_info.get("fArrayDim", 0) == 0:
            return None

        fName = cls_streamer_info["fName"]
        fTypeName = cls_streamer_info["fTypeName"]
        fArrayDim = cls_streamer_info["fArrayDim"]
        fMaxIndex = cls_streamer_info["fMaxIndex"]

        element_streamer_info = cls_streamer_info.copy()
        element_streamer_info["fArrayDim"] = 0

        element_tree_config = gen_tree_config(
            element_streamer_info,
            all_streamer_info,
        )

        flat_size = np.prod(fMaxIndex[:fArrayDim])
        assert (
            flat_size > 0
        ), f"flatten_size should be greater than 0, but got {flat_size}"

        # c-type number or TArray
        if top_type_name in num_typenames or top_type_name in tarray_typenames:
            return {
                "reader": ReaderType.CArray,
                "name": fName,
                "is_obj": False,
                "element_reader": element_tree_config,
                "flat_size": flat_size,
                "fMaxIndex": fMaxIndex,
                "fArrayDim": fArrayDim,
            }

        # TSTring
        elif top_type_name == "TString":
            return {
                "reader": ReaderType.CArray,
                "name": fName,
                "is_obj": True,
                "element_reader": element_tree_config,
                "flat_size": flat_size,
                "fMaxIndex": fMaxIndex,
                "fArrayDim": fArrayDim,
            }

        # STL
        elif top_type_name in stl_typenames:
            element_tree_config["is_top"] = False
            return {
                "reader": ReaderType.CArray,
                "name": fName,
                "is_obj": True,
                "flat_size": flat_size,
                "element_reader": element_tree_config,
                "fMaxIndex": fMaxIndex,
                "fArrayDim": fArrayDim,
            }

        else:
            raise ValueError(f"Unknown type: {top_type_name} for C-array: {fTypeName}")

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        reader_type = tree_config["reader"]
        if reader_type != ReaderType.CArray:
            return None

        element_reader = get_reader_instance(tree_config["element_reader"])

        return Cpp_CArrayReader(
            tree_config["name"],
            tree_config["is_obj"],
            tree_config["flat_size"],
            element_reader,
        )

    @classmethod
    def reconstruct_array(cls, raw_data, tree_config):
        if tree_config["reader"] != ReaderType.CArray:
            return None

        element_tree_config = tree_config["element_reader"]
        fMaxIndex = tree_config["fMaxIndex"]
        fArrayDim = tree_config["fArrayDim"]
        shape = [fMaxIndex[i] for i in range(fArrayDim)]

        element_data = reconstruct_array(
            raw_data,
            element_tree_config,
        )

        for s in shape[::-1]:
            element_data = ak.unflatten(element_data, s)

        return element_data


class BaseObjectReader(BaseReader):
    """
    Base class is what a custom class inherits from.
    It has fNBytes(uint32), fVersion(uint16) at the beginning.
    """

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name,
        cls_streamer_info,
        all_streamer_info,
        item_path,
    ):
        if top_type_name != "BASE":
            return None

        fType = cls_streamer_info["fType"]
        if fType != 0:
            return None

        fName = cls_streamer_info["fName"]
        sub_streamers: list = all_streamer_info[fName]

        sub_tree_configs = [
            gen_tree_config(s, all_streamer_info, item_path) for s in sub_streamers
        ]

        return {
            "reader": ReaderType.BaseObject,
            "name": fName,
            "sub_readers": sub_tree_configs,
        }

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        if tree_config["reader"] != ReaderType.BaseObject:
            return None

        sub_readers = [get_reader_instance(s) for s in tree_config["sub_readers"]]
        return Cpp_BaseObjectReader(tree_config["name"], sub_readers)

    @classmethod
    def reconstruct_array(cls, raw_data, tree_config):
        if tree_config["reader"] != ReaderType.BaseObject:
            return None

        sub_tree_configs = tree_config["sub_readers"]

        arr_dict = {}
        for s_cfg, s_data in zip(sub_tree_configs, raw_data):
            s_name = s_cfg["name"]
            s_reader_type = s_cfg["reader"]

            if s_reader_type == ReaderType.TObject:
                continue

            arr_dict[s_name] = reconstruct_array(s_data, s_cfg)

        return ak.Array(arr_dict)


class ObjectHeaderReader(BaseReader):
    """
    This class read an object starting with an object header.
    """

    @classmethod
    def priority(cls):
        return 0  # should be called last

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name,
        cls_streamer_info,
        all_streamer_info,
        item_path,
    ):
        sub_streamers: list = all_streamer_info[top_type_name]
        sub_tree_configs = [
            gen_tree_config(s, all_streamer_info, item_path) for s in sub_streamers
        ]
        return {
            "reader": ReaderType.ObjectHeader,
            "name": top_type_name,
            "sub_readers": sub_tree_configs,
        }

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        if tree_config["reader"] != ReaderType.ObjectHeader:
            return None

        sub_readers = [get_reader_instance(s) for s in tree_config["sub_readers"]]
        return Cpp_ObjectHeaderReader(tree_config["name"], sub_readers)

    @classmethod
    def reconstruct_array(cls, raw_data, tree_config):
        if tree_config["reader"] != ReaderType.ObjectHeader:
            return None

        sub_tree_configs = tree_config["sub_readers"]

        arr_dict = {}
        for s_cfg, s_data in zip(sub_tree_configs, raw_data):
            s_name = s_cfg["name"]
            s_reader_type = s_cfg["reader"]

            if s_reader_type == ReaderType.TObject:
                continue

            arr_dict[s_name] = reconstruct_array(s_data, s_cfg)

        return ak.Array(arr_dict)


class EmptyReader(BaseReader):
    """
    This class does nothing.
    """

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name,
        cls_streamer_info,
        all_streamer_info,
        item_path,
    ):
        return None

    @classmethod
    def get_reader_instance(cls, tree_config: dict):
        if tree_config["reader"] != ReaderType.Empty:
            return None

        return Cpp_EmptyReader(tree_config["name"])

    @classmethod
    def reconstruct_array(cls, raw_data, tree_config):
        if tree_config["reader"] != ReaderType.Empty:
            return None

        return np.empty(shape=(0,))


readers |= {
    CTypeReader,
    STLSequenceReader,
    STLMapReader,
    STLStringReader,
    TArrayReader,
    TStringReader,
    TObjectReader,
    CArrayReader,
    BaseObjectReader,
    ObjectHeaderReader,
    EmptyReader,
}


class AsCustom(uproot.CustomInterpretation):
    target_branches: set[str] = set()

    def __init__(self, branch, context, simplify):
        super().__init__(branch, context, simplify)

        # simplify streamer information
        self.all_streamer_info: dict[str, list[dict]] = {}
        for k, v in branch.file.streamers.items():
            cur_infos = [
                i.all_members for i in next(iter(v.values())).member("fElements")
            ]
            self.all_streamer_info[k] = cur_infos

    @classmethod
    def match_branch(
        cls,
        branch: uproot.behaviors.TBranch.TBranch,
        context: dict,
        simplify: bool,
    ) -> bool:
        """
        Args:
            branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch`` to
                interpret as an array.
            context (dict): Auxiliary data used in deserialization.
            simplify (bool): If True, call
                :ref:`uproot.interpretation.objects.AsObjects.simplify` on any
                :doc:`uproot.interpretation.objects.AsObjects` to try to get a
                more efficient interpretation.

        Accept arguments from `uproot.interpretation.identify.interpretation_of`,
        determine whether this interpretation can be applied to the given branch.
        """
        full_path = regularize_object_path(branch.object_path)
        return full_path in cls.target_branches

    def __repr__(self) -> str:
        """
        The string representation of the interpretation.
        """
        return f"AsCustom({self.typename})"

    def basket_array(
        self,
        data,
        byte_offsets,
        basket,
        branch,
        context,
        cursor_offset,
        library,
        interp_options,
    ):
        assert library.name == "ak", "Only awkward arrays are supported"

        full_branch_path = regularize_object_path(branch.object_path)

        # generate reader config
        tree_config = gen_tree_config_from_type_name(
            branch.streamer.typename, self.all_streamer_info, full_branch_path
        )

        # get reader
        reader = get_reader_instance(tree_config)

        # read data
        parser = Cpp_BinaryParser(data, byte_offsets)
        for _ in range(parser.n_entries):
            reader.read(parser)

        # recover raw data and return
        raw_data = reader.get_data()
        return reconstruct_array(raw_data, tree_config)


uproot.register_interpretation(AsCustom)
