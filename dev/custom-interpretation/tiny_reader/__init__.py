from __future__ import annotations

import awkward as ak
from AsCustom import (
    AsCustom,
    BaseReader,
    ReaderType,
    gen_tree_config,
    get_reader_instance,
    readers,
    reconstruct_array,
)

from .should_be_cpp import Cpp_MyTObjArrayReader


class MyTObjArrayReader(BaseReader):
    """
    This class reads a TObjArray from a binary parser.

    I know that there is only 1 kind of class in the TObjArray I will read,
    so I can use only 1 reader to read all elements in TObjArray.
    """

    @classmethod
    def gen_tree_config(
        cls,
        top_type_name: str,
        cls_streamer_info: dict,
        all_streamer_info: dict,
        item_path: str = "",
    ):
        if top_type_name != "TObjArray":
            return None

        obj_typename = "TMySubObject"

        if obj_typename not in all_streamer_info:
            return {
                "reader": "MyTObjArrayReader",
                "name": cls_streamer_info["fName"],
                "element_reader": {
                    "reader": ReaderType.Empty,
                    "name": obj_typename,
                },
            }

        sub_reader_config = []
        for s in all_streamer_info[obj_typename]:
            sub_reader_config.append(
                gen_tree_config(s, all_streamer_info, item_path + f".{obj_typename}")
            )

        return {
            "reader": "MyTObjArrayReader",
            "name": cls_streamer_info["fName"],
            "element_reader": {
                "reader": ReaderType.ObjectHeader,
                "name": obj_typename,
                "sub_readers": sub_reader_config,
            },
        }

    @staticmethod
    def get_reader_instance(
        reader_config: dict,
    ):
        if reader_config["reader"] != "MyTObjArrayReader":
            return None

        element_reader_config = reader_config["element_reader"]
        element_reader = get_reader_instance(element_reader_config)

        return Cpp_MyTObjArrayReader(reader_config["name"], element_reader)

    @staticmethod
    def reconstruct_array(raw_data, reader_config: dict):
        if reader_config["reader"] != "MyTObjArrayReader":
            return None

        counts, element_raw_data = raw_data
        element_reader_config = reader_config["element_reader"]
        element_data = reconstruct_array(
            element_raw_data,
            element_reader_config,
        )

        return ak.unflatten(element_data, counts)


def register():
    readers.add(MyTObjArrayReader)
    AsCustom.target_branches.add("/my_tree:my_obj/m_obj_array")
