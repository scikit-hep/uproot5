# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines utilities for adding components to the forth reader.
"""

import numpy as np

symbol_dict = {
    np.dtype("bool"): "?",
    np.dtype(">u1"): "B",
    np.dtype(">u2"): "H",
    np.dtype(">u4"): "I",
    np.dtype(">u8"): "Q",
    np.dtype(">i1"): "b",
    np.dtype(">i2"): "h",
    np.dtype(">i4"): "i",
    np.dtype(">i8"): "q",
    np.dtype(">f4"): "f",
    np.dtype(">f8"): "d",
}


class SpecialPathItem:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"SpecialPathItem({self.name!r})"

    def __eq__(self, other):
        return isinstance(other, SpecialPathItem) and self.name == other.name

    def __hash__(self):
        return hash((SpecialPathItem, self.name))


def add_to_path(forth_obj, context, field):
    if forth_obj is not None:
        context = dict(context)
        context["path"] = context["path"] + (field,)
    return context


def get_first_key_number(context, extra_fields=()):
    return abs(hash(context["path"] + extra_fields))


class Forth_Generator:
    def __init__(self, interp):
        self._interp = interp
        self.final_code = []
        self.final_header = []
        self.final_init = []
        self.discovered_form = {"form_key": "TOP", "content": {}, "class": "TOP"}
        self.awkward_model = Node("TOP")
        self.form_keys = []
        self.previous_model = self.awkward_model

    def _debug_forth(self):
        self._interp._debug_forth(self)

    def add_node_to_model(self, new_node, current_node=None, parent_node_name=None):
        if current_node is None:
            current_node = self.awkward_model
        if parent_node_name is None:
            parent_node_name = self.previous_model.name

        if (
            parent_node_name == current_node.name
        ) and parent_node_name != new_node.name:
            for child_node in current_node.children:
                if child_node.name == new_node.name:
                    return
            current_node.add_child(new_node)
        else:
            for child_node in current_node.children:
                self.add_node_to_model(new_node, child_node, parent_node_name)

    def add_to_header(self, code):
        self.final_header += code

    def add_to_init(self, code):
        self.final_init += code

    def add_to_final(self, code):
        self.final_code.extend(code)

    def append_form_key(self, key):
        if key not in self.form_keys:
            self.form_keys.append(key)

    def update_previous_model(self, model):
        self.previous_model = model


def get_forth_obj(context):
    """
    Returns a the Forth Generator object if ForthGeneration is to be done, else None.
    """
    if hasattr(context.get("forth"), "gen"):
        return context["forth"].gen
    else:
        return None


class Node:
    def __init__(
        self,
        name,
        dtype=None,
        pre_code=None,
        post_code=None,
        init_code=None,
        header_code=None,
        form_details=None,
        field_name=None,
        children=None,
    ):
        self._name = name
        self._dtype = dtype
        self._pre_code = [] if pre_code is None else pre_code
        self._post_code = [] if post_code is None else post_code
        self._init_code = "" if init_code is None else init_code
        self._header_code = "" if header_code is None else header_code
        self._form_details = {} if form_details is None else form_details
        self._field_name = field_name
        self._children = [] if children is None else children

    def __str__(self) -> str:
        return self._name

    def get_dict(self):
        dictionary = vars(self).copy()
        children_dicts = [i.get_dict() for i in self._children]
        dictionary["_children"] = children_dicts
        return dictionary

    @property
    def name(self):
        return self._name

    @property
    def pre_code(self):
        return self._pre_code

    @property
    def post_code(self):
        return self._post_code

    @property
    def header_code(self):
        return self._header_code

    @property
    def init_code(self):
        return self._init_code

    @property
    def form_details(self):
        return self._form_details

    @property
    def field_name(self):
        return self._field_name

    @property
    def children(self):
        return self._children

    @property
    def node(self):
        return self

    def add_to_pre(self, code):
        self._pre_code.append(code)

    def add_to_post(self, code):
        self._post_code.append(code)

    def add_to_header(self, code):
        self._header_code += code

    def add_to_init(self, code):
        self._init_code += code

    def add_form_details(self, form_details):
        self._form_details = form_details

    def change_field_name(self, new_name):
        self._field_name = new_name

    def add_child(self, child):
        self._children.append(child)

    def change_name(self, new_name):
        self._name = new_name

    def derive_form(self):
        if self._name.endswith(":prebuilt"):
            return self._form_details
        elif self._form_details.get("class") == "NumpyArray":
            assert len(self._children) == 0
            return self._form_details

        elif self._form_details.get("class") in ("ListOffsetArray", "RegularArray"):
            out = dict(self._form_details)

            if (
                self._form_details.get("parameters", {}).get("__array__")
                in ("string", "bytestring")
                and self._form_details.get("content", {}).get("class") == "NumpyArray"
            ):
                return out

            elif len(self._children) == 0:
                if out.get("content", {}).get("class") != "NumpyArray":
                    out["content"] = "NULL"
                return out

            elif (
                len(self._children) == 2
                and self._form_details.get("content", {}).get("class") == "RecordArray"
                and self._form_details["content"].get("parameters", {}).get("__array__")
                == "sorted_map"
            ):
                out["content"] = dict(out["content"])
                out["content"]["fields"] = None
                out["content"]["contents"] = []
                for child in self._children:
                    out["content"]["contents"].append(child.derive_form())
                return out

            assert len(self._children) == 1
            returnme = out
            while out.get("content", {}).get("class") in (
                "ListOffsetArray",
                "RegularArray",
            ):
                out = out["content"]
            out["content"] = self._children[0].derive_form()
            return returnme

        elif (
            self._name == "TOP"
            or self._name.startswith("dispatch-by-version ")
            or self._name.startswith("wrong-instance-version ")
            or self._name.startswith("start-of-model ")
        ):
            assert len(self._children) == 1
            return self._children[0].derive_form()

        else:
            # import json
            # print(json.dumps(self.get_dict(), indent=4))

            out = dict(self._form_details)
            out["class"] = "RecordArray"
            out["fields"] = []
            out["contents"] = []
            for child in self.children:
                if child.name.startswith("base-class "):
                    assert len(child.children) == 1
                    descendant = child.children[0]
                    if descendant.name.startswith("dispatch-by-version "):
                        assert len(descendant.children) == 1
                        descendant = descendant.children[0]
                    assert descendant.name.startswith("start-of-model ")
                    assert len(descendant.children) == 1
                    base_form = descendant.children[0].derive_form()
                    out["fields"].extend(base_form["fields"])
                    out["contents"].extend(base_form["contents"])
                else:
                    descendant = child
                    while descendant.field_name is None:
                        assert descendant.form_details.get("class") in (
                            "ListOffsetArray",
                            "RegularArray",
                        )
                        assert len(descendant.children) == 1
                        descendant = descendant.children[0]
                    out["fields"].append(descendant.field_name)
                    out["contents"].append(child.derive_form())
            return out


class UnwindProtect:
    def __init__(self, forth_obj, temporary_model):
        self.forth_obj = forth_obj
        self.temporary_model = temporary_model
        self.old_model = None

    def __enter__(self):
        self.old_model = self.forth_obj.previous_model
        self.forth_obj.update_previous_model(self.temporary_model)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.forth_obj.update_previous_model(self.old_model)


def convert_dtype(format):
    """Takes datatype codes from classses and returns the full datatype name.

    Args:
        format (string): The datatype in the dynamic class

    Returns:
        string: The datatype in words.
    """
    if format == "?":
        return "bool"
    elif format == "B":
        return "uint8"
    elif format == "H":
        return "uint16"
    elif format == "I":
        return "uint32"
    elif format == "Q":
        return "uint64"
    elif format == "b":
        return "int8"
    elif format == "h":
        return "int16"
    elif format == "i":
        return "int32"
    elif format == "q":
        return "int64"
    elif format == "f":
        return "float32"
    elif format == "d":
        return "float64"
    else:
        raise AssertionError(f"unexpected format type: {format!r}")
