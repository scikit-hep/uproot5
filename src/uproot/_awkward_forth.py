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


class Forth_Generator:
    def __init__(self, interp):
        self._interp = interp
        self.final_code = []
        self.final_header = []
        self.final_init = []
        self.discovered_form = {"form_key": "TOP", "content": {}, "class": "TOP"}
        self.awkward_model = Node("TOP")
        self.key_number = 0
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

    def append_code(self, tree, node_name, code, case):
        if tree.name == node_name:
            tree.append_code_snippet(code, case)
        else:
            for child_node in tree.children:
                self.append_code(child_node, node_name, code, case)

    def add_form(self, new_form, current_form, new_form_parent):
        if current_form["class"] == "RecordArray":
            for form_dict in current_form["contents"]:
                self.add_form(new_form, form_dict, new_form_parent)
        elif "content" in current_form.keys():
            if new_form_parent == current_form["form_key"]:
                if current_form["content"] == "NULL":
                    current_form.update({"content": new_form})
                elif not current_form["content"]:
                    current_form["content"].update(new_form)
                elif current_form["content"]["class"] == "RecordArray":
                    for children in current_form["content"]["contents"]:
                        if children["form_key"] == new_form["form_key"]:
                            return
                    current_form["content"]["contents"].append(new_form)
            else:
                self.add_form(new_form, current_form["content"], new_form_parent)

    def get_key_number(self):
        return self.key_number

    def update_key_number(self, value):
        self.key_number = value

    def increment_key_number(self):
        self.key_number += 1

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


def forth_stash(context):
    """
    Returns a ForthLevelStash object if ForthGeneration is to be done, else None.
    """
    if hasattr(context.get("forth"), "gen"):
        return ForthStash()
    else:
        return None


class ForthStash:
    def __init__(self):
        self._pre_code = []
        self._post_code = []
        self._header = ""
        self._init = ""
        self._node = None

    @property
    def pre_code(self):
        return self._pre_code

    @property
    def post_code(self):
        return self._post_code

    @property
    def header_code(self):
        return self._header

    @property
    def init_code(self):
        return self._init

    @property
    def node(self):
        return self._node

    def add_to_pre(self, code):
        self._pre_code.append(code)

    def add_to_post(self, code):
        self._post_code.append(code)

    def add_to_header(self, code):
        self._header += code

    def add_to_init(self, code):
        self._init += code

    def set_node(self, name, dtype):
        self._node = Node(
            name,
            dtype,
            self._pre_code,
            self._post_code,
            self._init,
            self._header,
        )


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
        children=None,
    ):
        self._name = name
        self._dtype = dtype
        self._pre_code = [] if pre_code is None else pre_code
        self._post_code = [] if post_code is None else post_code
        self._init_code = "" if init_code is None else init_code
        self._header_code = "" if header_code is None else header_code
        self._form_details = {} if form_details is None else form_details
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

    def add_child(self, child):
        self._children.append(child)

    def set_node(self, name, dtype):
        # FIXME: get rid of this function!
        self._name = name
        self._dtype = dtype

    def append_code_snippet(self, code, case):
        # FIXME: get rid of this function!
        if case == "pre":
            self._pre_code.append(code)
        elif case == "post":
            self._post_code.append(code)
        elif case == "header":
            self._header_code += code
        elif case == "init":
            self._init_code += code

    def derive_form(self):
        if self._form_details.get("class") == "NumpyArray":
            assert len(self._children) == 0
            return self._form_details

        elif self._form_details.get("class") == "ListOffsetArray":
            assert len(self._children) == 1
            out = dict(self._form_details)
            out["content"] = self._children[0].derive_form()
            return out

        elif self._form_details.get("class") == "RecordArray":
            out = dict(self._form_details)
            out["fields"] = []
            out["contents"] = []
            for child in self._children:
                if child.name.startswith("base-class "):
                    assert len(child._children) == 1
                    assert child._children[0].name.startswith("start-of-model ")
                    assert len(child._children[0]._children) == 1
                    assert (
                        child._children[0]._children[0]._form_details.get("class")
                        == "RecordArray"
                    )
                    base_form = child._children[0]._children[0].derive_form()
                    out["fields"].extend(base_form["fields"])
                    out["contents"].extend(base_form["contents"])
                else:
                    assert ":" in child.name
                    out["fields"].append(child.name.split(":", 1)[-1])
                    out["contents"].append(child.derive_form())

            return out

        elif (
            self._name == "TOP"
            or self._name.startswith("dispatch-by-version ")
            or self._name.startswith("wrong-instance-version ")
            or self._name.startswith("start-of-model ")
        ):
            assert len(self._children) == 1
            return self._children[0].derive_form()

        else:
            return "NULL"


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
