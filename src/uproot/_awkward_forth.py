# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines utilities for adding components to the forth reader.
"""

import json

import numpy
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


class ForthGenerator:
    def __init__(self):
        self.final_code = []
        self.final_header = []
        self.final_init = []
        self.discovered_form = {"form_key": "TOP", "content": {}}
        self.awkward_model = {"name": "TOP", "content": {}}
        self.node_count = 0
        self.form_keys = []
        self.previous_model = {"name": "TOP", "content": {}}

    def add_node_to_model(self, new_node, current_node):
        if new_node["parent_node"] == current_node["name"]:
            current_node["content"].update(new_node)
        else:
            self.add_node_to_model(new_node, current_node["content"])

    def add_form(self, new_form, current_form, new_form_parent):
        if new_form_parent == current_form["form_key"]:
            if current_form["content"] == "NULL":
                current_form.update({"content": new_form})
            else:
                current_form["content"].update(new_form)
        else:
            self.add_form(new_form, current_form["content"], new_form_parent)

    def set_awkward_model(self, dictionary):
        self.awkward_model = dictionary

    def get_node_count(self):
        return self.node_count

    def update_node_count(self, value):
        self.node_count = value

    def increment_node_count(self):
        self.node_count += 1

    def set_expected_nodes(self, awkward_form):
        self.expected_nodes = awkward_form.branch_depth[1]

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


def should_add_form(awkward_model):
    if "content" in awkward_model.keys():
        if awkward_model["content"] is None:
            return False
        elif len(awkward_model["content"].keys()) == 0:
            return True
        else:
            raise Exception


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
        self._form_key = []
        self._form = None
        self._node = None

    def add_to_pre(self, code):
        self._pre_code.append(code)

    def add_to_post(self, code):
        self._post_code.append(code)

    def add_form_key(self, form_key):
        self._form_key = form_key

    def add_to_header(self, code):
        self._header += code

    def add_to_init(self, code):
        self._init += code

    def add_form(self, form):
        if self._form is None:
            self._form = form

    def set_node(
        self,
        name,
        dtype,
        precode,
        postcode,
        initcode,
        headercode,
        num_child,
        content,
        parent_node,
    ):
        self._node = {
            "name": name,
            "type": dtype,
            "pre_code": precode,
            "post_code": postcode,
            "init_code": initcode,
            "header_code": headercode,
            "num_child": num_child,
            "content": content,
            "parent_node": parent_node,
        }

    def read_forth_AsVector(self, forthGenerator, values):
        key = forthGenerator.node_count
        forthGenerator.increment_node_count()
        node_key = f"node{key}"
        form_key = f"node{key}-offsets"
        self.add_to_header(f"output node{key}-offsets int64\n")
        self.add_to_init(f"0 node{key}-offsets <- stack\n")
        self.add_to_pre(f"stream !I-> stack\n dup node{key}-offsets +<- stack\n")

        if forthGenerator.previous_model["name"] != node_key:
            self.add_form_key(form_key)
            temp_aform = f'{{ "class":"ListOffsetArray", "offsets":"i64", "content": "NULL", "parameters": {{}}, "form_key": "node{key}"}}'
            self.add_form(json.loads(temp_aform))

        if not isinstance(values, numpy.dtype):
            self.add_to_pre("0 do\n")
            self.add_to_post("loop\n")

        self.set_node(
            node_key,
            "i64",
            self._pre_code,
            self._post_code,
            self._init,
            self._header,
            1,
            {},
            forthGenerator.previous_model["name"],
        )

        forthGenerator.add_node_to_model(self._node, forthGenerator.awkward_model)
        forthGenerator.add_form(
            self._form,
            forthGenerator.discovered_form,
            forthGenerator.previous_model["name"],
        )
        forthGenerator.append_form_key(self._form_key)
        forthGenerator.update_previous_model(self._node)

    def read_nested_forth(self, forthGenerator, symbol):
        key = forthGenerator.node_count
        forthGenerator.increment_node_count()
        node_key = f"node{key}"
        form_key = f"node{key}-data"
        self.add_to_header(f"output node{key}-data {convert_dtype(symbol)}\n")
        self.add_to_pre(f"stream #!{symbol}-> node{key}-data\n")
        if forthGenerator.previous_model["name"] != node_key:
            self.add_form_key(form_key)
            self.add_form(
                {
                    "class": "NumpyArray",
                    "primitive": f"{convert_dtype(symbol)}",
                    "form_key": f"node{key}",
                }
            )

        self.set_node(
            f"node{key}",
            "i64",
            self._pre_code,
            [],
            "",
            self._header,
            1,
            None,
            forthGenerator.previous_model["name"],
        )

        forthGenerator.add_node_to_model(self._node, forthGenerator.awkward_model)
        forthGenerator.add_form(
            self._form,
            forthGenerator.discovered_form,
            forthGenerator.previous_model["name"],
        )
        forthGenerator.append_form_key(self._form_key)


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
