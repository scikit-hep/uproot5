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


class Forth_Generator:
    def __init__(self):
        self.final_code = []
        self.final_header = []
        self.final_init = []
        self.discovered_form = {"form_key": "TOP", "content": {}, "class": "TOP"}
        self.awkward_model = Node("TOP")
        self.node_count = 0
        self.form_keys = []
        self.previous_model = self.awkward_model
        self.context = []

    def add_node_to_model(self, new_node, current_node):
        if (new_node.parent_name == current_node.name) and new_node.parent_name != new_node.name:
            for child_node in current_node.children:
                if child_node.name == new_node.name:
                    return
            current_node.add_child(new_node)
        else:
            for child_node in current_node.children:
                self.add_node_to_model(new_node,child_node)
    
    def append_code(self,tree,node_name,code,case):
        if tree.name == node_name:
            tree.append_code_snippet(code,case)
        else:
            for child_node in tree.children:
                    self.append_code(child_node,node_name,code,case)
                

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
                    current_form["content"]["contents"].append(new_form)
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

    def should_add_form(self):
        if "content" in self.awkward_model.keys():
            if self.awkward_model["content"] is None:
                return False
            elif len(self.awkward_model["content"].keys()) == 0:
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

    def get_pre(self):
        return self._pre_code

    def get_post(self):
        return self._post_code

    def get_header(self):
        return self._header

    def get_init(self):
        return self._init

    def get_node(self):
        return self._node

    def get_form(self):
        return self._form
    
    def get_form_key(self):
        return self._form_key

    def set_node(
        self,
        name,
        dtype,
        parent_node,
    ):
        self._node = Node(name,dtype,self._pre_code,self._post_code,self._init,self._header,parent_node)

    def read_forth_AsVector(self, forth_generator, values):
        key = forth_generator.node_count
        forth_generator.increment_node_count()
        node_key = f"node{key}"
        form_key = f"node{key}-offsets"
        self.add_to_header(f"output node{key}-offsets int64\n")
        self.add_to_init(f"0 node{key}-offsets <- stack\n")
        self.add_to_pre(f"stream !I-> stack\n dup node{key}-offsets +<- stack\n")

        if forth_generator.previous_model.name != node_key:
            self.add_form_key(form_key)
            temp_aform = f'{{ "class":"ListOffsetArray", "offsets":"i64", "content": "NULL", "parameters": {{}}, "form_key": "node{key}"}}'
            self.add_form(json.loads(temp_aform))

        if not isinstance(values, numpy.dtype):
            self.add_to_pre("0 do\n")
            self.add_to_post("loop\n")

        self.set_node(
            node_key,
            "i64",
            forth_generator.previous_model.name,
        )

        forth_generator.add_node_to_model(self._node, forth_generator.awkward_model)
        forth_generator.add_form(
            self._form,
            forth_generator.discovered_form,
            forth_generator.previous_model.name,
        )
        forth_generator.append_form_key(self._form_key)
        forth_generator.update_previous_model(self._node)

    def read_nested_forth(self, forth_generator, symbol):
        key = forth_generator.node_count
        forth_generator.increment_node_count()
        node_key = f"node{key}"
        form_key = f"node{key}-data"
        self.add_to_header(f"output node{key}-data {convert_dtype(symbol)}\n")
        self.add_to_pre(f"stream #!{symbol}-> node{key}-data\n")
        if forth_generator.previous_model.name != node_key:
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
            forth_generator.previous_model.name,
        )

        forth_generator.add_node_to_model(self._node, forth_generator.awkward_model)
        forth_generator.add_form(
            self._form,
            forth_generator.discovered_form,
            forth_generator.previous_model.name,
        )
        forth_generator.append_form_key(self._form_key)

class Node:
    def __init__(self,name,dtype=None,pre_code=None,post_code=None,init_code=None,header_code=None,parent_node_name=None):
        self._name = name
        self._dtype = dtype
        self._pre_code = pre_code
        self._post_code = post_code
        self._init_code = init_code
        self._header_code = header_code
        self._parent_node_name = parent_node_name
        self._num_of_children = 0
        self._children = []

    def __repr__(self) -> str:
        return self._name
    
    def print_tree(self,level=0,node=None):
        if node is None:
            node = self
        for var in vars(node):
            print("  "*level,"{}: {}".format(var,vars(node)[var]))
        for child in node.children:
            self.print_tree(level+1,child)
        print()
    
    def add_child(self,child):
        self._children.append(child)
        self._num_of_children+=1
    
    def append_code_snippet(self,code,case):
        if case == "pre":
            self._pre_code.append(code)
        elif case == "post":
            self._post_code.append(code)
        elif case == "header":
            self._header_code += code
        elif case == "init":
            self._init_code += code

    @property
    def num_of_children(self):
        return self._num_of_children
    @property
    def children(self):
        return self._children
    @property
    def name(self):
        return self._name
    @property
    def parent_name(self):
        return self._parent_node_name
    
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
