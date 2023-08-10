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

class ForthGenerator:

    def __init__(self):
        self.final_code = []
        self.final_header = []
        self.final_init = []
        self.discovered_form = None
        self.awkward_model = {"name": "TOP", "content": {}}
        self.node_count = 0
        self.forth_stashes = []
    
    def add_forth_stash(self, ForthStash):
        self.forth_stashes.append(ForthStash)
    
    def increment_node_count(self):
        self.node_count += 1

def should_add_form(awkward_model):
    if "content" in awkward_model.keys():
        if awkward_model["content"] is None:
            return False
        elif len(awkward_model["content"].keys()) == 0:
            return True
        else:
            raise Exception
    
def forth_stash(context,previous_model):
    """
    Returns a ForthLevelStash object if ForthGeneration is to be done, else None.
    """
    if hasattr(context.get("forth"), "gen"):
        return ForthStash(previous_model)
    else:
        return None

class ForthStash:
    def __init__(self,previous_model):
        self._pre_code = []
        self._post_code = []
        self._header = ""
        self._init = ""
        self._form_key = []
        self._form = None
        self._node = None
        self._previous_model = previous_model

    def add_to_pre(self, code):
        self._pre_code.append(code)

    def add_to_post(self, code):
        self._post_code.append(code)
    
    def add_form_key(self,form_key):
        self._form_key=form_key
    
    def add_to_header(self, code):
        self._header += code

    def add_to_init(self, code):
        self._init += code
    
    def add_form(self,form):
        if self._form is None:
            self._form = form
    
    def set_node(self,name,dtype,precode,postcode,initcode,headercode,num_child,content):
        self._node={
        "name": name,
        "type": dtype,
        "pre_code": precode,
        "post_code": postcode,
        "init_code": initcode,
        "header_code": headercode,
        "num_child": num_child,
        "content": content,
    }
        return self._node
        
    
    


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
    