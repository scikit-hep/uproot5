# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines utilities for adding components to the forth reader.
"""

import numpy as np

symbol_dict = {
    np.dtype(">f4"): "f",
    np.dtype(">f8"): "d",
    np.dtype(">i8"): "q",
    np.dtype(">i4"): "i",
    np.dtype(">i2"): "h",
    np.dtype(">u1"): "B",
    np.dtype(">u2"): "H",
    np.dtype(">u4"): "I",
    np.dtype(">u8"): "Q",
    np.dtype("bool"): "?",
}


class ForthGenerator:
    """
    This class is passed through the Forth code generation, collecting Forth snippets and concatenating them at the end.
    """

    def __init__(self, aform=None, count_obj=0, var_set=False):
        self.dummy_form = False
        self.top_dummy = None
        self.dummy_aform = None
        self.aform = aform
        self.top_form = None
        self.prev_form = None
        self.awkward_model = {"name": "TOP", "content": {}}
        self.top_node = self.awkward_model
        self.ref_list = []
        self.forth_code = {}
        self.forth_keys = {}
        self.final_code = []
        self.final_header = []
        self.final_init = []
        self.form_keys = []
        self.var_set = var_set
        self.count_obj = count_obj

    def traverse_aform(self):
        self.aform = self.aform.content

    def replace_form_and_model(self, form, model):
        temp_node = self.awkward_model
        temp_prev_form = self.prev_form
        temp_node_top = self.top_node
        self.awkward_model = model
        self.top_node = self.awkward_model
        temp_form = self.aform
        temp_form_top = self.top_form
        self.top_form = None
        self.aform = form
        return temp_node, temp_node_top, temp_form, temp_form_top, temp_prev_form

    def get_code_recursive(self, node):
        pre, post, init, header = self.tree_walk(node)
        return pre, post, init, header

    def tree_walk(self, node):
        if "content" in node.keys():
            if node["content"] is None:
                return (
                    "".join(node["pre_code"]),
                    "".join(node["post_code"]),
                    node["init_code"],
                    node["header_code"],
                )
            else:
                pre, post, init, header = self.tree_walk(node["content"])
                pre2 = "".join(node["pre_code"])
                pre2 = pre2 + pre
                post2 = "".join(node["post_code"])
                post2 = post2 + post
                init = node["init_code"] + init
                header = node["header_code"] + header
                return pre2 + post2, "", init, header
        elif self.var_set:
            return "", "", "", ""

    def should_add_form(self):
        if "content" in self.awkward_model.keys():
            if self.awkward_model["content"] is None:
                return False
            else:
                if len(self.awkward_model["content"].keys()) == 0:
                    return True
                elif self.awkward_model["content"]["name"] == "dynamic":
                    return True
                else:
                    return False

    def get_temp_form_top(self):
        return self.top_dummy

    def add_form(self, aform, conlen=0, traverse=1):
        if self.aform is None:
            self.aform = aform
            self.top_form = self.aform
            if traverse == 2:
                self.aform = self.aform["content"]
        else:
            if "content" in self.aform.keys():
                if self.aform["content"] == "NULL":
                    self.aform["content"] = aform
                    self.prev_form = self.aform
                    if traverse == 2:
                        self.aform = self.aform["content"]["content"]
                        self.prev_form = self.prev_form["content"]
                    else:
                        self.aform = self.aform["content"]
                else:
                    raise AssertionError("Form representation is corrupted.")
            elif "contents" in self.aform.keys():
                if self.aform["class"] == "RecordArray":
                    if self.prev_form is not None:
                        self.prev_form["content"] = aform
                        self.aform = aform
                    else:
                        self.top_form = aform
                        self.aform = aform
                elif len(self.aform["contents"]) == conlen:
                    pass
                else:
                    return False

    def set_dummy_none(self, temp_top, temp_form, temp_flag):
        self.top_dummy = temp_top
        self.dummy_aform = temp_form
        self.dummy_form = temp_flag

    def get_keys(self, num_keys):
        if num_keys == 1:
            key = self.count_obj
            self.count_obj += 1
            return key
        elif num_keys > 1:
            out = []
            for _i in range(num_keys):
                out.append(self.count_obj)
                self.count_obj += 1
            return out
        else:
            raise AssertionError("Number of keys cannot be less than 1")

    def add_form_key(self, key):
        self.form_keys.append(key)

    def go_to(self, aform):
        self.awkward_model = aform

    def become(self, aform):
        self.awkward_model = aform

    def check_model(self):
        return bool(self.awkward_model)

    def get_current_node(self):
        self.ref_list.append(self.top_node)
        return len(self.ref_list) - 1

    def get_ref(self, index):
        return self.ref_list[index]

    def enable_adding(self):
        if "content" in self.awkward_model.keys():
            if self.awkward_model["content"] is None:
                self.awkward_model["content"] = {}

    def add_node_whole(self, new_node, ref_latest):
        if "content" in self.awkward_model.keys():
            self.awkward_model["content"] = new_node
        else:
            self.awkward_model["contents"].append(new_node)
        self.awkward_model = ref_latest

    def add_node(self, name, code_attrs, dtype, num_child, content):
        if isinstance(self.awkward_model, dict):
            if (
                not bool(self.awkward_model["content"])
                and self.awkward_model["content"] is not None
            ):
                temp_obj = {
                    "name": name,
                    "type": dtype,
                    "pre_code": code_attrs["precode"],
                    "post_code": code_attrs["postcode"],
                    "init_code": code_attrs["initcode"],
                    "header_code": code_attrs["headercode"],
                    "num_child": num_child,
                    "content": content,
                }
                self.awkward_model["content"] = temp_obj
                self.awkward_model = self.awkward_model["content"]
                return temp_obj
            else:
                if self.awkward_model["content"] is not None:
                    if self.awkward_model["content"]["name"] == "dynamic":
                        temp_obj = {
                            "name": name,
                            "type": dtype,
                            "pre_code": code_attrs["precode"],
                            "post_code": code_attrs["postcode"],
                            "init_code": code_attrs["initcode"],
                            "header_code": code_attrs["headercode"],
                            "num_child": num_child,
                            "content": content,
                        }
                        self.awkward_model["content"] = temp_obj
                        self.awkward_model = self.awkward_model["content"]
                        return temp_obj
                    else:
                        temp_node = self.awkward_model
                        self.awkward_model = self.awkward_model["content"]
                        return temp_node
                else:
                    temp_node = self.awkward_model
                    self.awkward_model = self.awkward_model["content"]
                    return temp_node
        if isinstance(self.awkward_model, list):
            for elem in self.awkward_model:
                if name in elem.values():
                    return elem
            self.awkward_model.append(
                {
                    "name": name,
                    "type": dtype,
                    "pre_code": code_attrs["precode"],
                    "post_code": code_attrs["postcode"],
                    "init_code": code_attrs["initcode"],
                    "header_code": code_attrs["headercode"],
                    "num_child": num_child,
                    "content": content,
                }
            )
            temp_node = self.awkward_model[-1]
            self.awkward_model = self.awkward_model[-1]["content"]
            return temp_node

    def register_pre(self, ref):
        key = str(ref) + "pre"
        self.forth_sequence.append(key)

    def add_meta(self, ref, form_key, header, init):
        if self.forth_code[id(ref)] is None:
            self.forth_code[id(ref)] = {}
        for elem in form_key:
            self.form_keys.append(elem)
        self.forth_code[id(ref)]["forth_header"] = header
        self.forth_code[id(ref)]["forth_init"] = init

    def register_post(self, ref):
        key = str(ref) + "post"
        self.forth_sequence.append(key)

    def add_forth_code(self, ref, forth_exec_pre, forth_exec_post):
        if self.forth_code[ref] is None:
            self.forth_code[ref] = {}
        self.forth_code[ref][str(ref) + "pre"] = forth_exec_pre
        self.forth_code[ref][str(ref) + "post"] = forth_exec_post

    def add_to_final(self, code):
        if not isinstance(code, list):
            raise AssertionError("Something went wrong with Forth code generation.")
        self.final_code.extend(code)

    def add_to_header(self, code):
        self.final_header.append(code)

    def add_to_init(self, code):
        self.final_init.append(code)


def forth_stash(context):
    """
    Returns a ForthLevelStash object if ForthGeneration is to be done, else None.
    """
    if hasattr(context.get("forth"), "gen"):
        return ForthLevelStash(context["forth"].gen)
    else:
        return None


class ForthLevelStash:
    """
    Helper class to stash code at one level of Forth code generation. Keeps the code generation clean and maintains order for the code snippets.
    """

    def __init__(self, context):
        self._pre_code = []
        self._post_code = []
        self._header = ""
        self._init = ""
        self._form_key = []
        self._gen_obj = context

    def is_forth(self):
        return self.forth_present

    def get_gen_obj(self):
        return self._gen_obj

    def add_to_pre(self, code):
        self._pre_code.append(code)

    def add_to_post(self, code):
        self._post_code.append(code)

    def get_attrs(self):
        return {
            "precode": self._pre_code,
            "postcode": self._post_code,
            "initcode": self._init,
            "headercode": self._header,
        }

    def add_to_header(self, code):
        self._header += code

    def add_to_init(self, code):
        self._init += code


def convert_dtype(format):
    """Takes datatype codes from classses and returns the full datatype name.

    Args:
        format (string): The datatype in the dynamic class

    Returns:
        string: The datatype in words.
    """
    if format == "d":
        return "float64"
    elif format == "f":
        return "float32"
    elif format == "q":
        return "int64"
    elif format == "i":
        return "int32"
    elif format == "I":
        return "uint32"
    elif format == "?":
        return "bool"
    elif format == "h":
        return "int16"
    elif format == "H":
        return "uint16"
    elif format == "Q":
        return "uint64"
    elif format == "B":
        return "uint8"
