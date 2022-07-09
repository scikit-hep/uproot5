# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines utilities for adding components to the forth reader.
"""


class ForthGenerator:
    """
    This class is passed through the Forth code generation, collecting Forth snippets and concatenating them at the end.
    """

    def __init__(self, aform=None, count_obj=0, var_set=False):
        self.aform = aform
        self.top_form = None
        self.awkward_model = {}
        self.forth_code = {}
        self.forth_keys = {}
        self.final_code = []
        self.final_header = []
        self.final_init = []
        self.form_keys = []
        self.var_set = var_set
        self.count_obj = count_obj
        return

    def traverse_aform(self):
        self.aform = self.aform.content

    def should_add_form(self):
        return not bool(self.awkward_model)

    def add_form(self, aform):
        if self.aform is None:
            self.aform = aform
            self.top_form = self.aform

        else:
            if "content" in self.aform.keys():
                if self.aform["content"] == "NULL":
                    self.aform["content"] = aform
                    self.aform = self.aform["content"]
                else:
                    raise ValueError

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
            raise ValueError("Number of keys cannot be less than 1")

    def add_form_key(self, key):
        self.form_keys.append(key)

    def go_to(self, aform):
        aform["content"] = self.awkward_model
        self.awkward_model = aform

    def check_model(self):
        return bool(self.awkward_model)

    def add_node(
        self, name, precode, postcode, initcode, headercode, dtype, num_child, content
    ):
        if isinstance(self.awkward_model, dict):
            if not bool(self.awkward_model):
                self.awkward_model = {
                    "name": name,
                    "type": dtype,
                    "pre_code": precode,
                    "post_code": postcode,
                    "init_code": initcode,
                    "header_code": headercode,
                    "num_child": num_child,
                    "content": content,
                }
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
                    "pre_code": precode,
                    "post_code": postcode,
                    "init_code": initcode,
                    "header_code": headercode,
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
        return

    def add_to_final(self, code):
        if not isinstance(code, list):
            raise TypeError
        self.final_code.extend(code)
        return

    def add_to_header(self, code):
        self.final_header.append(code)
        return

    def add_to_init(self, code):
        self.final_init.append(code)
        return


class GenHelper:
    """
    Helper class to aid Forth code generation within one read/read_members function call.
    """

    def __init__(self, context):
        self.forth_present = False
        self._pre_code = []
        self._post_code = []
        self._header = ""
        self._init = ""
        self._form_key = []
        if "forth" in context.keys():
            self.forth_present = True
            self._gen_obj = context["forth"]

    def is_forth(self):
        return self.forth_present

    def get_gen_obj(self):
        return self._gen_obj

    def add_to_pre(self, code):
        self._pre_code.append(code)

    def get_pre(self):
        return self._pre_code

    def add_to_post(self, code):
        self._post_code.append(code)

    def get_post(self):
        return self._post_code

    def add_to_header(self, code):
        self._header += code

    def get_header(self):
        return self._header

    def add_to_init(self, code):
        self._init += code

    def get_init(self):
        return self._init
