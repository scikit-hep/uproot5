# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines utilities for adding components to the forth reader.
"""


class ForthObj:
    """
    This class is passed through the Forth code generation, collecting Forth snippets and concatenating them at the end.
    """

    def __init__(self, aform=None, count_obj=0, var_set=False):
        self.aform = aform
        self.forth_sequence = []
        self.forth_code = {}
        self.forth_keys = {}
        self.final_code = []
        self.final_header = []
        self.final_init = []
        self.form_keys = []
        self.var_set = var_set
        self.count_obj = count_obj
        return

    def init_keys(self, ref, start, stop):
        self.forth_keys[ref] = [start, stop]

    def traverse_aform(self):
        self.aform = self.aform.content

    def get_last_key(self):
        if len(self.forth_keys.keys()) > 0:
            return self.forth_keys[list(self.forth_keys.keys())[-1]][1]
        else:
            return -1

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

    def get_keys(self, ref):
        return self.forth_keys[ref]

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
