# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines utilities for adding components to the forth reader.
"""

import awkward as ak


class forth_obj():

    def __init__(self, aform, count_obj=0 , var_set=False):
        self.aform = aform
        self.forth_sequence = []
        self.forth_code = {}
        self.forth_keys = {}
        self.final_code = []
        self.final_header = []
        self.final_init = []
        self.var_set = var_set
        self.count_obj = count_obj
        return

    def init_keys(self, ref, start, stop):
        self.forth_keys[ref] = [start, stop]

    def get_last_key(self):
        if len(self.forth_keys.keys()) > 0:
            return self.forth_keys[list(self.forth_keys.keys())[-1]][1]
        else:
            return -1

    def register_pre(self, ref):
        key = str(id(ref)) + "pre"
        self.forth_sequence.append(key)

    def register_post(self, ref):
        key = str(id(ref)) + "post"
        self.forth_sequence.append(key)

    def get_keys(self, ref):
        return self.forth_keys[ref]

    def add_forth_code(self, ref , forth_header, forth_exec_pre, forth_exec_post, forth_init):
        if self.forth_code[id(ref)] is None:
            self.forth_code[id(ref)] = {}
        self.forth_code[id(ref)]["forth_header"] = forth_header
        self.forth_code[id(ref)][str(id(ref)) + "pre"] = forth_exec_pre
        self.forth_code[id(ref)][str(id(ref)) + "post"] = forth_exec_post
        self.forth_code[id(ref)]["forth_init"] = forth_init
        return

    def add_to_final(self, code):
        if not isinstance(code, list):
            print(code)
            raise TypeError
        self.final_code.extend(code)
        return

    def add_to_header(self, code):
        if not isinstance(code, list):
            raise TypeError
        self.final_header.extend(code)
        return

    def add_to_init(self, code):
        if not isinstance(code, list):
            raise TypeError
        self.final_init.extend(code)
        return


class _PreReadDoneError(Exception):
    pass
