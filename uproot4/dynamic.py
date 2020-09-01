# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Initially empty submodule into which new classes are dynamically added.
"""


def __getattr__(name):
    import uproot4.model
    import uproot4.deserialization
    import uproot4._util

    g = globals()
    if name not in g:
        g[name] = uproot4._util.new_class(name, (uproot4.model.DynamicModel,), {})

    return g[name]
