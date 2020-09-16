# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Initially empty submodule into which new classes are dynamically added.

The purpose of this namespace is to allow :py:class:`~uproot4.model.VersionedModel`
classes that were automatically generated from ROOT ``TStreamerInfo`` to be
pickled, with the help of :py:class:`~uproot4.model.DynamicModel`.

In `Python 3.7 and later <https://www.python.org/dev/peps/pep-0562>`__, attempts
to extract items from this namespace generate new :py:class:`~uproot4.model.DynamicModel`
classes, which are used as a container in which data from pickled
:py:class:`~uproot4.model.VersionedModel` instances are filled.
"""


def __getattr__(name):
    import uproot4.model
    import uproot4.deserialization
    import uproot4._util

    g = globals()
    if name not in g:
        g[name] = uproot4._util.new_class(name, (uproot4.model.DynamicModel,), {})

    return g[name]
