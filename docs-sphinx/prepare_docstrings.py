# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

import importlib
import inspect
import os.path

order = [
    "uproot4",
    "uproot4.reading",
    "uproot4.cache",
    "uproot4.model",
    "uproot4.streamers",
    "uproot4.compression",
    "uproot4.deserialization",
    "uproot4.source",
    "uproot4.interpretation",
    "uproot4.containers",
    "uproot4.language",
    "uproot4.models",
    "uproot4.behaviors",
    "uproot4.const",
    "uproot4.extras",
    "uproot4.version",
    "uproot4.dynamic",
    "uproot4._util",
]


toctree = open("toctree.txt", "w")
toctree.write(""".. toctree::
    :caption: Reference
    :hidden:

""")

def handle_module(modulename, module):
    if not os.path.exists(modulename + ".rst"):
        open(modulename + ".rst", "w").write("""{0}
{1}

.. automodule:: {0}
""".format(modulename, "=" * len(modulename)))
    toctree.write("    " + modulename + "\n")

    if (
        modulename != "uproot4" and
        modulename != "uproot4.dynamic" and
        all(not x.startswith("_") for x in modulename.split("."))
    ):
        def good(obj):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if obj.__module__ == modulename and obj.__name__ != "Report":
                    return True
            return False

        def line_order(pair):
            name, obj = pair
            return inspect.getsourcelines(obj)[1]

        for pair in sorted(inspect.getmembers(module, good), key=line_order):
            name, obj = pair
            if not name.startswith("_"):
                if inspect.isclass(obj):
                    handle_class(modulename + "." + name, obj)
                elif inspect.isfunction(obj):
                    handle_function(modulename + "." + name, obj)


def handle_class(classname, cls):
    if not os.path.exists(classname + ".rst"):
        open(classname + ".rst", "w").write("""{0}
{1}

.. autoclass:: {0}
    :members:
    :inherited-members:
""".format(classname, "=" * len(classname)))
    toctree.write("    " + classname + "\n")


def handle_function(functionname, cls):
    if not os.path.exists(functionname + ".rst"):
        open(functionname + ".rst", "w").write("""{0}
{1}

.. autofunction:: {0}
""".format(functionname, "=" * len(functionname)))
    toctree.write("    " + functionname + "\n")


for modulename in order:
    module = importlib.import_module(modulename)
    handle_module(modulename, module)
    if module.__file__.endswith("__init__.py") and modulename != "uproot4":
        for pair in sorted(inspect.getmembers(module, inspect.ismodule)):
            submodulename, submodule = pair
            handle_module(modulename + "." + submodulename, submodule)
