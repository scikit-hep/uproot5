# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

import importlib
import inspect
import pkgutil
import os.path
import sys

import uproot


order = [
    "uproot",
    "uproot.reading",
    "uproot.behaviors",
    "uproot.model",
    "uproot.streamers",
    "uproot.cache",
    "uproot.compression",
    "uproot.deserialization",
    "uproot.source",
    "uproot.interpretation",
    "uproot.containers",
    "uproot.language",
    "uproot.models",
    "uproot.exceptions",
]

toctree = open("uproot.toctree", "w")
toctree.write(
    """.. toctree::
    :caption: Reference
    :hidden:

"""
)
toctree2 = None


def ensure(filename, content):
    overwrite = not os.path.exists(filename)
    if not overwrite:
        overwrite = open(filename, "r").read() != content
    if overwrite:
        open(filename, "w").write(content)
        sys.stderr.write(filename + " (OVERWRITTEN)\n")
    else:
        sys.stderr.write(filename + "\n")


def handle_module(modulename, module):
    content = """{0}
{1}

.. automodule:: {0}
""".format(
        modulename, "=" * len(modulename)
    )
    ensure(modulename + ".rst", content)
    if toctree2 is None:
        toctree.write("    " + modulename + "\n")
    else:
        toctree2.write("    " + modulename + "\n")

    if modulename != "uproot" and all(
        not x.startswith("_") for x in modulename.split(".")
    ):

        def good(obj):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if obj.__module__ == modulename:
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
    def line_order(obj):
        if isinstance(obj, property):
            obj = obj.fget
        return inspect.getsourcelines(obj)[1]

    methods = {}
    mro = list(cls.__mro__)

    if hasattr(uproot, cls.__name__):
        title = "uproot." + cls.__name__
        upfront = True
    else:
        title = classname
        upfront = False

    for index, basecls in enumerate(mro):
        if basecls.__module__.startswith("uproot."):

            def good(obj):
                if inspect.ismethod(obj) or inspect.isfunction(obj):
                    module, name = obj.__module__, obj.__name__
                elif isinstance(obj, property):
                    module, name = obj.fget.__module__, obj.fget.__name__
                else:
                    module, name = "", ""
                if module.startswith("uproot."):
                    if index + 1 >= len(mro) or obj is not getattr(
                        mro[index + 1], name, None
                    ):
                        return True
                return False

            for name, obj in inspect.getmembers(basecls, good):
                if name in basecls.__dict__ and not name.startswith("_"):
                    fill = []
                    fill.append(".. _" + classname + "." + name + ":" + "\n")
                    fill.append(name)
                    fill.append("-" * len(fill[-1]))
                    fill.append("")
                    if basecls is not cls:
                        fill.append(
                            "Inherited from :doc:`{0}`.".format(
                                basecls.__module__ + "." + basecls.__name__
                            )
                        )
                        fill.append("")
                    if isinstance(obj, property):
                        fill.append(".. autoattribute:: " + classname + "." + name)
                    elif callable(obj):
                        fill.append(".. automethod:: " + classname + "." + name)
                    else:
                        fill.append(".. autoattribute:: " + classname + "." + name)
                    fill.append("")
                    methods[name] = (index, line_order(obj), "\n".join(fill))

    def prettymro(c):
        fullname = c.__module__ + "." + c.__name__
        if c.__module__.startswith("uproot."):
            return "#. :doc:`" + fullname + "`"
        else:
            return "#. ``" + fullname + "``"

    fullfilename = importlib.import_module(cls.__module__).__file__
    shortfilename = fullfilename[fullfilename.rindex("uproot/"):]
    link = "`{0} <https://github.com/scikit-hep/uproot4/blob/{1}/{2}>`__".format(
        cls.__module__, uproot.__version__, shortfilename
    )
    linelink = "`line {0} <https://github.com/scikit-hep/uproot4/blob/{1}/{2}#L{0}>`__".format(
        inspect.getsourcelines(cls)[1], uproot.__version__, shortfilename
    )

    inheritance_header = ""
    inheritance_footer = ""
    inheritance_sep = ""
    inheritance = [prettymro(c) for c in cls.__mro__[1:] if c is not object]
    if len(inheritance) > 0:
        longest_cell = max(max(len(x) for x in inheritance), 22)
        inheritance_header = """.. table::
    :class: note

    +-{0}-+
    | **Inheritance order:** {1}|
    +={2}=+
    | """.format("-" * longest_cell, " " * (longest_cell - 22), "=" * longest_cell)
        inheritance_footer = """ |
    +-{0}-+""".format("-" * longest_cell)
        inheritance = [x + " " * (longest_cell - len(x)) for x in inheritance]
        inheritance_sep = """ |
    | """

    content = """{0}
{1}

Defined in {2} on {3}.

{4}

.. autoclass:: {5}

{6}
""".format(
        title,
        "=" * len(title),
        link,
        linelink,
        inheritance_header + inheritance_sep.join(inheritance) + inheritance_footer,
        classname,
        "\n".join([text for index, line, text in sorted(methods.values())]),
    )

    ensure(classname + ".rst", content)
    if upfront or toctree2 is None:
        toctree.write("    " + classname + "\n")
        toctree2.write("    " + classname + " <" + classname + ">\n")
    else:
        toctree2.write("    " + classname + "\n")


def handle_function(functionname, cls):
    if hasattr(uproot, cls.__name__):
        title = "uproot." + cls.__name__
        upfront = True
    else:
        title = functionname
        upfront = False

    fullfilename = importlib.import_module(cls.__module__).__file__
    shortfilename = fullfilename[fullfilename.rindex("uproot/"):]
    link = "`{0} <https://github.com/scikit-hep/uproot4/blob/{1}/{2}>`__".format(
        cls.__module__, uproot.__version__, shortfilename
    )
    linelink = "`line {0} <https://github.com/scikit-hep/uproot4/blob/{1}/{2}#L{0}>`__".format(
        inspect.getsourcelines(cls)[1], uproot.__version__, shortfilename
    )

    content = """{0}
{1}

Defined in {2} on {3}.

.. autofunction:: {4}
""".format(title, "=" * len(title), link, linelink, functionname)
    ensure(functionname + ".rst", content)
    if upfront or toctree2 is None:
        toctree.write("    " + functionname + "\n")
        toctree2.write("    " + functionname + " <" + functionname + ">\n")
    else:
        toctree2.write("    " + functionname + "\n")


for modulename in order:
    module = importlib.import_module(modulename)

    if modulename != "uproot":
        toctree2 = open(modulename + ".toctree", "w")
        toctree2.write(
            """.. toctree::
    :caption: {0}
    :hidden:

""".format(modulename.replace("uproot.", ""))
        )

    handle_module(modulename, module)
    if module.__file__.endswith("__init__.py") and modulename != "uproot":
        for submodulename in sorted(
            [
                modulename + "." + name
                for loader, name, is_pkg in pkgutil.walk_packages(module.__path__)
            ]
        ):
            submodule = importlib.import_module(submodulename)
            handle_module(submodulename, submodule)

toctree.close()
toctree2.close()
