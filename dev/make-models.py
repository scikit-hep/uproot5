"""
Use this to make src/uproot/models/TH.py and src/uproot/models/TGraph.py.
"""

import numpy as np

import uproot

keys = [
    "th1c",
    "th1d",
    "th1f",
    "th1i",
    "th1s",
    "th2c",
    "th2d",
    "th2f",
    "th2i",
    "th2s",
    "th3c",
    "th3d",
    "th3f",
    "th3i",
    "th3s",
    "tprofile",
    "tprofile2d",
    "tprofile3d",
    "tgraphasymmerrors",
    "tgrapherrors",
    "tgraph",
]

superclasses = [
    # ("TCollection", 3),
    # ("TSeqCollection", 0),
    # ("TList", 5),
    # ("THashList", 0),
    ("TAttAxis", 4),
    ("TAxis", 10),
    # ("TAttMarker", 2),
    # ("TAttFill", 2),
    # ("TAttLine", 2),
    # ("TString", 2),
    # ("TObject", 1),
    # ("TNamed", 1),
    ("TH1", 8),
    ("TH2", 5),
    ("TAtt3D", 1),
    ("TH3", 6),
]


# with uproot.open("example-objects.root") as f:
#     f.file.streamers
#     pairs = []
#     for key in keys:
#         obj = f[key]
#         for pair in f.file.streamer_dependencies(obj.classname, obj.class_version):
#             if pair not in pairs and pair[0].lower() not in keys:
#                 pairs.append(pair)
#     for streamer_name, streamer_version in pairs:
#         print(streamer_name, streamer_version)


with uproot.open("example-objects.root") as f:
    f.file.streamers

    for classname, class_version in superclasses:
        cls = f.file.class_named(classname, class_version)
        print(cls.class_code)
        print(
            f"""
    writable = True

    def _serialize(self, out, header, name):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name)
        raise NotImplementedError("FIXME")
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = {class_version}
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))

class {uproot.model.classname_encode(classname)}(uproot.model.DispatchByVersion):
    \"\"\"
    A :doc:`uproot.model.DispatchByVersion` for ``{classname}``.
    \"\"\"

    known_versions = {{{class_version}: {cls.__name__}}}
"""
        )

    for key in keys:
        obj = f[key]
        print(type(obj).class_code)
        print(
            """
    class_rawstreamers = ("""
        )
        for streamer_name, streamer_version in f.file.streamer_dependencies(
            obj.classname, obj.class_version
        ):
            streamer = f.file.streamer_named(streamer_name, streamer_version)
            inner = streamer.serialize()
            header = np.array(
                [(len(inner) + 18) | uproot.const.kByteCountMask], ">u4"
            ).tobytes()
            preamble = b"\xff\xff\xff\xffTStreamerInfo\x00"
            full = header + preamble + inner + b"\x00"
            print(
                f"""        uproot._writing.RawStreamerInfo(
            None,
            {full},
            {streamer_name!r},
            {streamer_version},
        ),"""
            )
        print(
            f"""    )
    writable = True

    def _serialize(self, out, header, name):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name)
        raise NotImplementedError("FIXME")
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = {obj.class_version}
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))
"""
        )

        print(
            f"""
class {uproot.model.classname_encode(obj.classname)}(uproot.model.DispatchByVersion):
    \"\"\"
    A :doc:`uproot.model.DispatchByVersion` for ``{obj.classname}``.
    \"\"\"

    known_versions = {{{obj.class_version}: {type(obj).__name__}}}
"""
        )

    for classname, _ in superclasses:
        print(
            f"uproot.classes[{classname!r}] = {uproot.model.classname_encode(classname)}"
        )

    for key in keys:
        obj = f[key]
        print(
            f"uproot.classes[{obj.classname!r}] = {uproot.model.classname_encode(obj.classname)}"
        )
