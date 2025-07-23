from __future__ import annotations

import tiny_reader

import uproot

tiny_reader.register()

f = uproot.open("tiny_reader/gen-data/example.root")
f["my_tree"].show()

print("==================================")
arr = f["my_tree"].arrays()
arr.show(all=True)
