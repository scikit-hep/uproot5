Uproot 3 → 4 cheat-sheet
========================

The Uproot 3 → 4 transition was primarily motivated by Awkward Array 0 → 1. The interface of Awkward Array significantly changed and Awkward Arrays are output by Uproot functions, so this difference would be visible to you as a user of Uproot. 







The Uproot 3 → 4 transition was primarily motivated by Awkward Array 0 → 1. The interface of Awkward Array significantly changed and Awkward Array is used throughout Uproot. But this transition also gave us an opportunity to fix a few long-standing problems in Uproot itself:

* Uproot 3 does not interpret C++ strings according to any encoding, but that means that users have to deal with ``bytes`` in Python 3. Uproot 4 deals exclusively in ``str`` (interpreting all C++ strings as UTF-8 with surrogate-escapes).

* Uproot 3 returns NumPy arrays or Awkward Arrays, depending on the type of the data in the ROOT file. Uproot 4 has a ``library`` parameter to specify what kind of ouput you want.

* Uproot 3 has special-purpose functions for Pandas output. In Uproot 4, you get Pandas Series and DataFrames by asking for ``library="pd"``.

* Uproot 4 clearly distinguishes between directories and files.

* Uproot 4 allows "TTree" methods to be used on TBranches that contain TBranches.

* Uproot 4 allows ``"dir/subdir/branch/subbranch"`` to pass through directories and branches.

* Uproot 4 fully supports Python's ``with`` statement. (Uproot 3 couldn't close some types of files!)

* Uproot 4 has several fundamental architectural improvements:

  * Batches requests for remote sources (HTTP multipart GET and XRootD vector-read).

  * Requests exactly the byte ranges of interest—doesn't round to equal-sized chunks.

  * Doesn't read a file's "streamers" (TStreamerInfo) unless new classes are encountered.

  * Classes are organized by version number to dispatch first by version, rather than "correcting" mistaken versions.

  * Much better debugging output when something can't be deserialized.

  * ROOT objects that don't initiate file-reading (e.g. TTree) are pickleable.

  * Code is distributed into smaller submodules: better organization/maintainability.

  * Depends only on NumPy—the rest is pay-as-you-go (in terms of installation requirements).

Except for the architectural improvements, these changes are visible to users. The architectural improvements motivated a fresh rewrite, much like 

.. image:: https://raw.githubusercontent.com/scikit-hep/uproot4/jpivarski/write-cheat-sheet/docs-img/diagrams/uproot-awkward-timeline.png
  :width: 100%

