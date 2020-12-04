---
name: Bug report
about: Something seems to be broken…
title: ''
labels: bug (unverified)
assignees: ''

---

Thank you for submitting an issue! (I know that it takes time and effort to do so.)

Note that we'll be closing the issue as soon as a solution is proposed. This is not meant to be unfriendly; it's for our own bookkeeping. If you think the first answer/solution is unsatisfactory, please do continue the thread and we'll reopen it or otherwise address it.

Please attach a small ROOT file that reproduces the issue! If small and public, you can drag-and-drop it into the issue—rename the extension to "txt" so that GitHub allows it. If large, you can put it on some large-file service (e.g. Dropbox). In general, we can't access XRootD URLs (most are not public).

Include the version number (and update if necessary, if you're not using the [latest version](https://pypi.org/project/uproot/)).

```python
>>> import uproot
>>> uproot.__version__
```
