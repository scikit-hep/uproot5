# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines an :doc:`uproot4.interpretation.Interpretation` and temporary array for
object data (Python objects or Awkward Array data structures).

The :doc:`uproot4.interpretation.objects.AsObjects` describes fully generic
objects using a :doc:`uproot4.interpretation.model.Model` (or
:doc:`uproot4.interpretation.containers`). These objects require a
non-vectorized loop to deserialize.

The :doc:`uproot4.interpretation.objects.AsStridedObjects` describes fixed-width
objects that can be described as a ``numpy.dtype``. These objects can be
interpreted as a single, vectorized cast, and are therefore much faster to
deserialize.

The :doc:`uproot4.interpretation.object.ObjectArray` and
:doc:`uproot4.interpretation.object.StridedObjectArray` classes only hold data
while an array is being built from ``TBaskets``. Its final form is determined
by :doc:`uproot4.interpretation.library`.
"""

from __future__ import absolute_import

import numpy

import uproot4.interpretation
import uproot4.interpretation.strings
import uproot4.interpretation.jagged
import uproot4.interpretation.numerical
import uproot4.containers
import uproot4.model
import uproot4.source.chunk
import uproot4.source.cursor
import uproot4._util


def awkward_can_optimize(interpretation, form):
    """
    If True, the Awkward Array library can convert data of a given
    :doc:`uproot4.interpretation.Interpretation` and ``ak.forms.Form`` into
    arrays without resorting to ``ak.from_iter`` (i.e. rapidly).

    If ``awkward1._connect._uproot`` cannot be imported, this function always
    returns False.
    """
    try:
        import awkward1._connect._uproot
    except ImportError:
        return False
    else:
        return awkward1._connect._uproot.can_optimize(interpretation, form)


class AsObjects(uproot4.interpretation.Interpretation):
    """
    Args:
        model (:doc:`uproot4.model.Model` or :doc:`uproot4.containers.Container`): The
            full Uproot deserialization model for the data.
        branch (None or :doc:`uproot4.behavior.TBranch.TBranch`): The ``TBranch``
            from which the data are drawn.

    Integerpretation for arrays of any kind of data that might reside in a
    ROOT ``TTree``. This interpretation prescribes the full (slow)
    deserialization process.

    :doc:`uproot4.interpretation.objects.AsObjects.simplify` attempts to
    replace this interpretation with a faster-to-read equivalent, but not all
    data types can be simplified.
    """

    def __init__(self, model, branch=None):
        self._model = model
        self._branch = branch

    @property
    def model(self):
        """
        The full Uproot deserialization model for the data
        (:doc:`uproot4.model.Model` or :doc:`uproot4.containers.Container`).
        """
        return self._model

    @property
    def branch(self):
        """
        The ``TBranch`` from which the data are drawn. May be None.
        """
        return self._branch

    def __repr__(self):
        if isinstance(self._model, type):
            model = self._model.__name__
        else:
            model = repr(self._model)
        return "AsObjects({0})".format(model)

    def __eq__(self, other):
        return isinstance(other, AsObjects) and self._model == other._model

    @property
    def numpy_dtype(self):
        return numpy.dtype(numpy.object)

    @property
    def cache_key(self):
        content_key = uproot4.containers._content_cache_key(self._model)
        return "{0}({1})".format(type(self).__name__, content_key)

    @property
    def typename(self):
        if isinstance(self._model, uproot4.containers.AsContainer):
            return self._model.typename
        else:
            return uproot4.model.classname_decode(self._model.__name__)[0]

    def awkward_form(self, file, index_format="i64", header=False, tobject_header=True):
        if isinstance(self._model, type):
            return self._model.awkward_form(
                self._branch.file, index_format, header, tobject_header
            )
        else:
            return self._model.awkward_form(
                self._branch.file, index_format, header, tobject_header
            )

    def basket_array(
        self, data, byte_offsets, basket, branch, context, cursor_offset, library
    ):
        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
            library=library,
        )
        assert basket.byte_offsets is not None

        output = None
        if isinstance(library, uproot4.interpretation.library.Awkward):
            form = self.awkward_form(branch.file, index_format="i64")

            if awkward_can_optimize(self, form):
                import awkward1._connect._uproot

                extra = {
                    "interpretation": self,
                    "basket": basket,
                    "branch": branch,
                    "context": context,
                    "cursor_offset": cursor_offset,
                }
                output = awkward1._connect._uproot.basket_array(
                    form, data, byte_offsets, extra
                )

        if output is None:
            output = ObjectArray(
                self._model, branch, context, byte_offsets, data, cursor_offset
            ).to_numpy()

        self.hook_after_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            output=output,
            cursor_offset=cursor_offset,
            library=library,
        )

        return output

    def final_array(
        self, basket_arrays, entry_start, entry_stop, entry_offsets, library, branch
    ):
        self.hook_before_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
        )
        trimmed = []
        start = entry_offsets[0]
        for basket_num, stop in enumerate(entry_offsets[1:]):
            if start <= entry_start and entry_stop <= stop:
                local_start = entry_start - start
                local_stop = entry_stop - start
                trimmed.append(basket_arrays[basket_num][local_start:local_stop])

            elif start <= entry_start < stop:
                local_start = entry_start - start
                local_stop = stop - start
                trimmed.append(basket_arrays[basket_num][local_start:local_stop])

            elif start <= entry_stop <= stop:
                local_start = 0
                local_stop = entry_stop - start
                trimmed.append(basket_arrays[basket_num][local_start:local_stop])

            elif entry_start < stop and start <= entry_stop:
                trimmed.append(basket_arrays[basket_num])

            start = stop

        if all(type(x).__module__.startswith("awkward1") for x in basket_arrays.values()):
            assert isinstance(library, uproot4.interpretation.library.Awkward)
            awkward1 = library.imported
            output = awkward1.concatenate(trimmed, mergebool=False, highlevel=False)
        else:
            output = numpy.concatenate(trimmed)

        self.hook_before_library_finalize(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
        )

        output = library.finalize(output, branch, self, entry_start, entry_stop)

        self.hook_after_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
        )

        return output

    def simplify(self):
        """
        Attempts to replace this :doc:`uproot4.interpretation.objects.AsObjects`
        with an interpretation that can be executed more quickly.

        If there isn't a simpler interpretation, then this method returns
        ``self``.
        """
        if self._branch is not None:
            try:
                return self._model.strided_interpretation(
                    self._branch.file,
                    header=False,
                    tobject_header=True,
                    original=self._model,
                )
            except CannotBeStrided:
                pass

        if isinstance(self._model, uproot4.containers.AsString):
            header_bytes = 0
            if self._model.header:
                header_bytes = 6
            return uproot4.interpretation.strings.AsStrings(
                header_bytes,
                self._model.length_bytes,
                self._model.typename,
                original=self._model,
            )

        if isinstance(
            self._model, (uproot4.containers.AsArray, uproot4.containers.AsVector),
        ):
            header_bytes = 0
            if (
                isinstance(self._model, uproot4.containers.AsArray)
                and self._model.speedbump
            ):
                header_bytes += 1
            if (
                isinstance(self._model, uproot4.containers.AsVector)
                and self._model.header
            ):
                header_bytes += 10

            if isinstance(self._model.values, numpy.dtype):
                content = uproot4.interpretation.numerical.AsDtype(self._model.values)
                return uproot4.interpretation.jagged.AsJagged(
                    content, header_bytes, self._model.typename, original=self._model
                )

            if self._branch is not None:
                try:
                    content = self._model.values.strided_interpretation(
                        self._branch.file,
                        header=False,
                        tobject_header=True,
                        original=self._model.values,
                    )
                    return uproot4.interpretation.jagged.AsJagged(
                        content,
                        header_bytes,
                        self._model.typename,
                        original=self._model,
                    )
                except CannotBeStrided:
                    pass

        return self


def _unravel_members(members):
    out = []
    for name, member in members:
        if isinstance(member, AsStridedObjects):
            for n, m in _unravel_members(member.members):
                out.append((name + "/" + n, m))
        else:
            out.append((name, member))
    return out


def _strided_awkward_form(
    awkward1, classname, members, file, index_format, header, tobject_header
):
    contents = {}
    for name, member in members:
        if isinstance(member, AsStridedObjects):
            cname = uproot4.model.classname_decode(member._model.__name__)[0]
            contents[name] = _strided_awkward_form(
                awkward1,
                cname,
                member._members,
                file,
                index_format,
                header,
                tobject_header,
            )
        else:
            contents[name] = uproot4._util.awkward_form(
                member, file, index_format, header, tobject_header
            )
    return awkward1.forms.RecordForm(contents, parameters={"__record__": classname})


class AsStridedObjects(uproot4.interpretation.numerical.AsDtype):
    """
    Args:
        model (:doc:`uproot4.model.Model` or :doc:`uproot4.containers.Container`): The
            full Uproot deserialization model for the data.
        members (list of (str, :doc:`uproot4.interpretation.Interpretation`) tuples): The
            name and fixed-width interpretation for each member of the objects.
        original (None, :doc:`uproot4.model.Model`, or :doc:`uproot4.containers.Container`): If
            this interpretation is derived from
            :doc:`uproot4.interpretation.objects.AsObjects.simplify`, this is a
            reminder of the original
            :doc:`uproot4.interpretation.objects.AsObjects.model`.

    Interpretation for an array (possibly
    :doc:`uproot4.interpretation.jagged.AsJagged`) of fixed-size objects. Since
    the objects have a fixed number of fields with a fixed number of bytes each,
    the whole array (or :doc:`uproot4.interpretation.jagged.AsJagged.content`)
    can be interpreted in one vectorized array-cast. Therefore, this
    interpretation is faster than :doc:`uproot4.interpretation.objects.AsObjects`
    *when it is possible*.

    Unlike :doc:`uproot4.interpretation.numerical.AsDtype` with a
    `structured array <https://numpy.org/doc/stable/user/basics.rec.html>`__,
    the objects in the final array have the methods required by its ``model``.
    If the ``library`` is :doc:`uproot4.interpretation.library.NumPy`, these
    are instantiated as Python objects (slow); if
    :doc:`uproot4.interpretation.library.Awkward`, they are behaviors passed to
    the Awkward Array's local
    `behavior <https://awkward-array.readthedocs.io/en/latest/ak.behavior.html>`__.
    """

    def __init__(self, model, members, original=None):
        self._model = model
        self._members = members
        self._original = original
        super(AsStridedObjects, self).__init__(_unravel_members(members))

    @property
    def model(self):
        """
        The full Uproot deserialization model for the data
        (:doc:`uproot4.model.Model` or :doc:`uproot4.containers.Container`).
        """
        return self._model

    @property
    def members(self):
        """
        The name (str) and fixed-width
        :doc:`uproot4.interpretation.Interpretation` for each member of the
        objects as a list of 2-tuple pairs.
        """
        return self._members

    @property
    def original(self):
        """
        If not None, this was the original
        :doc:`uproot4.interpretation.objects.AsObjects.model` from an
        :doc:`uproot4.interpretation.objects.AsObjects` that was simplified
        into this :doc:`uproot4.interpretation.objects.AsStridedObjects`.
        """
        return self._original

    def __repr__(self):
        return "AsStridedObjects({0})".format(self._model.__name__)

    def __eq__(self, other):
        return isinstance(other, AsStridedObjects) and self._model == other._model

    @property
    def numpy_dtype(self):
        return numpy.dtype(numpy.object)

    def awkward_form(self, file, index_format="i64", header=False, tobject_header=True):
        import awkward1

        cname = uproot4.model.classname_decode(self._model.__name__)[0]
        return _strided_awkward_form(
            awkward1, cname, self._members, file, index_format, header, tobject_header
        )

    @property
    def cache_key(self):
        return "{0}({1})".format(type(self).__name__, self._model.__name__)

    @property
    def typename(self):
        return uproot4.model.classname_decode(self._model.__name__)[0]

    def _wrap_almost_finalized(self, array):
        return StridedObjectArray(self, array)


class CannotBeStrided(Exception):
    """
    Exception used to stop recursion over
    :doc:`uproot4.model.Model.strided_interpretation` and
    :doc:`uproot4.containers.AsContainer.strided_interpretation` as soon as a
    non-conforming type is found.
    """

    pass


class CannotBeAwkward(Exception):
    """
    Exception used to stop recursion over
    :doc:`uproot4.interpretation.Interpretation.awkward_form`,
    :doc:`uproot4.model.Model.awkward_form` and
    :doc:`uproot4.containers.AsContainer.awkward_form` as soon as a
    non-conforming type is found.
    """

    def __init__(self, because):
        self.because = because


class ObjectArray(object):
    """
    Args:
        model (:doc:`uproot4.model.Model` or :doc:`uproot4.containers.Container`): The
            full Uproot deserialization model for the data.
        branch (:doc:`uproot4.behavior.TBranch.TBranch`): The ``TBranch`` from
            which the data are drawn.
        context (dict): Auxiliary data used in deserialization.
        byte_offsets (array of ``numpy.int32``): Index where each entry of the
            ``byte_content`` starts and stops.
        byte_content (array of ``numpy.uint8``): Raw but uncompressed data,
            directly from
            :doc:`uproot4.interpretation.Interpretation.basket_array`.
        cursor_offset (int): Correction to the integer keys used in
            :doc:`uproot4.source.cursor.Cursor.refs` for objects deserialized
            by reference (:doc:`uproot4.deserialization.read_object_any`).

    Temporary array filled by
    :doc:`uproot4.interpretation.objects.AsObjects.basket_array`, which will be
    turned into a NumPy, Awkward, or other array, depending on the specified
    :doc:`uproot4.interpretation.library.Library`.
    """

    def __init__(
        self, model, branch, context, byte_offsets, byte_content, cursor_offset
    ):
        self._model = model
        self._branch = branch
        self._context = context
        self._byte_offsets = byte_offsets
        self._byte_content = byte_content
        self._cursor_offset = cursor_offset
        self._detached_file = self._branch.file.detached

    def __repr__(self):
        return "ObjectArray({0}, {1}, {2}, {3}, {4}, {5})".format(
            self._model,
            self._branch,
            self._context,
            self._byte_offsets,
            self._byte_content,
            self._cursor_offset,
        )

    @property
    def model(self):
        """
        The full Uproot deserialization model for the data
        (:doc:`uproot4.model.Model` or :doc:`uproot4.containers.Container`).
        """
        return self._model

    @property
    def branch(self):
        """
        The ``TBranch`` from which the data are drawn.
        """
        return self._branch

    @property
    def context(self):
        """
        Auxiliary data used in deserialization (dict).
        """
        return self._context

    @property
    def byte_offsets(self):
        """
        Index where each entry of the ``byte_content`` starts and stops.
        """
        return self._byte_offsets

    @property
    def byte_content(self):
        """
        Raw but uncompressed data, directly from
        :doc:`uproot4.interpretation.Interpretation.basket_array`.
        """
        return self._byte_content

    @property
    def cursor_offset(self):
        """
        Correction to the integer keys used in
        :doc:`uproot4.source.cursor.Cursor.refs` for objects deserialized by
        reference (:doc:`uproot4.deserialization.read_object_any`).
        """
        return self._cursor_offset

    def to_numpy(self):
        """
        Convert this ObjectArray into a NumPy ``dtype="O"`` (object) array.
        """
        output = numpy.empty(len(self), dtype=numpy.dtype(numpy.object))
        for i in range(len(self)):
            output[i] = self[i]
        return output

    def __len__(self):
        return len(self._byte_offsets) - 1

    def __getitem__(self, where):
        if uproot4._util.isint(where):
            byte_start = self._byte_offsets[where]
            byte_stop = self._byte_offsets[where + 1]
            data = self._byte_content[byte_start:byte_stop]
            chunk = uproot4.source.chunk.Chunk.wrap(self._branch.file.source, data)
            cursor = uproot4.source.cursor.Cursor(
                0, origin=-(byte_start + self._cursor_offset)
            )
            return self._model.read(
                chunk,
                cursor,
                self._context,
                self._branch.file,
                self._detached_file,
                self._branch,
            )

        elif isinstance(where, slice):
            return ObjectArray(
                self._model,
                self._branch,
                self._context,
                self._byte_offsets[where],
                self._byte_content,
                self._cursor_offset,
            )

        else:
            raise NotImplementedError(repr(where))

    def __iter__(self):
        source = self._branch.file.source
        context = self._context
        file = self._branch.file
        selffile = self._detached_file
        branch = self._branch
        byte_start = self._byte_offsets[0]
        for byte_stop in self._byte_offsets[1:]:
            data = self._byte_content[byte_start:byte_stop]
            chunk = uproot4.source.chunk.Chunk.wrap(source, data)
            cursor = uproot4.source.cursor.Cursor(0, origin=-self._cursor_offset)
            yield self._model.read(chunk, cursor, context, file, selffile, branch)
            byte_start = byte_stop


def _strided_object(path, interpretation, data):
    out = interpretation._model.empty()
    for name, member in interpretation._members:
        p = name
        if len(path) != 0:
            p = path + "/" + name
        if isinstance(member, AsStridedObjects):
            out._members[name] = _strided_object(p, member, data)
        else:
            out._members[name] = data[p]
    return out


class StridedObjectArray(object):
    """
    Args:
        interpretation (:doc:`uproot4.interpretation.objects.AsStridedObjects`): The
            interpretation that produced this array.
        array (array): Underlying array object, which may be NumPy or another
            temporary array.

    Temporary array filled by
    :doc:`uproot4.interpretation.objects.AsStridedObjects.basket_array`, which
    will be turned into a NumPy, Awkward, or other array, depending on the
    specified :doc:`uproot4.interpretation.library.Library`.
    """

    def __init__(self, interpretation, array):
        self._interpretation = interpretation
        self._array = array

    @property
    def interpretation(self):
        """
        The interpretation that produced this array.
        """
        return self._interpretation

    @property
    def array(self):
        """
        Underlying array object, which may be NumPy or another temporary array.
        """
        return self._array

    def __repr__(self):
        return "StridedObjectArray({0}, {1})".format(self._interpretation, self._array)

    def __len__(self):
        return len(self._array)

    def __getitem__(self, where):
        if uproot4._util.isint(where):
            return _strided_object("", self._interpretation, self._array[where])

        else:
            return StridedObjectArray(self._interpretation, self._array[where])

    def __iter__(self):
        interpretation = self._interpretation
        for x in self._array:
            yield _strided_object("", interpretation, x)
