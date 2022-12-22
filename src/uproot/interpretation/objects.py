# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines an :doc:`uproot.interpretation.Interpretation` and
temporary array for object data (Python objects or Awkward Array data structures).

The :doc:`uproot.interpretation.objects.AsObjects` describes fully generic
objects using a :doc:`uproot.model.Model` (or a :doc:`uproot.containers.AsContainer`).
These objects require a non-vectorized loop to deserialize.

The :doc:`uproot.interpretation.objects.AsStridedObjects` describes fixed-width
objects that can be described as a ``numpy.dtype``. These objects can be
interpreted as a single, vectorized cast, and are therefore much faster to
deserialize.

The :doc:`uproot.interpretation.objects.ObjectArray` and
:doc:`uproot.interpretation.objects.StridedObjectArray` classes only hold data
while an array is being built from ``TBaskets``. Its final form is determined
by the :doc:`uproot.interpretation.library.Library`.
"""

import contextlib
import threading

import numpy

import uproot
import uproot._awkward_forth


class AsObjects(uproot.interpretation.Interpretation):
    """
    Args:
        model (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`): The
            full Uproot deserialization model for the data.
        branch (None or :doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch``
            from which the data are drawn.

    Integerpretation for arrays of any kind of data that might reside in a
    ROOT ``TTree``. This interpretation prescribes the full (slow)
    deserialization process.

    :ref:`uproot.interpretation.objects.AsObjects.simplify` attempts to
    replace this interpretation with a faster-to-read equivalent, but not all
    data types can be simplified.
    """

    def __init__(self, model, branch=None):
        self._model = model
        self._branch = branch
        self._forth = True
        self._forth_lock = threading.Lock()
        self._forth_form_keys = None
        self._complete_forth_code = None
        self._form = None

    @property
    def model(self):
        """
        The full Uproot deserialization model for the data
        (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`).
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
        return f"AsObjects({model})"

    def __eq__(self, other):
        return isinstance(other, AsObjects) and self._model == other._model

    @property
    def numpy_dtype(self):
        return numpy.dtype(object)

    @property
    def cache_key(self):
        content_key = uproot.containers._content_cache_key(self._model)
        return f"{type(self).__name__}({content_key})"

    @property
    def typename(self):
        if isinstance(self._model, uproot.containers.AsContainer):
            return self._model.typename
        else:
            return uproot.model.classname_decode(self._model.__name__)[0]

    def awkward_form(
        self,
        file,
        context=None,
        index_format="i64",
        header=False,
        tobject_header=False,
        breadcrumbs=(),
    ):
        context = self._make_context(
            context, index_format, header, tobject_header, breadcrumbs
        )
        if isinstance(self._model, type):
            return self._model.awkward_form(self._branch.file, context)
        else:
            return self._model.awkward_form(self._branch.file, context)

    def basket_array(
        self,
        data,
        byte_offsets,
        basket,
        branch,
        context,
        cursor_offset,
        library,
        options,
    ):
        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
            library=library,
            options=options,
        )
        assert basket.byte_offsets is not None

        if self._forth and isinstance(library, uproot.interpretation.library.Awkward):
            output = self.basket_array_forth(
                data,
                byte_offsets,
                basket,
                branch,
                context,
                cursor_offset,
                library,
                options,
            )
        else:
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
            options=options,
        )

        return output

    def basket_array_forth(
        self,
        data,
        byte_offsets,
        basket,
        branch,
        context,
        cursor_offset,
        library,
        options,
    ):
        awkward = uproot.extras.awkward()
        import awkward.forth

        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
            library=library,
            options=options,
        )
        assert basket.byte_offsets is not None

        if not hasattr(context["forth"], "vm"):
            # context["forth"] is a threading.local()
            context["forth"].vm = None
            context["forth"].gen = uproot._awkward_forth.ForthGenerator()

        if self._complete_forth_code is None:
            with self._forth_lock:
                # all threads have to wait until complete_forth_code is ready

                if self._complete_forth_code is None:
                    # another thread didn't make it while this thread waited
                    # this thread tries to make it now
                    output = self._discover_forth(
                        data, byte_offsets, branch, context, cursor_offset
                    )

                    if output is not None:
                        # Forth discovery was unsuccessful; return Python-derived
                        # output and maybe another basket will be more fruitful
                        self.hook_after_basket_array(
                            data=data,
                            byte_offsets=byte_offsets,
                            basket=basket,
                            branch=branch,
                            context=context,
                            output=output,
                            cursor_offset=cursor_offset,
                            library=library,
                            options=options,
                        )
                        return output

        # if we didn't return already, either this thread generated the
        # complete_forth_code or another did
        assert self._complete_forth_code is not None

        if context["forth"].vm is None:
            try:
                context["forth"].vm = awkward.forth.ForthMachine64(
                    self._complete_forth_code
                )
            except Exception as err:
                raise type(err)(
                    str(err)
                    + "\n\nForth code generated for this data type:\n\n"
                    + self._complete_forth_code
                ) from err

        # get data using Forth
        byte_start = byte_offsets[0]
        byte_stop = byte_offsets[-1]
        temp_data = data[byte_start:byte_stop]
        context["forth"].vm.begin(
            {
                "stream": numpy.array(temp_data),
                "byteoffsets": numpy.array(byte_offsets[:-1]),
                "bytestops": numpy.array(byte_offsets[1:]),
            }
        )
        context["forth"].vm.stack_push(len(byte_offsets) - 1)
        context["forth"].vm.resume()
        container = {}
        container = context["forth"].vm.outputs
        output = awkward.from_buffers(self._form, len(byte_offsets) - 1, container)

        self.hook_after_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            output=output,
            cursor_offset=cursor_offset,
            library=library,
            options=options,
        )

        return output

    def _discover_forth(self, data, byte_offsets, branch, context, cursor_offset):
        output = numpy.empty(len(byte_offsets) - 1, dtype=numpy.dtype(object))
        context["cancel_forth"] = False
        for i in range(len(byte_offsets) - 1):
            byte_start = byte_offsets[i]
            byte_stop = byte_offsets[i + 1]
            temp_data = data[byte_start:byte_stop]
            chunk = uproot.source.chunk.Chunk.wrap(branch.file.source, temp_data)
            cursor = uproot.source.cursor.Cursor(
                0, origin=-(byte_start + cursor_offset)
            )
            if "forth" in context.keys():
                context["forth"].gen.var_set = False
            output[i] = self._model.read(
                chunk,
                cursor,
                context,
                branch.file,
                branch.file.detached,
                branch,
            )
            if context["cancel_forth"] and "forth" in context.keys():
                del context["forth"]
            if "forth" in context.keys():
                context["forth"].gen.awkward_model = context["forth"].gen.top_node
                if not context["forth"].gen.var_set:
                    context["forth"].prereaddone = True
                    self._assemble_forth(
                        context["forth"].gen, context["forth"].gen.top_node["content"]
                    )
                    self._complete_forth_code = f"""input stream
    input byteoffsets
    input bytestops
    {"".join(context["forth"].gen.final_header)}
    {"".join(context["forth"].gen.final_init)}
    0 do
    byteoffsets I-> stack
    stream seek
    {"".join(context["forth"].gen.final_code)}
    loop
    """
                    self._forth_form_keys = tuple(context["forth"].gen.form_keys)
                    self._form = context["forth"].gen.top_form
                    return None  # we should re-read all the data with Forth
        return output  # Forth-generation was unsuccessful: this is Python output

    def _assemble_forth(self, forth_obj, awkward_model):
        forth_obj.add_to_header(awkward_model["header_code"])
        forth_obj.add_to_init(awkward_model["init_code"])
        forth_obj.add_to_final(awkward_model["pre_code"])
        if "content" in awkward_model.keys():
            temp_content = awkward_model["content"]
            if isinstance(temp_content, list):
                for elem in temp_content:
                    self._assemble_forth(forth_obj, elem)
            elif isinstance(temp_content, dict):
                self._assemble_forth(forth_obj, awkward_model["content"])
            else:
                pass
        forth_obj.add_to_final(awkward_model["post_code"])

    def final_array(
        self,
        basket_arrays,
        entry_start,
        entry_stop,
        entry_offsets,
        library,
        branch,
        options,
    ):
        self.hook_before_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            options=options,
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

        if len(basket_arrays) == 0:
            output = numpy.array([], dtype=self.numpy_dtype)
        elif all(
            uproot._util.from_module(x, "awkward") for x in basket_arrays.values()
        ):
            assert isinstance(library, uproot.interpretation.library.Awkward)
            awkward = library.imported
            output = awkward.concatenate(trimmed, mergebool=False, highlevel=False)
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

        output = library.finalize(
            output, branch, self, entry_start, entry_stop, options
        )

        self.hook_after_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
            options=options,
        )

        return output

    def simplify(self):
        """
        Attempts to replace this :doc:`uproot.interpretation.objects.AsObjects`
        with an interpretation that can be executed more quickly.

        If there isn't a simpler interpretation, then this method returns
        ``self``.
        """
        if self._branch is not None:
            with contextlib.suppress(CannotBeStrided):
                return self._model.strided_interpretation(
                    self._branch.file,
                    header=False,
                    tobject_header=True,
                    breadcrumbs=(),
                    original=self._model,
                )

        if isinstance(self._model, uproot.containers.AsString):
            header_bytes = 0
            if self._model.header:
                header_bytes = 6
            return uproot.interpretation.strings.AsStrings(
                header_bytes,
                self._model.length_bytes,
                self._model.typename,
                original=self._model,
            )

        if isinstance(
            self._model,
            (
                uproot.containers.AsArray,
                uproot.containers.AsRVec,
                uproot.containers.AsVector,
            ),
        ):
            header_bytes = 0
            if (
                isinstance(self._model, uproot.containers.AsArray)
                and self._model.speedbump
            ):
                header_bytes += 1
            if (
                isinstance(
                    self._model, (uproot.containers.AsRVec, uproot.containers.AsVector)
                )
                and self._model.header
            ):
                header_bytes += 10

            if isinstance(self._model.values, numpy.dtype):
                content = uproot.interpretation.numerical.AsDtype(self._model.values)
                return uproot.interpretation.jagged.AsJagged(
                    content, header_bytes, self._model.typename, original=self._model
                )

            if self._branch is not None:
                with contextlib.suppress(CannotBeStrided):
                    content = self._model.values.strided_interpretation(
                        self._branch.file,
                        header=False,
                        tobject_header=True,
                        breadcrumbs=(),
                        original=self._model.values,
                    )
                    if (
                        isinstance(self._model, uproot.containers.AsArray)
                        and self._model.inner_shape
                    ):
                        if not isinstance(
                            content, uproot.interpretation.numerical.AsDtype
                        ):
                            raise RuntimeError("I shouldn't be here")
                        content.reshape(self._model.inner_shape)
                    return uproot.interpretation.jagged.AsJagged(
                        content,
                        header_bytes,
                        self._model.typename,
                        original=self._model,
                    )

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


def _strided_awkward_form(awkward, classname, members, file, context):
    contents = {}
    for name, member in members:
        if not context["header"] and name in ("@num_bytes", "@instance_version"):
            pass
        elif not context["tobject_header"] and name in (
            "@num_bytes",
            "@instance_version",
            "@fUniqueID",
            "@fBits",
            "@pidf",
        ):
            pass
        else:
            if isinstance(member, AsStridedObjects):
                cname = uproot.model.classname_decode(member._model.__name__)[0]
                contents[name] = _strided_awkward_form(
                    awkward, cname, member._members, file, context
                )
            else:
                contents[name] = uproot._util.awkward_form(member, file, context)

    return awkward.forms.RecordForm(
        list(contents.values()),
        list(contents.keys()),
        parameters={"__record__": classname},
    )


class AsStridedObjects(uproot.interpretation.numerical.AsDtype):
    """
    Args:
        model (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`): The
            full Uproot deserialization model for the data.
        members (list of (str, :doc:`uproot.interpretation.Interpretation`) tuples): The
            name and fixed-width interpretation for each member of the objects.
        original (None, :doc:`uproot.model.Model`, or :doc:`uproot.containers.AsContainer`): If
            this interpretation is derived from
            :ref:`uproot.interpretation.objects.AsObjects.simplify`, this is a
            reminder of the original
            :ref:`uproot.interpretation.objects.AsObjects.model`.

    Interpretation for an array (possibly
    :doc:`uproot.interpretation.jagged.AsJagged`) of fixed-size objects. Since
    the objects have a fixed number of fields with a fixed number of bytes each,
    the whole array (or :ref:`uproot.interpretation.jagged.AsJagged.content`)
    can be interpreted in one vectorized array-cast. Therefore, this
    interpretation is faster than :doc:`uproot.interpretation.objects.AsObjects`
    *when it is possible*.

    Unlike :doc:`uproot.interpretation.numerical.AsDtype` with a
    `structured array <https://numpy.org/doc/stable/user/basics.rec.html>`__,
    the objects in the final array have the methods required by its ``model``.
    If the ``library`` is :doc:`uproot.interpretation.library.NumPy`, these
    are instantiated as Python objects (slow); if
    :doc:`uproot.interpretation.library.Awkward`, they are behaviors passed to
    the Awkward Array's local
    `behavior <https://awkward-array.readthedocs.io/en/latest/ak.behavior.html>`__.
    """

    def __init__(self, model, members, original=None):
        self._model = model
        self._members = members
        self._original = original
        super().__init__(_unravel_members(members))

    @property
    def model(self):
        """
        The full Uproot deserialization model for the data
        (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`).
        """
        return self._model

    @property
    def members(self):
        """
        The name (str) and fixed-width
        :doc:`uproot.interpretation.Interpretation` for each member of the
        objects as a list of 2-tuple pairs.
        """
        return self._members

    @property
    def original(self):
        """
        If not None, this was the original
        :ref:`uproot.interpretation.objects.AsObjects.model` from an
        :doc:`uproot.interpretation.objects.AsObjects` that was simplified
        into this :doc:`uproot.interpretation.objects.AsStridedObjects`.
        """
        return self._original

    def __repr__(self):
        if self.inner_shape:
            return "AsStridedObjects({}, {})".format(
                self._model.__name__, self.inner_shape
            )
        else:
            return f"AsStridedObjects({self._model.__name__})"

    def __eq__(self, other):
        return isinstance(other, AsStridedObjects) and self._model == other._model

    @property
    def numpy_dtype(self):
        return numpy.dtype(object)

    def awkward_form(
        self,
        file,
        context=None,
        index_format="i64",
        header=False,
        tobject_header=False,
        breadcrumbs=(),
    ):
        context = self._make_context(
            context, index_format, header, tobject_header, breadcrumbs
        )
        awkward = uproot.extras.awkward()
        cname = uproot.model.classname_decode(self._model.__name__)[0]
        form = _strided_awkward_form(awkward, cname, self._members, file, context)
        for dim in reversed(self.inner_shape):
            form = awkward.forms.RegularForm(form, dim)
        return form

    @property
    def cache_key(self):
        return f"{type(self).__name__}({self._model.__name__})"

    @property
    def typename(self):
        return uproot.model.classname_decode(self._model.__name__)[0]

    def _wrap_almost_finalized(self, array):
        return StridedObjectArray(self, array)


class CannotBeStrided(Exception):
    """
    Exception used to stop recursion over
    :ref:`uproot.model.Model.strided_interpretation` and
    :ref:`uproot.containers.AsContainer.strided_interpretation` as soon as a
    non-conforming type is found.
    """

    pass


class CannotBeAwkward(Exception):
    """
    Exception used to stop recursion over
    :ref:`uproot.interpretation.Interpretation.awkward_form`,
    :ref:`uproot.model.Model.awkward_form` and
    :ref:`uproot.containers.AsContainer.awkward_form` as soon as a
    non-conforming type is found.
    """

    def __init__(self, because):
        self._because = because

    @property
    def because(self):
        return self._because


class ObjectArray:
    """
    Args:
        model (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`): The
            full Uproot deserialization model for the data.
        branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch`` from
            which the data are drawn.
        context (dict): Auxiliary data used in deserialization.
        byte_offsets (array of ``numpy.int32``): Index where each entry of the
            ``byte_content`` starts and stops.
        byte_content (array of ``numpy.uint8``): Raw but uncompressed data,
            directly from
            :ref:`uproot.interpretation.Interpretation.basket_array`.
        cursor_offset (int): Correction to the integer keys used in
            :ref:`uproot.source.cursor.Cursor.refs` for objects deserialized
            by reference (:doc:`uproot.deserialization.read_object_any`).

    Temporary array filled by
    :ref:`uproot.interpretation.objects.AsObjects.basket_array`, which will be
    turned into a NumPy, Awkward, or other array, depending on the specified
    :doc:`uproot.interpretation.library.Library`.
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
        return "ObjectArray({}, {}, {}, {}, {}, {})".format(
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
        (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`).
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
        :ref:`uproot.interpretation.Interpretation.basket_array`.
        """
        return self._byte_content

    @property
    def cursor_offset(self):
        """
        Correction to the integer keys used in
        :ref:`uproot.source.cursor.Cursor.refs` for objects deserialized by
        reference (:doc:`uproot.deserialization.read_object_any`).
        """
        return self._cursor_offset

    def to_numpy(self):
        """
        Convert this ObjectArray into a NumPy ``dtype="O"`` (object) array.
        """
        output = numpy.empty(len(self), dtype=numpy.dtype(object))
        for i in range(len(self)):
            output[i] = self[i]
        return output

    def __len__(self):
        return len(self._byte_offsets) - 1

    def __getitem__(self, where):
        if uproot._util.isint(where):
            byte_start = self._byte_offsets[where]
            byte_stop = self._byte_offsets[where + 1]
            data = self._byte_content[byte_start:byte_stop]
            chunk = uproot.source.chunk.Chunk.wrap(self._branch.file.source, data)
            cursor = uproot.source.cursor.Cursor(
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
            chunk = uproot.source.chunk.Chunk.wrap(source, data)
            cursor = uproot.source.cursor.Cursor(0, origin=-self._cursor_offset)
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


class StridedObjectArray:
    """
    Args:
        interpretation (:doc:`uproot.interpretation.objects.AsStridedObjects`): The
            interpretation that produced this array.
        array (array): Underlying array object, which may be NumPy or another
            temporary array.

    Temporary array filled by
    :ref:`uproot.interpretation.objects.AsStridedObjects.basket_array`, which
    will be turned into a NumPy, Awkward, or other array, depending on the
    specified :doc:`uproot.interpretation.library.Library`.
    """

    def __init__(self, interpretation, array):
        self._interpretation = interpretation
        self._array = array
        if not isinstance(self._array, numpy.ndarray):
            raise NotImplementedError(
                "non-numpy temporary arrays require changes in library interpretations"
            )

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

    @property
    def shape(self):
        assert self._array.shape == (len(self),) + self._interpretation.inner_shape
        return self._array.shape

    def __repr__(self):
        return f"StridedObjectArray({self._interpretation}, {self._array})"

    def __len__(self):
        return len(self._array)

    def __getitem__(self, where):
        out = self._array[where]
        if isinstance(out, tuple):
            return _strided_object("", self._interpretation, out)
        else:
            return StridedObjectArray(self._interpretation, out)

    def __iter__(self):
        for x in self._array:
            if isinstance(x, tuple):
                yield _strided_object("", self._interpretation, x)
            else:
                yield StridedObjectArray(self._interpretation, x)

    def ndenumerate(self):
        for i, x in numpy.ndenumerate(self._array):
            yield i, _strided_object("", self._interpretation, x)
