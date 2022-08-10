
import numpy
import uproot
from uproot._util import no_filter
from uproot.behaviors.TBranch import HasBranches, TBranch, _regularize_entries_start_stop


def _get_dask_array(
    files,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    step_size="100 MB",
    custom_classes=None,
    allow_missing=False,
    real_options=None,
):
    dask, da = uproot.extras.dask()
    hasbranches = []
    common_keys = None
    is_self = []

    count = 0
    for file_path, object_path in files:
        obj = uproot._util.regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, real_options
        )

        if obj is not None:
            count += 1

            if isinstance(obj, TBranch) and len(obj.keys(recursive=True)) == 0:
                original = obj
                obj = obj.parent
                is_self.append(True)

                def real_filter_branch(branch):
                    return branch is original and filter_branch(branch)

            else:
                is_self.append(False)
                real_filter_branch = filter_branch

            hasbranches.append(obj)

            new_keys = obj.keys(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=real_filter_branch,
                full_paths=full_paths,
            )

            if common_keys is None:
                common_keys = new_keys
            else:
                new_keys = set(new_keys)
                common_keys = [key for key in common_keys if key in new_keys]

    if count == 0:
        raise ValueError(
            "allow_missing=True and no TTrees found in\n\n    {}".format(
                "\n    ".join(
                    "{"
                    + "{}: {}".format(
                        repr(f.file_path if isinstance(f, HasBranches) else f),
                        repr(f.object_path if isinstance(f, HasBranches) else o),
                    )
                    + "}"
                    for f, o in files
                )
            )
        )

    if len(common_keys) == 0 or not (all(is_self) or not any(is_self)):
        raise ValueError(
            "TTrees in\n\n    {}\n\nhave no TBranches in common".format(
                "\n    ".join(
                    "{"
                    + "{}: {}".format(
                        repr(f.file_path if isinstance(f, HasBranches) else f),
                        repr(f.object_path if isinstance(f, HasBranches) else o),
                    )
                    + "}"
                    for f, o in files
                )
            )
        )

    dask_dict = {}

    @dask.delayed
    def delayed_get_array(ttree, key, start, stop):
        return ttree[key].array(library="np", entry_start=start, entry_stop=stop)

    for key in common_keys:
        dask_arrays = []
        for ttree in hasbranches:
            entry_start, entry_stop = _regularize_entries_start_stop(
                ttree.tree.num_entries, None, None
            )
            entry_step = 0
            if uproot._util.isint(step_size):
                entry_step = step_size
            else:
                entry_step = ttree.num_entries_for(step_size, expressions=f"{key}")

            dt = ttree[key].interpretation.numpy_dtype
            if dt.subdtype is None:
                inner_shape = ()
            else:
                dt, inner_shape = dt.subdtype

            def foreach(start):
                stop = min(start + entry_step, entry_stop)
                length = stop - start

                delayed_array = delayed_get_array(ttree, key, start, stop)
                shape = (length,) + inner_shape
                dask_arrays.append(
                    da.from_delayed(delayed_array, shape=shape, dtype=dt)
                )

            for start in range(entry_start, entry_stop, entry_step):
                foreach(start)

        dask_dict[key] = da.concatenate(dask_arrays)
    return dask_dict


def _get_dask_array_delay_open(
    files,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    custom_classes=None,
    allow_missing=False,
    real_options=None,
):
    dask, da = uproot.extras.dask()
    ffile_path, fobject_path = files[0]
    obj = uproot._util.regularize_object_path(
        ffile_path, fobject_path, custom_classes, allow_missing, real_options
    )
    common_keys = obj.keys(
        recursive=recursive,
        filter_name=filter_name,
        filter_typename=filter_typename,
        filter_branch=filter_branch,
        full_paths=full_paths,
    )

    dask_dict = {}
    delayed_open_fn = dask.delayed(uproot._util.regularize_object_path)

    @dask.delayed
    def delayed_get_array(ttree, key):
        return ttree[key].array(library="np")

    for key in common_keys:
        dask_arrays = []
        for file_path, object_path in files:
            delayed_tree = delayed_open_fn(
                file_path, object_path, custom_classes, allow_missing, real_options
            )
            delayed_array = delayed_get_array(delayed_tree, key)
            dt = obj[key].interpretation.numpy_dtype
            if dt.subdtype is not None:
                dt, inner_shape = dt.subdtype

            dask_arrays.append(
                da.from_delayed(delayed_array, shape=(numpy.nan,), dtype=dt)
            )
        dask_dict[key] = da.concatenate(dask_arrays, allow_unknown_chunksizes=True)
    return dask_dict

class _UprootRead:
    def __init__(self, hasbranches, branches) -> None:
        self.hasbranches = hasbranches
        self.branches = branches

    def __call__(self, i_start_stop):
        i, start, stop = i_start_stop
        return self.hasbranches[i].arrays(
            self.branches, entry_start=start, entry_stop=stop
        )

class _UprootOpenAndRead:
    def __init__(
        self, custom_classes, allow_missing, real_options, common_keys
    ) -> None:
        self.custom_classes = custom_classes
        self.allow_missing = allow_missing
        self.real_options = real_options
        self.common_keys = common_keys

    def __call__(self, file_path_object_path):
        file_path, object_path = file_path_object_path
        ttree = uproot._util.regularize_object_path(
            file_path,
            object_path,
            self.custom_classes,
            self.allow_missing,
            self.real_options,
        )
        return ttree.arrays(self.common_keys)

def _get_dak_array(
    files,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    step_size="100 MB",
    custom_classes=None,
    allow_missing=False,
    real_options=None,
):
    dask_awkward = uproot.extras.dask_awkward()

    hasbranches = []
    common_keys = None
    is_self = []

    count = 0
    for file_path, object_path in files:
        obj = uproot._util.regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, real_options
        )

        if obj is not None:
            count += 1

            if isinstance(obj, TBranch) and len(obj.keys(recursive=True)) == 0:
                original = obj
                obj = obj.parent
                is_self.append(True)

                def real_filter_branch(branch):
                    return branch is original and filter_branch(branch)

            else:
                is_self.append(False)
                real_filter_branch = filter_branch

            hasbranches.append(obj)

            new_keys = obj.keys(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=real_filter_branch,
                full_paths=full_paths,
            )

            if common_keys is None:
                common_keys = new_keys
            else:
                new_keys = set(new_keys)
                common_keys = [key for key in common_keys if key in new_keys]

    if count == 0:
        raise ValueError(
            "allow_missing=True and no TTrees found in\n\n    {}".format(
                "\n    ".join(
                    "{"
                    + "{}: {}".format(
                        repr(f.file_path if isinstance(f, HasBranches) else f),
                        repr(f.object_path if isinstance(f, HasBranches) else o),
                    )
                    + "}"
                    for f, o in files
                )
            )
        )

    if len(common_keys) == 0 or not (all(is_self) or not any(is_self)):
        raise ValueError(
            "TTrees in\n\n    {}\n\nhave no TBranches in common".format(
                "\n    ".join(
                    "{"
                    + "{}: {}".format(
                        repr(f.file_path if isinstance(f, HasBranches) else f),
                        repr(f.object_path if isinstance(f, HasBranches) else o),
                    )
                    + "}"
                    for f, o in files
                )
            )
        )


    partition_args = []
    for i, ttree in enumerate(hasbranches):
        entry_start, entry_stop = _regularize_entries_start_stop(
            ttree.num_entries, None, None
        )
        entry_step = 0
        if uproot._util.isint(step_size):
            entry_step = step_size
        else:
            entry_step = ttree.num_entries_for(step_size, expressions=common_keys)

        def foreach(start):
            stop = min(start + entry_step, entry_stop)
            partition_args.append((i, start, stop))

        for start in range(entry_start, entry_stop, entry_step):
            foreach(start)

    first_basket_start, first_basket_stop = hasbranches[0][common_keys[0]].basket_entry_start_stop(0)
    first_basket = hasbranches[0].arrays(common_keys, entry_start=first_basket_start, entry_stop=first_basket_stop)
    meta = dask_awkward.core.typetracer_array(first_basket)

    return dask_awkward.from_map(
        _UprootRead(hasbranches, common_keys),
        partition_args,
        label="from-uproot",
        meta=meta,
    )


def _get_dak_array_delay_open(
    files,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    custom_classes=None,
    allow_missing=False,
    real_options=None,
):
    dask_awkward = uproot.extras.dask_awkward()

    ffile_path, fobject_path = files[0]
    obj = uproot._util.regularize_object_path(
        ffile_path, fobject_path, custom_classes, allow_missing, real_options
    )
    common_keys = obj.keys(
        recursive=recursive,
        filter_name=filter_name,
        filter_typename=filter_typename,
        filter_branch=filter_branch,
        full_paths=full_paths,
    )


    first_basket_start, first_basket_stop = obj[common_keys[0]].basket_entry_start_stop(0)
    first_basket = obj.arrays(common_keys, entry_start=first_basket_start, entry_stop=first_basket_stop)
    meta = dask_awkward.core.typetracer_array(first_basket)

    return dask_awkward.from_map(
        _UprootOpenAndRead(custom_classes, allow_missing, real_options, common_keys),
        files,
        label="from-uproot",
        meta=meta,
    )

