Options (type; default):

* initial_directory_bytes (int; 256): The number of bytes to allocate for new directories so that
  TKeys can be added to them without needing immediate rewriting of the block.
* initial_streamers_bytes (int; 1024): The number of bytes to allocate for a new list of streamers
  so that streamers can be added to it without needing immediate rewriting
* uuid_function (callable; ``uuid.uuid1``): Function to create the file's UUID and/or any directory's UUID.
* compression (:doc:`uproot.compression.Compression` or None; ``uproot.ZLIB(1)``): Compression algorithm
  and level for new objects added to the file. Can be updated after creating
  the :doc:`uproot.writing.writable.WritableFile`.

See :doc:`uproot.writing.writable.WritableFile` for details on these options.

Additional options are passed as ``storage_options`` to the fsspec filesystem
