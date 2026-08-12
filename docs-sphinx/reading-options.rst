Options (type; default):

* handler (:doc:`uproot.source.chunk.Source` class; None)
    Class implementing reading from the data source.
    If None, deduced from input file type.
* timeout (float for HTTP, int for XRootD; default defined by source implementation)
    The time in seconds to wait before giving up on the connection.
    Ignored for non-internet sources like local file paths.
* max_num_elements (None or int; None)
   The maximum number of elements to be requested in a single vector read, when using XRootD.
* num_workers (int; 1)
    Number of tasks to spawn for reading, only used by some source types
* use_threads (bool; False on the emscripten platform (i.e. in a web browser), else True)
    Use multi-threading when spawning workers.
* num_fallback_workers (int; 10)
    Number of tasks to spawn for reading in fallback mode (for example, multi-threading
    requests instead of a multipart GET for an http source)
* begin_chunk_size (memory_size; 403, the smallest a ROOT file can be)
    Size of first chunk that we attempt to read in bytes.
* minimal_ttree_metadata (bool; True)
    Skip rarely used metadata and defer reading of embedded TBaskets
* http_max_header_bytes (int; 21784)
    Maximum size of HTTP packet in bytes when the source is http

