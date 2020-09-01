# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines a Python-like Future and Executor for Uproot in three levels:

1. :doc:`uproot4.source.futures.NoFuture` and
   :doc:`uproot4.source.futures.TrivialExecutor`: interface only, all activity
   is synchronous.
2. :doc:`uproot4.source.futures.Future`, :doc:`uproot4.source.futures.Worker`,
   and :doc:`uproot4.source.futures.ThreadPoolExecutor`: similar to Python's
   own Future, Thread, and ThreadPoolExecutor, though only a minimal
   implementation is provided. These exist to unify behavior between Python 2
   and 3 and provide a base class for the following.
3. :doc:`uproot4.source.futures.ResourceFuture`,
   :doc:`uproot4.source.futures.ResourceWorker`,
   and :doc:`uproot4.source.futures.ResourceThreadPoolExecutor`: like the above
   except that a :doc:`uproot4.source.chunk.Resource` is associated with every
   worker. When the threads are shut down, the resources (i.e. file handles)
   are released.

These classes implement a *subset* of Python's Future and Executor interfaces.
"""

from __future__ import absolute_import

import os
import sys
import threading
import time

import uproot4

try:
    import queue
except ImportError:
    import Queue as queue


def delayed_raise(exception_class, exception_value, traceback):
    """
    Raise an exception from a background thread on the main thread.
    """
    if uproot4._util.py2:
        exec("raise exception_class, exception_value, traceback")
    else:
        raise exception_value.with_traceback(traceback)


##################### use-case 1: trivial Futures/Executor (satisfying formalities)


class NoFuture(object):
    def __init__(self, result):
        self._result = result

    def result(self, timeout=None):
        return self._result


class TrivialExecutor(object):
    def __repr__(self):
        return "<TrivialExecutor at 0x{0:012x}>".format(id(self))

    def submit(self, task, *args):
        return task(*args)

    def shutdown(self, wait=True):
        pass


##################### use-case 2: Python-like Futures/Executor for compute


class Future(object):
    def __init__(self, task, args):
        self._task = task
        self._args = args
        self._finished = threading.Event()
        self._result = None
        self._excinfo = None

    def result(self, timeout=None):
        self._finished.wait(timeout=timeout)
        if self._excinfo is None:
            return self._result
        else:
            delayed_raise(*self._excinfo)

    def _run(self):
        try:
            self._result = self._task(*self._args)
        except Exception:
            self._excinfo = sys.exc_info()
        self._finished.set()


class Worker(threading.Thread):
    def __init__(self, work_queue):
        super(Worker, self).__init__()
        self.daemon = True
        self._work_queue = work_queue

    @property
    def work_queue(self):
        return self._work_queue

    def run(self):
        while True:
            future = self._work_queue.get()
            if future is None:
                break
            assert not isinstance(future, ResourceFuture)
            future._run()


class ThreadPoolExecutor(object):
    def __init__(self, num_workers=None):
        if num_workers is None:
            if hasattr(os, "cpu_count"):
                num_workers = os.cpu_count()
            else:
                import multiprocessing

                num_workers = multiprocessing.cpu_count()

        self._work_queue = queue.Queue()
        self._workers = []
        for x in uproot4._util.range(num_workers):
            self._workers.append(Worker(self._work_queue))
        for worker in self._workers:
            worker.start()

    def __repr__(self):
        return "<ThreadPoolExecutor ({0} workers) at 0x{1:012x}>".format(
            len(self._workers), id(self)
        )

    @property
    def num_workers(self):
        return len(self._workers)

    @property
    def workers(self):
        return self._workers

    def submit(self, task, *args):
        future = Future(task, args)
        self._work_queue.put(future)
        return future

    def shutdown(self, wait=True):
        while True:
            for worker in self._workers:
                if worker.is_alive():
                    self._work_queue.put(None)
            if any(worker.is_alive() for worker in self._workers):
                time.sleep(0.001)
            else:
                break


##################### use-case 3: worker-bound resources for I/O


class ResourceFuture(Future):
    def __init__(self, task):
        super(ResourceFuture, self).__init__(task, None)
        self._notify = None

    def _set_notify(self, notify):
        self._notify = notify

    def _set_excinfo(self, excinfo):
        if not self._finished.is_set():
            self._excinfo = excinfo
            self._finished.set()
            if self._notify is not None:
                self._notify()

    def _run(self, resource):
        try:
            self._result = self._task(resource)
        except Exception:
            self._excinfo = sys.exc_info()
        self._finished.set()
        if self._notify is not None:
            self._notify()


class ResourceWorker(Worker):
    def __init__(self, work_queue, resource):
        super(ResourceWorker, self).__init__(work_queue)
        self._resource = resource

    @property
    def resource(self):
        return self._resource

    def run(self):
        while True:
            future = self._work_queue.get()
            if future is None:
                break
            assert isinstance(future, ResourceFuture)
            future._run(self._resource)


class ResourceThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, resources):
        self._closed = False

        if len(resources) < 1:
            raise ValueError("at least one worker is required")

        self._work_queue = queue.Queue()
        self._workers = []
        for resource in resources:
            self._workers.append(ResourceWorker(self._work_queue, resource))
        for worker in self._workers:
            worker.start()

    def __repr__(self):
        return "<ResourceThreadPoolExecutor ({0} workers) at 0x{1:012x}>".format(
            len(self._workers), id(self)
        )

    def submit(self, future):
        assert isinstance(future, ResourceFuture)
        if self.closed:
            raise OSError(
                "resource is closed for file {0}".format(
                    self._workers[0].resource.file_path
                )
            )
        self._work_queue.put(future)
        return future

    @property
    def closed(self):
        return self._closed

    def close(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        for worker in self._workers:
            worker.resource.__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        self.shutdown()
        for worker in self._workers:
            worker.resource.__exit__(exception_type, exception_value, traceback)
        self._closed = True
