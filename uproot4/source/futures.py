# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines Futures and Executors for Uproot Sources.

These are distinct from Python's built-in Futures and Executors because each
Thread in the thread pools are associated with one Resource, such as an open
file handle.
"""

from __future__ import absolute_import

import os
import sys
import time
import threading

try:
    import queue
except ImportError:
    import Queue as queue

import uproot4._util


def delayed_raise(exception_class, exception_value, traceback):
    """
    Raise an exception from a background thread on the main thread.
    """
    if uproot4._util.py2:
        exec("raise exception_class, exception_value, traceback")
    else:
        raise exception_value.with_traceback(traceback)


class Future(object):
    """
    Abstract base class for Futures, which have the same interface as Python
    Futures.
    """


class Executor(object):
    """
    Abstract base class for Executors, which have the same interface as Python
    Executors.
    """


class TrivialFuture(Future):
    """
    A Future that is filled as soon as it is created.
    """

    def __init__(self, result):
        """
        Creates a TrivialFuture preloaded with a `result`.
        """
        self._result = result

    def cancel(self):
        return False

    def cancelled(self):
        return False

    def running(self):
        return False

    def done(self):
        return True

    def result(self, timeout=None):
        return self._result

    def exception(self, timeout=None):
        return None

    def add_done_callback(self, fn):
        return fn(self)


class TrivialExecutor(Executor):
    """
    An Executor that doesn't manage any Threads or Resources.
    """

    def __repr__(self):
        return "<TrivialExecutor at 0x{0:012x}>".format(id(self))

    @property
    def num_workers(self):
        """
        Always returns 0, which indicates the lack of background workers.
        """
        return 0

    def __enter__(self):
        """
        Returns self.
        """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Does nothing.
        """
        pass

    def submit(self, fn, *args, **kwargs):
        """
        Immediately evaluate the function `fn` with `args` and `kwargs`.
        """
        if isinstance(fn, TrivialFuture):
            return fn
        else:
            return TrivialFuture(fn(*args, **kwargs))

    def map(self, func, *iterables):
        """
        Like Python's Executor.
        """
        for x in iterables:
            yield func(x)

    def shutdown(self, wait=True):
        """
        Does nothing.
        """
        pass


class ResourceExecutor(Executor):
    """
    An Executor that doesn't manage any Threads, but does manage Resources,
    such as file handles (as a context manager).
    """

    def __init__(self, resource):
        """
        Args:
            resource (Resource): Something to pass `__enter__` and `__exit__`
                to when entering and exiting the scope of a context block.
        """
        self._resource = resource

    @property
    def num_workers(self):
        """
        Always returns 0, which indicates the lack of background workers.
        """
        return 0

    def __enter__(self):
        """
        Passes `__enter__` to the Resource.
        """
        self._resource.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Passes `__exit__` to the Resource.
        """
        self._resource.__exit__(exception_type, exception_value, traceback)

    @property
    def closed(self):
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        return self._resource.closed

    def _prepare(self, fn, *args, **kwargs):
        return TrivialFuture(fn(self._resource, *args, **kwargs))

    def submit(self, fn, *args, **kwargs):
        """
        Immediately evaluate the function `fn` with `resource` as a first
        argument, before `args` and `kwargs`.
        """
        if isinstance(fn, TrivialFuture):
            return fn
        else:
            return self._prepare(fn, *args, **kwargs)

    def map(self, func, *iterables):
        """
        Like Python's Executor.
        """
        for x in iterables:
            yield func(self._resource, x)

    def shutdown(self, wait=True):
        """
        Manually calls `__exit__`.
        """
        self.__exit__(None, None, None)


class TaskFuture(Future):
    """
    A Future that waits for a `result` to be filled (by an Executor or one of
    its Threads).

    Contains one `threading.Event` to block `result` until ready.
    """

    def __init__(self, task):
        """
        Args:
            task (None or callable): A zero-argument callable that produces
                the `result` or None if the `result` is assigned externally.
        """
        self._task = task
        self._finished = threading.Event()
        self._result = None
        self._excinfo = None
        self._callback = None

    def cancel(self):
        raise NotImplementedError

    def cancelled(self):
        raise NotImplementedError

    def running(self):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError

    def _set_finished(self):
        self._finished.set()
        if self._callback is not None:
            self._callback(self)

    def result(self, timeout=None):
        """
        Wait for the `threading.Event` to be set, and then either return the
        `result` or raise the exception that occurred on the filling Thread.
        """
        self._finished.wait(timeout=timeout)
        if self._excinfo is None:
            return self._result
        else:
            delayed_raise(*self._excinfo)

    def exception(self, timeout=None):
        raise NotImplementedError

    def add_done_callback(self, fn):
        self._callback = fn


class ThreadResourceWorker(threading.Thread):
    """
    A Python Thread that controls one Resource and watches a `work_queue` for
    Futures to evaluate (as callables).
    """

    def __init__(self, resource, work_queue):
        """
        Args:
            resource (Resource): First argument passed to each Future's `task`
                callable.
            work_queue (queue.Queue): FIFO for work.

        This Thread pulls items from the `work_queue` or waits for it to be
        filled.

        If it receives a None from the `work_queue`, it shuts down.
        """
        super(ThreadResourceWorker, self).__init__()
        self.daemon = True
        self._resource = resource
        self._work_queue = work_queue

    @property
    def resource(self):
        """
        First argument passed to each Future's `task` callable.
        """
        return self._resource

    @property
    def work_queue(self):
        """
        FIFO for work.
        """
        return self._work_queue

    def run(self):
        """
        Listens to the `work_queue`, processing each Future it recieves.

        If it finds a None on the `work_queue`, the Thread shuts down.
        """
        while True:
            future = self._work_queue.get()
            if future is None:
                break

            assert isinstance(future, TaskFuture)
            try:
                if self._resource is None:
                    future._result = future._task()
                else:
                    future._result = future._task(self._resource)
            except Exception:
                future._excinfo = sys.exc_info()
            future._set_finished()


class ThreadPoolExecutor(Executor):
    """
    An Executor that manages only Threads, not Resources.

    All Threads are shut down when exiting a context block.
    """

    def __init__(self, num_workers=None):
        """
        Args:
            num_workers (None or int): Number of threads to launch; if None,
                use os.cpu_count().
        """
        if num_workers is None:
            if hasattr(os, "cpu_count"):
                num_workers = os.cpu_count()

            else:
                import multiprocessing

                num_workers = multiprocessing.cpu_count()

        self._work_queue = queue.Queue()
        self._workers = []
        for x in range(num_workers):
            self._workers.append(ThreadResourceWorker(None, self._work_queue))
        for thread in self._workers:
            thread.start()

    def __repr__(self):
        return "<ThreadPoolExecutor ({0} workers) at 0x{1:012x}>".format(
            len(self._workers), id(self)
        )

    @property
    def num_workers(self):
        """
        The number of Threads in this thread pool.
        """
        return len(self._workers)

    @property
    def workers(self):
        """
        The Threads in this thread pool.
        """
        return self._workers

    def __enter__(self):
        """
        Returns self.
        """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Shuts down the Threads in the thread pool.
        """
        self.shutdown()

    def submit(self, fn, *args, **kwargs):
        """
        Submits a function to be evaluated by a Thread in the thread pool.
        """
        if len(args) != 0 or len(kwargs) != 0:
            task = TaskFuture(lambda: fn(*args, **kwargs))
        else:
            task = TaskFuture(fn)

        self._work_queue.put(task)
        return task

    def map(self, func, *iterables):
        """
        Like Python's Executor.
        """
        futures = [self.submit(func, x) for x in iterables]
        for future in futures:
            yield future.result()

    def shutdown(self, wait=True):
        """
        Puts None on the `work_queue` until all Threads get the message and
        shut down.
        """
        while any(thread.is_alive() for thread in self._workers):
            for x in range(len(self._workers)):
                self._work_queue.put(None)
            time.sleep(0.001)


class ThreadResourceExecutor(Executor):
    """
    An Executor that manages Threads as well as Resources, such as file handles
    (one Resource per Thread).

    All Threads are shut down and Resources are released when exiting a context
    block.
    """

    def __init__(self, resources):
        """
        Args:
            resources (iterable of Resource): Resources, such as file handles,
                to manage; spawns one Thread per Resource.
        """
        self._closed = False
        self._work_queue = queue.Queue()
        self._workers = [ThreadResourceWorker(x, self._work_queue) for x in resources]
        for thread in self._workers:
            thread.start()

    @property
    def num_workers(self):
        """
        The number of Threads in this thread pool.
        """
        return len(self._workers)

    @property
    def workers(self):
        """
        The Threads in this thread pool.
        """
        return self._workers

    def __enter__(self):
        """
        Passes `__enter__` to the Resources attached to each worker.
        """
        for thread in self._workers:
            thread.resource.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Passes `__exit__` to the Resources attached to each worker and shuts
        down the Threads in the thread pool.
        """
        self.shutdown()
        for thread in self._workers:
            thread.resource.__exit__(exception_type, exception_value, traceback)

    @property
    def closed(self):
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        return self._closed

    def _prepare(self, fn, *args, **kwargs):
        if len(args) != 0 or len(kwargs) != 0:
            return TaskFuture(lambda: fn(*args, **kwargs))
        else:
            return TaskFuture(fn)

    def submit(self, fn, *args, **kwargs):
        """
        Submits a function to be evaluated by a Thread in the thread pool.

        The Resource associated with that Thread is passed as the first argument
        to the callable `fn`.
        """
        if isinstance(fn, TaskFuture):
            task = fn
        else:
            task = self._prepare(fn)

        self._work_queue.put(task)
        return task

    def map(self, func, *iterables):
        """
        Like Python's Executor.
        """
        futures = [self.submit(func, x) for x in iterables]
        for future in futures:
            yield future.result()

    def shutdown(self, wait=True):
        """
        Puts None on the `work_queue` until all Threads get the message and
        shut down.
        """
        self._closed = True
        while any(thread.is_alive() for thread in self._workers):
            for x in range(len(self._workers)):
                self._work_queue.put(None)
            time.sleep(0.001)
