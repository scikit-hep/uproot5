# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a Python-like Future and Executor for Uproot in three levels:

1. :doc:`uproot.source.futures.TrivialFuture` and
   :doc:`uproot.source.futures.TrivialExecutor`: interface only, all activity
   is synchronous.
2. :doc:`uproot.source.futures.Future`, :doc:`uproot.source.futures.Worker`,
   and :doc:`uproot.source.futures.ThreadPoolExecutor`: similar to Python's
   own Future, Thread, and ThreadPoolExecutor, though only a minimal
   implementation is provided. These exist to unify behavior between Python 2
   and 3 and provide a base class for the following.
3. :doc:`uproot.source.futures.ResourceFuture`,
   :doc:`uproot.source.futures.ResourceWorker`,
   and :doc:`uproot.source.futures.ResourceThreadPoolExecutor`: like the above
   except that a :doc:`uproot.source.chunk.Resource` is associated with every
   worker. When the threads are shut down, the resources (i.e. file handles)
   are released.

These classes implement a *subset* of Python's Future and Executor interfaces.
"""


import os
import queue
import sys
import threading
import time


def delayed_raise(exception_class, exception_value, traceback):
    """
    Raise an exception from a background thread on the main thread.
    """
    raise exception_value.with_traceback(traceback)


##################### use-case 1: trivial Futures/Executor (satisfying formalities)


class TrivialFuture:
    """
    Formally satisfies the interface for a :doc:`uproot.source.futures.Future`
    object, but it is already complete at the time when it is constructed.
    """

    def __init__(self, result):
        self._result = result

    def result(self, timeout=None):
        """
        The result of this (Trivial)Future.
        """
        return self._result


class TrivialExecutor:
    """
    Formally satisfies the interface for a
    :doc:`uproot.source.futures.ThreadPoolExecutor`, but the
    :ref:`uproot.source.futures.TrivialExecutor.submit` method computes its
    ``task`` synchronously.
    """

    def __repr__(self):
        return f"<TrivialExecutor at 0x{id(self):012x}>"

    def submit(self, task, *args):
        """
        Immediately runs ``task(*args)``.
        """
        return TrivialFuture(task(*args))

    def shutdown(self, wait=True):
        """
        Does nothing, since this object does not have threads to stop.
        """
        pass


##################### use-case 2: Python-like Futures/Executor for compute


class Future:
    """
    Args:
        task (function): The function to evaluate.
        args (tuple): Arguments for the function.

    Like Python 3 ``concurrent.futures.Future`` except that it has only
    the subset of the interface Uproot needs and is available in Python 2.

    The :doc:`uproot.source.futures.ResourceFuture` extends this class.
    """

    def __init__(self, task, args):
        self._task = task
        self._args = args
        self._finished = threading.Event()
        self._result = None
        self._excinfo = None

    def result(self, timeout=None):
        """
        Waits until the task completes (with a ``timeout``) and returns its
        result.

        If the task raises an exception in its background thread, this function
        raises that exception on the thread on which it is called.
        """
        self._finished.wait(timeout=timeout)
        if self._excinfo is None:
            return self._result
        else:
            delayed_raise(*self._excinfo)

    def _run(self):
        try:
            if self._task is None:
                raise RuntimeError("cannot run Future twice")
            self._result = self._task(*self._args)
        except Exception:
            self._excinfo = sys.exc_info()
        self._finished.set()
        del self._task, self._args
        self._task = None
        self._args = ()


class Worker(threading.Thread):
    """
    Args:
        work_queue (``queue.Queue``): The worker calls ``get`` on this queue
            for tasks in the form of :doc:`uproot.source.futures.Future`
            objects and runs them. If it ever gets a None value, the thread
            is stopped.

    A ``threading.Thread`` for the
    :doc:`uproot.source.futures.ThreadPoolExecutor`.
    """

    def __init__(self, work_queue):
        super().__init__()
        self.daemon = True
        self._work_queue = work_queue

    @property
    def work_queue(self):
        """
        The worker calls ``get`` on this queue for tasks in the form of
        :doc:`uproot.source.futures.Future` objects and runs them. If it ever
        gets a None value, the thread is stopped.
        """
        return self._work_queue

    def run(self):
        """
        Listens to the :ref:`uproot.source.futures.Worker.work_queue` and
        executes each :doc:`uproot.source.futures.Future` it receives until it
        receives None.
        """
        future = None
        while True:
            del future  # don't hang onto a reference while waiting for more work
            future = self._work_queue.get()
            if future is None:
                break
            assert not isinstance(future, ResourceFuture)
            future._run()


class ThreadPoolExecutor:
    """
    Args:
        num_workers (None or int): The number of workers to start. If None,
            use ``os.cpu_count()``.

    Like Python 3 ``concurrent.futures.ThreadPoolExecutor`` except that it has
    only the subset of the interface Uproot needs and is available in Python 2.

    The :doc:`uproot.source.futures.ResourceThreadPoolExecutor` extends this
    class.
    """

    def __init__(self, num_workers=None):
        if num_workers is None:
            if hasattr(os, "cpu_count"):
                num_workers = os.cpu_count()
            else:
                import multiprocessing

                num_workers = multiprocessing.cpu_count()

        self._work_queue = queue.Queue()
        self._workers = []
        for _ in range(num_workers):
            self._workers.append(Worker(self._work_queue))
        for worker in self._workers:
            worker.start()

    def __repr__(self):
        return "<ThreadPoolExecutor ({} workers) at 0x{:012x}>".format(
            len(self._workers), id(self)
        )

    @property
    def num_workers(self):
        """
        The number of workers.
        """
        return len(self._workers)

    @property
    def workers(self):
        """
        A list of workers (:doc:`uproot.source.futures.Worker`).
        """
        return self._workers

    def submit(self, task, *args):
        """
        Pass the ``task`` and ``args`` onto the workers'
        :ref:`uproot.source.futures.Worker.work_queue` as a
        :doc:`uproot.source.futures.Future` so that it will be executed when
        one is available.
        """
        future = Future(task, args)
        self._work_queue.put(future)
        return future

    def shutdown(self, wait=True):
        """
        Stop every :doc:`uproot.source.futures.Worker` by putting None
        on the :ref:`uproot.source.futures.Worker.work_queue` until none of
        them satisfy ``worker.is_alive()``.
        """
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
    """
    Args:
        task (function): The function to evaluate with a
            :doc:`uproot.source.chunk.Resource` as its first argument.

    A :doc:`uproot.source.futures.Future` that uses the
    :doc:`uproot.source.chunk.Resource` associated with the
    :doc:`uproot.source.futures.ResourceWorker` that runs it.
    """

    def __init__(self, task):
        super().__init__(task, None)
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
            del self._notify
            self._notify = None


class ResourceWorker(Worker):
    """
    Args:
        work_queue (``queue.Queue``): The worker calls ``get`` on this queue
            for tasks in the form of :doc:`uproot.source.futures.Future`
            objects and runs them. If it ever gets a None value, the thread
            is stopped.

    A :doc:`uproot.source.futures.Worker` that is bound to a
    :doc:`uproot.source.chunk.Resource`. This
    :ref:`uproot.source.futures.ResourceWorker.resource` is the first argument
    passed to each :doc:`uproot.source.futures.ResourceFuture` that it
    executes.
    """

    def __init__(self, work_queue, resource):
        super().__init__(work_queue)
        self._resource = resource

    @property
    def resource(self):
        """
        The :doc:`uproot.source.chunk.Resource` that is bound to this worker.
        """
        return self._resource

    def run(self):
        """
        Listens to the :ref:`uproot.source.futures.ResourceWorker.work_queue`
        and executes each :doc:`uproot.source.futures.ResourceFuture` it
        receives (with :ref:`uproot.source.futures.ResourceWorker.resource` as
        its first argument) until it receives None.
        """
        future = None
        while True:
            del future  # don't hang onto a reference while waiting for more work
            future = self._work_queue.get()
            if future is None:
                break
            assert isinstance(future, ResourceFuture)
            future._run(self._resource)


class ResourceThreadPoolExecutor(ThreadPoolExecutor):
    """
    Args:
        resources (list of :doc:`uproot.source.chunk.Resource`): Resources to
            wrap as :doc:`uproot.source.futures.ResourceFuture` objects.

    A :doc:`uproot.source.futures.ThreadPoolExecutor` whose workers are bound
    to resources, such as file handles.
    """

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
        return "<ResourceThreadPoolExecutor ({} workers) at 0x{:012x}>".format(
            len(self._workers), id(self)
        )

    def submit(self, future):
        """
        Pass the ``task`` onto the workers'
        :ref:`uproot.source.futures.ResourceWorker.work_queue` as a
        :doc:`uproot.source.futures.ResourceFuture` so that it will be
        executed with its :ref:`uproot.source.futures.ResourceWorker.resource`
        when that worker is available.
        """
        assert isinstance(future, ResourceFuture)
        if self.closed:
            raise OSError(
                "resource is closed for file {}".format(
                    self._workers[0].resource.file_path
                )
            )
        self._work_queue.put(future)
        return future

    def close(self):
        """
        Stops all :doc:`uproot.source.futures.ResourceWorker` threads and frees
        their :ref:`uproot.source.futures.ResourceWorker.resource`.
        """
        self.__exit__(None, None, None)

    @property
    def closed(self):
        """
        True if the :doc:`uproot.source.futures.ResourceWorker` threads have
        been stopped and their
        :ref:`uproot.source.futures.ResourceWorker.resource` freed.
        """
        return self._closed

    def __enter__(self):
        for worker in self._workers:
            worker.resource.__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        self.shutdown()
        for worker in self._workers:
            worker.resource.__exit__(exception_type, exception_value, traceback)
        self._closed = True
