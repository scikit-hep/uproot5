# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import time
import threading

try:
    import queue
except ImportError:
    import Queue as queue

import uproot4._util


class TrivialFuture(object):
    def __init__(self, result):
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


class ResourceExecutor(object):
    def __init__(self, resource):
        self._resource = resource

    @property
    def num_workers(self):
        return 1

    @property
    def ready(self):
        return self._resource.ready

    def __enter__(self):
        self._resource.__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        self._resource.__exit__(exception_type, exception_value, traceback)

    def submit(self, fn, *args, **kwargs):
        return TrivialFuture(fn(self._resource, *args, **kwargs))

    def map(self, func, *iterables):
        for x in iterables:
            yield func(self._resource, x)

    def shutdown(self, wait=True):
        pass


class TaskFuture(object):
    def __init__(self, task, *args, **kwargs):
        self._task = task
        self._args = args
        self._kwargs = kwargs
        self._finished = threading.Event()
        self._excinfo = None

    def cancel(self):
        raise NotImplementedError

    def cancelled(self):
        raise NotImplementedError

    def running(self):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError

    def result(self, timeout=None):
        self._finished.wait(timeout=timeout)
        if self._excinfo is None:
            return self._result
        else:
            cls, err, trc = self._excinfo
            if uproot4._util.py2:
                exec("raise cls, err, trc")
            else:
                raise err.with_traceback(trc)

    def exception(self, timeout=None):
        raise NotImplementedError

    def add_done_callback(self, fn):
        raise NotImplementedError


class ThreadResourceWorker(threading.Thread):
    def __init__(self, resource, work_queue):
        super(ThreadResourceWorker, self).__init__()
        self.daemon = True
        self._resource = resource
        self._work_queue = work_queue

    @property
    def resource(self):
        return self._resource

    @property
    def work_queue(self):
        return self._work_queue

    def run(self):
        while True:
            future = self._work_queue.get()
            if future is None:
                break

            assert isinstance(future, TaskFuture)
            try:
                future._result = future._task(
                    self._resource, *future._args, **future._kwargs
                )
            except Exception:
                future._excinfo = sys.exc_info()
            future._finished.set()


class ThreadResourceExecutor(object):
    def __init__(self, resources):
        self._work_queue = queue.Queue()
        self._workers = [ThreadResourceWorker(x, self._work_queue) for x in resources]
        for thread in self._workers:
            thread.start()

    @property
    def num_workers(self):
        return len(self._workers)

    @property
    def workers(self):
        return self._workers

    @property
    def ready(self):
        return all(x.resource.ready for x in self._workers)

    def __enter__(self):
        for thread in self._workers:
            thread.resource.__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        self.shutdown()
        for thread in self._workers:
            thread.resource.__exit__(exception_type, exception_value, traceback)

    def submit(self, fn, *args, **kwargs):
        task = TaskFuture(fn, *args, **kwargs)
        self._work_queue.put(task)
        return task

    def map(self, func, *iterables):
        futures = [self.submit(func, x) for x in iterables]
        for future in futures:
            yield future.result()

    def shutdown(self, wait=True):
        while any(thread.is_alive() for thread in self._workers):
            for x in range(len(self._workers)):
                self._work_queue.put(None)
            time.sleep(0.001)
