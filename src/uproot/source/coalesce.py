"""Read coalescing algorithms

Inspired in part by https://github.com/cms-sw/cmssw/blob/master/IOPool/TFileAdaptor/src/ReadRepacker.h
"""

from __future__ import annotations

import queue
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable

import uproot.source.chunk


@dataclass
class CoalesceConfig:
    max_range_gap: int = 32 * 1024
    max_request_ranges: int = 1024
    max_request_bytes: int = 10 * 1024 * 1024
    min_first_request_bytes: int = 32 * 1024


DEFAULT_CONFIG = CoalesceConfig()


class SliceFuture:
    def __init__(self, parent: Future, s: slice | int):
        self._parent = parent
        self._s = s

    def add_done_callback(self, callback, *, context=None):
        self._parent.add_done_callback(callback)

    def result(self, timeout=None):
        return self._parent.result(timeout=timeout)[self._s]


@dataclass
class RangeRequest:
    start: int
    stop: int
    future: Future | None


@dataclass
class Cluster:
    ranges: list[RangeRequest]

    @property
    def start(self):
        # since these are built from sorted ranges, this is the min start
        return self.ranges[0].start

    @property
    def stop(self):
        return max(range.stop for range in self.ranges)

    def __len__(self):
        return self.stop - self.start

    def set_future(self, future: Future):
        for range in self.ranges:
            local_start = range.start - self.start
            local_stop = range.stop - self.start
            range.future = SliceFuture(future, slice(local_start, local_stop))


@dataclass
class CoalescedRequest:
    clusters: list[Cluster]

    def ranges(self):
        return [(cluster.start, cluster.stop) for cluster in self.clusters]

    def set_future(self, future: Future):
        for i, cluster in enumerate(self.clusters):
            cluster.set_future(SliceFuture(future, i))


def _merge_adjacent(ranges: list[RangeRequest], config: CoalesceConfig):
    sorted_ranges = sorted(ranges, key=lambda r: r.start)
    cluster = Cluster([])
    for current_range in sorted_ranges:
        if cluster.ranges and current_range.start - cluster.stop > config.max_range_gap:
            yield cluster
            cluster = Cluster([])
        cluster.ranges.append(current_range)
    if cluster.ranges:
        yield cluster


def _coalesce(ranges: list[RangeRequest], config: CoalesceConfig):
    clusters: list[Cluster] = []
    request_bytes: int = 0
    first_request = True
    for cluster in _merge_adjacent(ranges, config):
        if clusters and (
            len(clusters) + 1 >= config.max_request_ranges
            or request_bytes + len(cluster) >= config.max_request_bytes
            or (first_request and request_bytes >= config.min_first_request_bytes)
        ):
            yield CoalescedRequest(clusters)
            clusters = []
            request_bytes = 0
            first_request = False
        clusters.append(cluster)
        request_bytes += len(cluster)
    if clusters:
        yield CoalescedRequest(clusters)


def coalesce_requests(
    ranges: list[tuple[int, int]],
    submit_fn: Callable[[list[tuple[int, int]]], Future],
    source: uproot.source.chunk.Source,
    notifications: queue.Queue,
    config: CoalesceConfig | None = None,
):
    if config is None:
        config = DEFAULT_CONFIG
    all_requests = [RangeRequest(start, stop, None) for start, stop in ranges]
    for merged_request in _coalesce(all_requests, config):
        future = submit_fn(merged_request.ranges())
        merged_request.set_future(future)

    def chunkify(req: RangeRequest):
        chunk = uproot.source.chunk.Chunk(source, req.start, req.stop, req.future)
        req.future.add_done_callback(uproot.source.chunk.notifier(chunk, notifications))
        return chunk

    return list(map(chunkify, all_requests))
