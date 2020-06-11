#!/usr/bin/python

from __future__ import absolute_import
import os
import threading
try:
    import queue
except ImportError:
    import Queue as queue
import logging
import time
from .search import BayesianSearch
from .comm import create_comm


class Tuner(object):
    """ Tuning partition size and credit size using Bayesian Optimization."""

    def __init__(self, rank, arch, credit_tuning, partition_tuning, logger=None):
        """Init auto-tuner.

        Arguments:
            rank: The rank of the worker.
            arch: The communication architecture, either "ps" or "allreduce".
            credit_tuning: Enable tuning credit if True.
            partition_tuning: Enable tuning partition if True.
            logger: The logging handler.
        """
        self._rank = rank
        self._arch = arch
        self._credit_tuning = credit_tuning
        self._partition_tuning = partition_tuning
        if logger is None:
            self._logger = logging.getLogger("ByteScheduler")
        else:
            self._logger = logger
        # self._tuner = None
        if partition_tuning:
            self._comm = create_comm(rank=self._rank)
        else:
            self._comm = create_comm(rank=self._rank, host="localhost", port=58888)
        self._exit = False
        self._timestamps = []
        self._effective_point = None
        self.stopped = False

        # # Credit and partition in terms of million parameters.
        # space = dict()
        # if credit_tuning:
        #     if arch == "ps":
        #         space["credit"] = (1.0, 16.0)
        #     else:
        #         space["credit"] = (4.0, 64.0)

        # if partition_tuning:
        #     space["partition"] = (2.0, 32.0)

        # max_num_steps = 15
        # self._tuner = BayesianSearch(space=space, max_num_steps=max_num_steps)

        self.avg_duration = None
        self.avg_count = 0
        self.tune_thres = 0.05

    def next_point(self):
        """Core will call this function at the beginning of each step
        None means no point to try and the scheduler will use existing partition and credit settings
        the point is a dict, containing 3 keys: step, credit, partition. step is used to signal at which step to apply
        the new credit and partition.

        Returns:
            A dict of hyper-parameters for probing in the next step.
        """
        point = self._comm.get()
        if point:
            self._effective_point = point
        return point

    def record(self, current_point, step):
        print("[Tunner] record(): step = %d"%(step))
        """The scheduler will call this function at the beginning of each step,
        so that the tuner can collect timestamp information and derive feedback (i.e., duration of one training step).
        """
        if self._rank != 0:
            return
        if self._effective_point is None or self._effective_point == current_point:
            self._timestamps.append(time.time())

            # require at least 2 timestamps to calculating differences (durations)
            if len(self._timestamps) > 2:
                self._tune(current_point, step)

    def exit(self):
        """Stop tuning."""
        self._comm.shutdown()

    def _tune(self, current_point, step):
        print("[Tunner] _tune(): step = %d"%(step))
        """Run one step tuning."""
        step_durations = []
        print("[Tunner] len(self._timestamps) = %d"%(len(self._timestamps)))
        print(self._timestamps)
        
        # calculate durations
        for i in range(len(self._timestamps) - 1):
            step_durations.append(self._timestamps[i + 1] - self._timestamps[i])
            # continue
        print(step_durations)
        if step_durations:
            # next_point = None
            self._timestamps = []
            avg_step_duration = sum(step_durations) / len(step_durations)
            self.avg_count = self.avg_count + len(step_durations)
            print("[Tunner] self.avg_count")
            print(self.avg_count)
            if self.avg_duration is None:
                self.avg_duration = avg_step_duration
            elif avg_step_duration < (self.avg_duration * (1 + self.tune_thres)):
                print("[Tunner] additive-increase")
                print("self.avg_duration")
                print(self.avg_duration)
                self.avg_duration = (self.avg_duration * self.avg_count + avg_step_duration * len(step_durations)) / (self.avg_count + len(step_durations))
                self.avg_count = self.avg_count + len(step_durations)
                # print("[Tunner] exit point 1")
                next_point = current_point
                # print("next_point")
                # print(next_point)
                next_point["credit"] = next_point["credit"] + next_point["partition"] # additive-increase
                next_point["step"] = step + 1 # tune in next 2 iter
                print("[Tunner] exit point 2")
                self._comm.broadcast(next_point)
                print("[Tunner] exit point 3")
            
            elif avg_step_duration > (self.avg_duration * (1 + self.tune_thres)):
                print("[Tunner] multiplicative-decrease")
                # throughput degradation larger than threshold, possibly networking status change
                next_point = current_point
                next_point["credit"] = next_point["credit"] * self.avg_duration / avg_step_duration # multiplicative-decrease
                next_point["step"] = step + 1 # tune in next 2 iter

                self.avg_duration = None
                self.avg_count = 0

                self._comm.broadcast(next_point)
            print("[Tunner] exit point 3")
        print("[Tunner] _tune(): done")
            # point = {}
            # for k in current_point:
            #     if k == "step":
            #         continue

            #     # credit and partition is in unit of Million in tuning algorithm.
            #     point[k] = current_point[k] / float(10 ** 6)

            # self._tuner.put(point, avg_step_duration)
            # next_point, stop = self._tuner.step()
            # self.stopped = stop

            # if stop:
            #     all_points = self._tuner.get()
            #     for k, v in all_points.items():
            #         all_points[k] = float('%.3f' % (1 / v))
            #     self._logger.info("config steptime (steps/sec) map {}".format(all_points))

            # if next_point is not None:
            #     for k in next_point:
            #         # credit and partition is in unit of Million in tuning algorithm.
            #         next_point[k] = int(next_point[k] * (10**6))
            #     next_point["step"] = step + 10
            #     # broadcast
            #     self._comm.broadcast(next_point)
