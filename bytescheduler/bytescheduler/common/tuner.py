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
        # if partition_tuning:
        #     self._comm = create_comm(rank=self._rank)
        # else:
        #     self._comm = create_comm(rank=self._rank, host="localhost", port=58888)
        self._comm = create_comm(rank=self._rank)
        self._exit = False
        self._timestamps = []
        self._effective_point = None
        self.stopped = False

        self.avg_duration = None
        self.avg_count = 0
        self.last_duration_before_increased = None

        self.tune_thres = float(os.environ.get('BYTESCHEDULER_TUNE_THRES', 0.0))
        self.ss_thres = float(os.environ.get('SLOW_START_THRES', 1.5))
        self.collect_freq = int(os.environ.get('COLLECT_FREQ', 1))

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
        # print("[Tunner] record(): step = %d"%(step))
        """The scheduler will call this function at the beginning of each step,
        so that the tuner can collect timestamp information and derive feedback (i.e., duration of one training step).
        """
        if self._rank != 0:
            return
        if self._effective_point is None or self._effective_point == current_point:
            self._timestamps.append(time.time())

            # require at least 2 timestamps to calculating differences (durations)
            if len(self._timestamps) > self.collect_freq:
                self._tune(current_point, step)

    def exit(self):
        """Stop tuning."""
        self._comm.shutdown()

    def _tune(self, current_point, step):
        # print("[Tunner] _tune(): step = %d"%(step))
        """Run one step tuning."""
        step_durations = []
        
        # calculate durations
        for i in range(len(self._timestamps) - 1):
            step_durations.append(self._timestamps[i + 1] - self._timestamps[i])
        
        if step_durations:
            self._timestamps = []
            avg_step_duration = sum(step_durations) / len(step_durations)
            self.avg_count = self.avg_count + len(step_durations)
            if self.avg_duration is None:
                self.avg_duration = avg_step_duration
                self.avg_count = 1
            
            elif avg_step_duration <= (self.avg_duration * (1 + self.tune_thres)):
                # print("[Tunner] additive-increase")
                self.avg_duration = (self.avg_duration * self.avg_count + avg_step_duration * len(step_durations)) / (self.avg_count + len(step_durations))
                self.avg_count = self.avg_count + len(step_durations)

                next_point = current_point
                next_point["credit"] = next_point["credit"] + next_point["partition"] # additive-increase
                next_point["step"] = step + 1 # tune in next 2 iter
                self._comm.broadcast(next_point)
            
            elif avg_step_duration > (self.avg_duration * (1 + self.tune_thres)):
                if self.last_duration_before_increased == None:
                    self.last_duration_before_increased = self.avg_duration
                
                if avg_step_duration <= (self.last_duration_before_increased * (1 + self.ss_thres)):
                    # print("[Tunner] multiplicative-decrease")
                    next_point = current_point
                    next_point["credit"] = next_point["credit"] * self.avg_duration / avg_step_duration # multiplicative-decrease
                    if next_point["credit"] < next_point["partition"]:
                        next_point["credit"] = next_point["partition"]
                    next_point["step"] = step + 1 # tune in next iter

                    self.avg_duration = avg_step_duration
                    self.avg_count = 1

                    self._comm.broadcast(next_point)
                else:
                    print("[Tunner] slow-start-------------------------------------------------------------------")
                    self.last_duration_before_increased = None
                    # throughput degradation larger than ss_thres, possibly networking status change
                    next_point = current_point
                    next_point["credit"] = next_point["partition"] # slow-start
                    next_point["step"] = step + 1 # tune in next iter

                    self.avg_duration = avg_step_duration
                    self.avg_count = 1

                    self._comm.broadcast(next_point)

