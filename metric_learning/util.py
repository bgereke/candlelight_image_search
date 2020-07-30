# coding=utf-8





import sys

import numpy as np
import torch


class SimpleLogger(object):
    def __init__(self, logfile, terminal):
        ZERO_BUFFER_SIZE = 0  # immediately flush logs

        self.log = open(logfile, 'a')
        self.terminal = terminal

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()