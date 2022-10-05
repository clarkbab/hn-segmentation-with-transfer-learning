from collections import namedtuple
import os
import pandas as pd
from typing import Dict, List, Optional

from hnas import logging

DATA_VAR = 'HNAS_DATA'

class Directories:
    @property
    def models(self):
        filepath = os.path.join(self.root, 'models')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

    @property
    def datasets(self):
        filepath = os.path.join(self.root, 'datasets')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

    @property
    def files(self):
        filepath = os.path.join(self.root, 'files')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath
    
    @property
    def evaluations(self):
        filepath = os.path.join(self.root, 'evaluations')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

    @property
    def root(self):
        if DATA_VAR not in os.environ:
            raise ValueError(f"Must set env var '{DATA_VAR}'.")
        return os.environ[DATA_VAR]

    @property
    def runs(self):
        filepath = os.path.join(self.root, 'runs')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

    @property
    def temp(self):
        filepath = os.path.join(self.root, 'tmp')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

    @property
    def reports(self):
        filepath = os.path.join(self.root, 'reports')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

class Formatting:
    @property
    def metrics(self):
        return '.6f'

    @property
    def sample_digits(self):
        return 5

directories = Directories()
formatting = Formatting()
