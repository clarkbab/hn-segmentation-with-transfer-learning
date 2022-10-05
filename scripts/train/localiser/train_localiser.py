import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from hnas.training import train_localiser

if __name__ == '__main__':
    fire.Fire(train_localiser)
