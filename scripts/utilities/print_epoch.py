import os
import sys
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

from hnas import config

region = sys.argv[1]
run = sys.argv[2]
model = (f'segmenter-{region}', run, 'last.ckpt')
path = os.path.join(config.directories.models, *model)
state = torch.load(path, map_location=torch.device('cpu'))

print(f"""
Model: {model}
Epoch: {state['epoch']}
""")
