import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from hnas.evaluation.dataset.nifti import create_segmenter_evaluation

fire.Fire(create_segmenter_evaluation)

# Sample args:
# --dataset PMCC-HN-TEST --region Parotid_L --segmenter "(...)"
