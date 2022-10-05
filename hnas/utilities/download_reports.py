import os
from typing import List

from hnas import dataset as ds

def download_reports(datasets: List[str]) -> None:
    for dataset in datasets:
        set = ds.get(dataset, 'nifti')
        report_path = os.path.join(set.path, 'reports')
        dest_path = f'/mnt/c/Users/BACLARK/Documents/Research/Topics/Transfer\ learning/Reports/{dataset}'
        os.makedirs(dest_path, exist_ok=True)
        cmd = f'scp spartan:{report_path} {dest_path}'
        os.system(cmd)
