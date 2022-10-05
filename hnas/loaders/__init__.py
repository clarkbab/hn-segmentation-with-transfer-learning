from typing import List, Optional, Union
from .loader import Loader

def get_n_train_max(
    dataset: Union[str, List[str]],
    region: str,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> int:
    tl, vl, _ = Loader.build_loaders(dataset, region, n_folds=n_folds, test_fold=test_fold)
    n_train = len(tl) + len(vl)
    return n_train

