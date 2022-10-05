from dataclasses import replace
from dicompylercore import dvhcalc
import os
import pandas as pd
import pydicom as dcm
from tqdm import tqdm
from typing import List, Optional, Union

from hnas import config
from hnas import dataset as ds
from hnas.dataset.dicom import RTSTRUCTConverter
from hnas.loaders import Loader
from hnas import logging
from hnas.metrics import dice
from hnas.models import replace_checkpoint_alias
from hnas.reporting.loaders import load_loader_manifest
from hnas import types
from hnas.utils import append_row, encode, load_csv, save_csv

def load_segmenter_dose_evaluation(
    datasets: Union[str, List[str]],
    localiser: types.ModelName,
    segmenter: types.ModelName,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_model_manifest: bool = False) -> pd.DataFrame:
    localiser = replace_checkpoint_alias(*localiser, use_manifest=use_model_manifest)
    segmenter = replace_checkpoint_alias(*segmenter, use_manifest=use_model_manifest)
    filepath = os.path.join(config.directories.evaluations, 'segmenter', *localiser, *segmenter, encode(datasets), f'dose-eval-folds-{n_folds}-test-{test_fold}.csv') 
    if not os.path.exists(filepath):
        raise ValueError(f"Segmenter dose evaluation for datasets '{datasets}', localiser '{localiser}' and segmenter '{segmenter}' not found.")
    data = pd.read_csv(filepath, dtype={'patient-id': str})
    return data

def create_segmenter_dose_evaluation_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_loader_manifest: bool = False,
    use_model_manifest: bool = False) -> None:
    localiser = replace_checkpoint_alias(*localiser, use_manifest=use_model_manifest)
    segmenter = replace_checkpoint_alias(*segmenter, use_manifest=use_model_manifest)
    logging.info(f"Creating dose evaluation for region '{region}', fold '{test_fold}', with localiser '{localiser}' and segmenter '{segmenter}'.")

    # Build test loader.
    if use_loader_manifest:
        man_df = load_loader_manifest(datasets, region, n_folds=n_folds, test_fold=test_fold)
        samples = man_df[['dataset', 'patient-id']].to_numpy()
    else:
        _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)
        test_dataset = test_loader.dataset
        samples = [test_dataset.__get_item(i) for i in range(len(test_dataset))]

    cols = {
        'fold': int,
        'dataset': str,
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    set_cache = {}
    for dataset, pat_id in tqdm(samples):
        # Get GT dose.
        if dataset in set_cache:
            set = set_cache[dataset]
        else:
            set_cache[dataset] = ds.get(dataset, 'dicom')
            set = set_cache[dataset]
        try:
            patient = set.patient(pat_id)
        except ValueError as e:
            logging.error(str(e))
            continue
        rtstruct_gt_path = patient.default_rtstruct.path
        rtdose_gt_path = patient.default_rtdose.path
        roi_info_gt = patient.default_rtstruct.get_region_info()
        roi_info_gt = dict((info['name'], id) for id, info in roi_info_gt.items())
        dose_gt = dvhcalc.get_dvh(rtstruct_gt_path, rtdose_gt_path, roi_info_gt[region])
        max_dose_gt = dose_gt.dose_constraint(0.03, volume_units='cc').value
        mean_dose_gt = dose_gt.mean

        # Get predicted dose.
        if config.environ('PETER_MAC_HACK') == 'True':
            base_path = 'S:\\ImageStore\\HN_AI_Contourer\\short\\dicom'
            if dataset == 'PMCC-HN-TEST':
                pred_path = os.path.join(base_path, 'test')
            elif dataset == 'PMCC-HN-TRAIN':
                pred_path = os.path.join(base_path, 'train')
        else:
            pred_path = os.path.join(set.path, 'predictions', 'segmenter')
        rtstruct_pred_path = os.path.join(pred_path, *localiser, *segmenter, f'{pat_id}.dcm')
        dose_pred = dvhcalc.get_dvh(rtstruct_pred_path, rtdose_gt_path, roi_info_gt[region])
        max_dose_pred = dose_pred.dose_constraint(0.03, volume_units='cc').value
        mean_dose_pred = dose_pred.mean

        # Get diff.
        metrics = {}
        max_dose_diff = max_dose_pred - max_dose_gt
        metrics['max-dose-diff'] = max_dose_diff
        mean_dose_diff = mean_dose_pred - mean_dose_gt
        metrics['mean-dose-diff'] = mean_dose_diff

        for metric, value in metrics.items():
            data = {
                'fold': test_fold,
                'dataset': dataset,
                'patient-id': pat_id,
                'region': region,
                'metric': metric,
                'value': value
            }
            df = append_row(df, data)
            
    df = df.astype(cols)
    filepath = os.path.join(config.directories.evaluations, 'segmenter', *localiser, *segmenter, encode(datasets), f'dose-eval-folds-{n_folds}-test-{test_fold}.csv')
    df.to_csv(filepath, index=False)

def create_dose_evaluation(
    pat_file: str,
    models: Union[str, List[str]],
    output_file: str) -> None:
    if type(models) == str:
        models = [models]

    # Load patients.
    pdf = load_csv(pat_file)

    # Get datasets.
    datasets = list(sorted(pdf.dataset.unique())) 

    # Convert regions to comma-delimited string.
    pdf = pdf.assign(region=pdf.groupby(['dataset', 'patient-id'])['region'].transform(','.join))
    pdf = pdf.drop(columns=['region'])
    pdf = pdf.drop_duplicates()

    # Get sets.
    sets = dict((d, ds.get(d, 'dicom')) for d in datasets)
    region_maps = dict((d, sets[d].region_map) for d in datasets)

    # Create dataframe.
    cols = {
        'dataset': str,
        'patient-id': str,
        'region': str,
        'rtstruct': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for i in tqdm(range(len(pdf))):
        # Get row.
        row = pdf.iloc[i]

        # Load ground truth RTSTRUCT. Catch exception when RTDOSE isn't present.
        try:
            patient_gt = sets[row.dataset].patient(row['patient-id'])
            rtstruct_gt = patient_gt.default_rtstruct
        except ValueError as e:
            logging.error(str(e))
            continue

        # Load ground truth map from region name to ROI number - predictions should have same mapping.
        info_gt = RTSTRUCTConverter.get_roi_info(rtstruct_gt.get_rtstruct())
        region_map_gt = region_maps[row.dataset]
        if region_map_gt is not None:
            info_gt = dict((region_map_gt.to_internal(name), int(id)) for id, name in info_gt)
        else:
            info_gt = dict((name, int(id)) for id, name in info_gt)

        # Load ground truth RTDOSE.
        rtdose_gt = patient_gt.default_rtdose
        assert rtdose_gt.get_rtdose().DoseUnits == 'GY'

        # load model RTSTRUCTs.
        rtstructs = [rtstruct_gt.get_rtstruct()]
        names = ['ground-truth']
        paths = [rtstruct_gt.path]
        for model in models:
            # Load model prediction.
            filepath = os.path.join(sets[row.dataset].path, 'predictions', model, f"{row['patient-id']}.dcm")
            rtstruct = dcm.read_file(filepath)
            rtstructs.append(rtstruct)
            names.append(model)
            paths.append(filepath)

        # Add dose metrics.
        for name, path, rtstruct in zip(names, paths, rtstructs):
            # Get ROI info. 
            info = RTSTRUCTConverter.get_roi_info(rtstruct)
            def to_internal(name):
                if region_maps[row.dataset] is None:
                    return name
                else:
                    return region_maps[row.dataset].to_internal(name)
            info = dict((to_internal(name), int(id)) for id, name in info)

            for region in row.regions.split(','):
                # Check region IDs.
                assert info[region] == info_gt[region]
                
                # Get DVH calcs.
                res = dvhcalc.get_dvh(path, rtdose_gt.path, info[region])
                
                # Add metrics.
                metrics = ['mean-dose', 'max-dose']
                metric_attrs = ['mean', 'max']
                for metric, metric_attr in zip(metrics, metric_attrs):
                    # Get value.
                    value = getattr(res, metric_attr)
                    
                    data = {
                        'dataset': row.dataset,
                        'patient-id': row['patient-id'],
                        'region': region,
                        'rtstruct': name,
                        'metric': metric,
                        'value': value
                    }
                    df = append_row(df, data)

    # Write evaluation.
    df = df.astype(cols)
    save_csv(df, 'dose-evals', output_file, overwrite=True)

def evaluate_model(
    dataset: str,
    localiser: types.Model,
    segmenter: types.Model,
    region: str) -> pd.DataFrame:
    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Evaluating on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Evaluating on CPU...')

    # Load dataset.
    set = ds.get(dataset, 'dicom')
    pats = set.list_patients(region=region)

    # Load model if not already loaded.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    if type(segmenter) == tuple:
        segmenter = Segmenter.load(*segmenter)

    # Create dataframe.
    cols = {
        'patient-id': str,
        'region': str,
        'metric': str
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat in tqdm(pats):
        # Get pred/ground truth.
        pred = get_two(set, pat, localiser, segmenter, device=device)
        label = set.patient(pat).region_data()[region]

        # Add metrics.
        dsc_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'dice'
        }
        hd_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'hausdorff'
        }
        hd_avg_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'average-hausdorff'
        }
        sd_avg_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'average-surface'
        }
        sd_med_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'median-surface'
        }
        sd_std_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'std-surface'
        }
        sd_max_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'max-surface'
        }

        # Dice.
        dsc_score = dice(pred, label)
        dsc_data[region] = dsc_score
        df = df.append(dsc_data, ignore_index=True)

        # Hausdorff.
        spacing = set.patient(pat).ct_spacing()
        hd, hd_avg = hausdorff_distance(pred, label, spacing)
        hd_data[region] = hd
        hd_avg_data[region] = hd_avg
        df = append_row(df, hd_data)
        df = append_row(df, hd_avg_data)

        # Symmetric surface distance.
        sd_mean, sd_median, sd_std, sd_max = symmetric_surface_distance(pred, label, spacing)
        sd_mean_data[region] = sd_mean
        sd_median_data[region] = sd_median
        sd_std_data[region] = sd_std
        sd_max_data[region] = sd_max
        df = append_row(df, sd_mean)
        df = append_row(df, sd_median)
        df = append_row(df, sd_std)
        df = append_row(df, sd_max)

    # Set index.
    df = df.set_index('patient-id')

    return df
