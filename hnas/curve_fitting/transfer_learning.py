import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from hnas.regions.tolerances import get_region_tolerance
import numpy as np
import os
import pandas as pd
from scipy.optimize import least_squares
import seaborn as sns
from tqdm import tqdm
from typing import Callable, List, Literal, Optional, Tuple, Union

from hnas import config
from hnas.evaluation.dataset.nifti import load_segmenter_evaluation_from_loader
from hnas.loaders import Loader, get_loader_n_train
from hnas import logging
from hnas.regions import RegionNames

DEFAULT_FONT_SIZE = 15
DEFAULT_MAX_NFEV = int(1e6)
DEFAULT_METRIC_LEGEND_LOCS = {
    'dice': 'lower right',
    'hd': 'upper right',
    'hd-95': 'upper right',
    'msd': 'upper right'
}
DEFAULT_METRIC_LABELS = {
    'dice': 'DSC',
    'hd': 'HD [mm]',
    'hd-95': '95HD [mm]',
    'msd': 'MSD [mm]',
}
DEFAULT_METRIC_Y_LIMS = {
    'dice': (0, 1),
    'hd': (0, None),
    'hd-95': (0, None),
    'msd': (0, None)
}
for region in RegionNames:
    tol = get_region_tolerance(region)
    DEFAULT_METRIC_LEGEND_LOCS[f'apl-mm-tol-{tol}'] = 'upper right'
    DEFAULT_METRIC_LEGEND_LOCS[f'dm-surface-dice-tol-{tol}'] = 'lower right'
    DEFAULT_METRIC_LABELS[f'apl-mm-tol-{tol}'] = fr'APL, $\tau$={tol}mm'
    DEFAULT_METRIC_LABELS[f'dm-surface-dice-tol-{tol}'] = fr'Surface DSC, $\tau$={tol}mm'
    DEFAULT_METRIC_Y_LIMS[f'apl-mm-tol-{tol}'] = (0, None)
    DEFAULT_METRIC_Y_LIMS[f'dm-surface-dice-tol-{tol}'] = (0, 1)

DEFAULT_N_SAMPLES = int(1e4)
LOG_SCALE_X_UPPER_LIMS = [100, 150, 200, 300, 400, 600, 800]
LOG_SCALE_X_TICK_LABELS = [5, 10, 20, 50, 100, 200, 400, 800]

def __bootstrap_n_train_sample(x, n_samples, seed=42):
    np.random.seed(seed)
    return np.random.choice(x, size=(len(x), n_samples), replace=True)

def create_bootstrap_predictions(
    region: str,
    model: str,
    metric: str, 
    stat: str,
    samples: np.ndarray,
    n_trains: Union[int, List[int]],
    raise_error: bool = True,
    weights: bool = False) -> None:
    logging.info(f"Creating bootstrap predictions for metric '{metric}', model '{model}', stat '{stat}', n_trains '{n_trains}', region '{region}'...")

    # Get placeholders.
    n_preds = np.max(n_trains) + 1
    n_samples = len(samples)
    n_params = 3
    params = np.zeros((n_samples, n_params))
    preds = np.zeros((n_samples, n_preds))

    # Get weights.
    weights = 1 / np.var(samples, axis=0).mean(axis=1) if weights else None

    for i in tqdm(range(n_samples)):
        sample = samples[i, :, :]
        
        # Flatten data.
        x = np.array([])
        y = np.array([])
        w = np.array([]) if weights is not None else None
        for j, (n_train, n_train_sample) in enumerate(zip(n_trains, sample)):
            x = np.concatenate((x, n_train * np.ones(len(n_train_sample))))
            y = np.concatenate((y, n_train_sample))
            if weights is not None:
                w = np.concatenate((w, weights[j] * np.ones(len(n_train_sample))))
            
        # Fit curve.
        try:
            p_init = get_p_init_v2(metric)
            p_opt, _, _ = fit_curve(p_init, x, y, weights=w)
        except ValueError as e:
            if raise_error:
                logging.error(f"Error when fitting sample '{i}':")
                raise e
            else:
                return x, y, w
        params[i] = p_opt
        
        # Create prediction points.
        x = np.linspace(0, n_preds - 1, num=n_preds)
        y = f(x, p_opt)
        preds[i] = y

    # Set 'clinical' model predictions to 'NaN' as 'clinical' model must be trained on
    # 5 or more training cases.
    if model == 'clinical':
        preds[:, :5] = np.nan
        
    # Save data.
    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'curve-fitting', 'bootstrap', 'preds', model, metric, stat)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=preds, params=params)
    logging.info('Bootstrap predictions created.')

def create_bootstrap_samples(
    region: str, 
    model: str, 
    metric: str, 
    stat: str,
    data: pd.DataFrame, 
    n_samples: int = DEFAULT_N_SAMPLES) -> Tuple[np.ndarray, List[int]]:
    logging.info(f"Creating bootstrap samples for region '{region}', model '{model}', metric '{metric}', stat '{stat}', n_samples '{n_samples}'...")

    # Bootstrap each 'n_train=...' sample to create a 3D array of 'n_samples' samples for each 'n_train'.
    boot_df = data[(data.metric == metric) & (data['model-type'] == model) & (data.region == region)]
    boot_df = boot_df.pivot(index=['region', 'model-type', 'n-train', 'metric'], columns='fold', values='value')
    boot_data = np.moveaxis(np.apply_along_axis(lambda x: __bootstrap_n_train_sample(x, n_samples), arr=boot_df.values, axis=1), 2, 0)
    n_trains = boot_df.reset_index()['n-train'].values
    
    # Save data.
    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'curve-fitting', 'bootstrap', 'samples', model, metric, stat)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=boot_data, n_trains=n_trains)
    logging.info(f'Bootstrap samples created.')

    return boot_data, list(n_trains)

def create_bootstrap_samples_and_predictions(
    datasets: List[str],
    region: str,
    models: Union[str, List[str]],
    metrics: Union[str, List[str]],
    stats: Union[str, List[str]],
    n_trains: Union[Optional[int], List[Optional[int]]] = [5, 10, 20, 50, 100, 200, None],
    n_folds: int = 5,
    n_samples: int = DEFAULT_N_SAMPLES,
    test_folds: Union[int, List[int]] = list(range(5))) -> None:
    metrics = [metrics] if type(metrics) == str else metrics
    models = [models] if type(models) == str else models
    stats = [stats] if type(stats) == str else stats
    test_folds = [test_folds] if type(test_folds) == int else test_folds

    # Load data.
    _, mean_df, min_df, max_df = load_evaluation(datasets, region, models, n_trains, n_folds=n_folds, test_folds=test_folds)

    # Create samples and prediction from curve fitting.
    for metric in metrics:
        for model in models:
            if 'mean' in stats:
                samples, n_trains = create_bootstrap_samples(region, model, metric, 'mean', mean_df, n_samples=n_samples)
                create_bootstrap_predictions(region, model, metric, 'mean', samples, n_trains)
            if 'min' in stats:
                samples, n_trains = create_bootstrap_samples(region, model, metric, 'min', min_df, n_samples=n_samples)
                create_bootstrap_predictions(region, model, metric, 'min', samples, n_trains)
            if 'max' in stats:
                samples, n_trains = create_bootstrap_samples(region, model, metric, 'max', max_df, n_samples=n_samples)
                create_bootstrap_predictions(region, model, metric, 'max', samples, n_trains)

def f(
    x: np.ndarray,
    params: Tuple[float]) -> Union[float, List[float]]:
    return -params[0] / (x - params[1]) + params[2]

def get_p_init_v2(metric: str) -> Tuple[float, float, float]:
    if 'apl-mm-tol-' in metric:
        return (-1, -1, 1)
    if 'dm-surface-dice-tol-' in metric:
        return (1, -1, 1)
    metric_p_inits = {
        'dice': (1, -1, 1),
        'hd': (-1, -1, 1),
        'hd-95': (-1, -1, 1),
        'msd': (-1, -1, 1)
    }
    return metric_p_inits[metric]

def get_p_init(
    x: List[float],
    y: List[float],
    y_mid_ratio: float = 1e-1) -> Tuple[float]:
    # Get min/max points.
    x_min = np.min(x)
    y_min = np.array(y)[np.equal(x, x_min)].mean()
    x_max = np.max(x)
    y_max = np.array(y)[np.equal(x, x_max)].mean()

    # Fit a line curved slightly so that we're fitting quadrant 2 or 3.
    x_mid = (x_max + x_min) / 2
    y_mid = y_max - y_mid_ratio * (y_max - y_min)
        
    # Fit a curve to the points.
    x = [x_min, x_mid, x_max]
    y = [y_min, y_mid, y_max]
    params = fit_curve_to_points(x, y)

    return params

def fit_curve(
    p_init: Tuple[float],
    x: List[float],
    y: List[float], 
    max_nfev: int = DEFAULT_MAX_NFEV, 
    raise_error: bool = True, 
    weights: Optional[List[float]] = None):
    # Make fit.
    x_min = np.min(x)
    result = least_squares(residuals(f), p_init, args=(x, y, weights), bounds=((-np.inf, -np.inf, 0), (np.inf, x_min, np.inf)), max_nfev=max_nfev)

    # Check fit status.
    status = result.status
    if raise_error and status < 1:
        raise ValueError(f"Curve fit failed")

    # Check final parameter values.
    p_opt = result.x
    if raise_error: 
        if p_opt[1] >= x_min:
            raise ValueError(f"Vertical asymp. position 'x={p_opt[1]:.3f}' was right of minimum x value 'x={x_min}'. Bad 'p_init'?")
        elif p_opt[2] < 0:
            raise ValueError(f"Horizontal asymp. position 'y={p_opt[2]:.3f}' was below zero.")

    return p_opt, result.cost, result.jac

def fit_curve_to_points(
    x: List[float],
    y: List[float]) -> Tuple[float, float, float]:
    assert len(x) == 3 and len(y) == 3
    
    # Solve for b, c.
    A = np.array([
        [y[1] - y[0], x[1] - x[0]],
        [y[2] - y[0], x[2] - x[0]]
    ])
    b = np.array([x[1] * y[1] - x[0] * y[0], x[2] * y[2] - x[0] * y[0]])
    b, c = np.linalg.solve(A, b)
    
    # Solve for a.
    a = -b * c + y[0] * b + x[0] * c - x[0] * y[0]
    
    return a, b, c

def get_bootstrap_statistic(
    region: str,
    model: str,
    metric: str, 
    stat: str,
    n_train: int,
    n_samples: int = DEFAULT_N_SAMPLES):
    # Get data for 'n_train'.
    data, _ = load_bootstrap_predictions(region, model, metric, stat, n_samples=n_samples)
    data_n_train = data[:, n_train]

    # Calculate statistic.
    if stat == 'mean':
        value = data_n_train.mean()
    elif stat == 'min':
        value = data_n_train.min()
    elif stat == 'max':
        value = data_n_train.max()
    
    return value

def load_bootstrap_predictions(
    region: str,
    model: str,
    metric: str, 
    stat: str,
    n_samples: int = DEFAULT_N_SAMPLES):
    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'curve-fitting', 'bootstrap', 'preds', model, metric, stat)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    f = np.load(filepath)
    data = f['data']
    params = f['params']
    return data, params

def load_bootstrap_samples(
    region: str,
    model: str,
    metric: str,
    stat: str, 
    n_samples: int = DEFAULT_N_SAMPLES):
    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'curve-fitting', 'bootstrap', 'samples', model, metric, stat)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    f = np.load(filepath)
    data = f['data']
    n_trains = f['n_trains']
    return data, n_trains

def load_evaluation(
    datasets: Union[str, List[str]],
    regions: Union[str, List[str]],
    models: Union[str, List[str]],
    n_trains: Union[Optional[int], List[Optional[int]]],
    n_folds: int = 5,
    test_folds: Union[int, List[int]] = list(range(5))) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = [datasets] if type(datasets) == str else datasets
    models = [models] if type(models) == str else models
    n_trains = [n_trains] if type(n_trains) == str else n_trains
    regions = [regions] if type(regions) == str else regions
    test_folds = [test_folds] if type(test_folds) == str else test_folds

    # Load evaluations and combine.
    dfs = []
    for region in regions:
        for test_fold in test_folds:
            # Add public evaluation.
            localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'BEST')
            seg_run = 'public-1gpu-150epochs'
            segmenter = (f'segmenter-{region}', seg_run, 'BEST')
            df = load_segmenter_evaluation_from_loader(datasets, localiser, segmenter, n_folds, test_fold)
            df['model-type'] = f'transfer-fold-{test_fold}-samples-0'
            dfs.append(df)

            # Get number of training cases.
            n_train_max = get_loader_n_train(datasets, region, n_folds=n_folds, test_fold=test_fold)

            # Add clinical/transfer evaluations.
            for model in (model for model in models if model != 'public'):
                for n_train in n_trains:
                    # Skip if we've exceeded available number of training samples.
                    if n_train is not None and n_train > n_train_max:
                        continue

                    seg_run = f'{model}-fold-{test_fold}-samples-{n_train}'
                    segmenter = (f'segmenter-{region}', seg_run, 'BEST')
                    df = load_segmenter_evaluation_from_loader(datasets, localiser, segmenter, n_folds, test_fold)
                    df['model-type'] = seg_run
                    dfs.append(df)
                   
    # Save dataframe.
    df = pd.concat(dfs, axis=0)

    # Add num-train (same for each fold).
    df = df.assign(**{ 'n-train': df['model-type'].str.split('-').apply(lambda x: x[-1]) })
    none_nums = {}
    for region in regions:
        tl, vl, _ = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=0)
        n_train = len(tl) + len(vl)
        none_nums[region] = n_train
    df.loc[df['n-train'] == 'None', 'n-train'] = df[df['n-train'] == 'None']['region'].apply(lambda r: none_nums[r])
    df['n-train'] = df['n-train'].astype(int)

    # Replace model names.
    df['model-type'] = df['model-type'].str.split('-').apply(lambda l: l[0])
    df['model'] = df['model-type'] + '-' + df['n-train'].astype(str)

    # Sort values.
    df = df.sort_values(['fold', 'region', 'model-type', 'n-train'])

    # Add region counts.
    # count_df = df.groupby(['fold', 'region', 'model-type', 'metric'])['patient-id'].count().reset_index()

    # Get mean values.
    mean_df = df.groupby(['fold', 'region', 'model-type', 'n-train', 'metric'])['value'].mean().reset_index()

    # Get min values.
    min_df = df.groupby(['fold', 'region', 'model-type', 'n-train', 'metric'])['value'].quantile(0.05).reset_index()
    max_df = df.groupby(['fold', 'region', 'model-type', 'n-train', 'metric'])['value'].quantile(0.95).reset_index()

    return df, mean_df, min_df, max_df

def megaplot(
    regions: Union[str, List[str]],
    models: Union[str, List[str]],
    metrics: Union[str, List[str], np.ndarray],
    stats: Union[str, List[str]],
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    fontsize: float = DEFAULT_FONT_SIZE,
    inner_wspace: float = 0.05,
    legend_locs: Optional[Union[str, List[str]]] = None,
    model_labels: Optional[Union[str, List[str]]] = None,
    outer_hspace: float = 0.2,
    outer_wspace: float = 0.2,
    savepath: Optional[str] = None,
    y_lim: bool = True) -> None:
    if type(metrics) == str:
        metrics = np.repeat([[metrics]], len(regions), axis=0)
    elif type(metrics) == list:
        metrics = np.repeat([metrics], len(regions), axis=0)
    n_metrics = metrics.shape[1]
    dfs = [dfs] if type(dfs) == pd.DataFrame else dfs
    dfs = dfs * n_metrics if len(dfs) == 1 else dfs         # Broadcast 'dfs' to 'n_metrics'.
    if legend_locs is not None:
        if type(legend_locs) == str:
            legend_locs = [legend_locs]
        assert len(legend_locs) == n_metrics
    models = [models] if type(models) == str else models
    n_models = len(models)
    if model_labels is not None:
        if type(model_labels) == str:
            model_labels = [model_labels]
        assert len(model_labels) == n_models
    regions = [regions] if type(regions) == str else regions
    n_regions = len(regions)
    stats = [stats] if type(stats) == str else stats
    stats = stats * n_metrics if len(stats) == 1 else stats         # Broadcast 'stats' to 'n_metrics'.

    # Lookup tables.
    # matplotlib.rc('text', usetex=True)
    # matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Create nested subplots.
    fig = plt.figure(constrained_layout=False, figsize=(6 * metrics.shape[1], 6 * n_regions))
    outer_gs = fig.add_gridspec(hspace=outer_hspace, nrows=n_regions, ncols=metrics.shape[1], wspace=outer_wspace)
    for i, region in enumerate(regions):
        for j in range(n_metrics):
            df = dfs[j]
            metric = metrics[i, j]
            stat = stats[j]
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[i, j], width_ratios=[1, 19], wspace=0.05)
            axs = [plt.subplot(cell) for cell in inner_gs]
            y_lim = DEFAULT_METRIC_Y_LIMS[metric] if y_lim else (None, None)
            legend_loc = legend_locs[j] if legend_locs is not None else DEFAULT_METRIC_LEGEND_LOCS[metric]
            plot_bootstrap_fit(region, models, metric, stat, df, axs=axs, fontsize=fontsize, legend_loc=legend_loc, model_labels=model_labels, split=True, wspace=inner_wspace, x_scale='log', y_label=DEFAULT_METRIC_LABELS[metric], y_lim=y_lim)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)

    plt.show()

def plot_bootstrap_fit(
    region: str, 
    models: Union[str, List[str]],
    metric: str,
    stat: str,
    df: pd.DataFrame,
    alpha_ci: float = 0.2,
    alpha_points: float = 1.0,
    axs: Optional[Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]]] = None,
    figsize: Tuple[float, float] = (8, 6),
    fontsize: float = DEFAULT_FONT_SIZE,
    fontweight: Literal['normal', 'bold'] = 'normal',
    index: Optional[int] = None,
    legend_loc: Optional[str] = None,
    model_labels: Optional[Union[str, List[str]]] = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    show_ci: bool = True,
    show_limits: bool = False,
    show_points: bool = True,
    split: bool = True,
    title: str = '',
    wspace: float = 0.05,
    x_scale: str = 'log',
    y_label: str = '',
    y_lim: Optional[Tuple[float, float]] = None):
    colours = sns.color_palette('colorblind')[:len(models)]
    legend_loc = DEFAULT_METRIC_LEGEND_LOCS[metric] if legend_loc is None else legend_loc
    models = [models] if type(models) == str else models
    if model_labels is not None:
        model_labels = [model_labels] if type(model_labels) == str else model_labels
        assert len(model_labels) == len(models)
        
    if axs is None:
        plt.figure(figsize=figsize)
        if split:
            _, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 19]})
        else:
            axs = [plt.gca()]
    
    for ax in axs:
        # Plot data.
        for i, model in enumerate(models):
            colour = colours[i]
            model_label = model_labels[i] if model_labels is not None else model

            if index is not None:
                # Load bootstrapped sample.
                samples, n_trains = load_bootstrap_samples(region, model, metric, stat, n_samples=n_samples)
                sample = samples[index]
                x_raw = np.array([])
                y_raw = np.array([])
                for n_train, n_train_sample in zip(n_trains, sample):
                    x_raw = np.concatenate((x_raw, n_train * np.ones(len(n_train_sample))))
                    y_raw = np.concatenate((y_raw, n_train_sample))

                # Load prediction on bootstrapped sample.
                preds, _ = load_bootstrap_predictions(region, model, metric, stat, n_samples=n_samples)
                pred = preds[index]

                # Plot.
                x = np.linspace(0, len(pred) - 1, num=len(pred))
                ax.plot(x, pred, color=colour, label=model_label)
                if show_points:
                    ax.scatter(x_raw, y_raw, color=colour, marker='o', alpha=alpha_points)
            else:
                # Load raw data and bootstrapped predictions.
                x_raw, y_raw = raw_data(region, model, metric, df)
                preds, _ = load_bootstrap_predictions(region, model, metric, stat, n_samples=n_samples)

                # Calculate means.
                means = preds.mean(axis=0)

                # Plot.
                x = np.linspace(0, len(means) - 1, num=len(means))
                ax.plot(x, means, color=colour, label=model_label)
                if show_ci:
                    low_ci = np.quantile(preds, 0.025, axis=0)
                    high_ci = np.quantile(preds, 0.975, axis=0)
                    ax.fill_between(x, low_ci, high_ci, color=colour, alpha=alpha_ci)
                if show_limits:
                    min = preds.min(axis=0)
                    max = preds.max(axis=0)
                    ax.plot(x, min, c='black', linestyle='--', alpha=0.5)
                    ax.plot(x, max, c='black', linestyle='--', alpha=0.5)
                if show_points:
                    ax.scatter(x_raw, y_raw, color=colour, marker='o', alpha=alpha_points)

    # Set axis scale.
    if split:
        axs[1].set_xscale(x_scale)
    else:
        axs[0].set_xscale(x_scale)

    # Set x tick labels.
    x_upper_lim = None
    if split:
        axs[0].set_xticks([0])
        if x_scale == 'log':
            x_upper_lim = axs[1].get_xlim()[1]      # Record x upper lim as setting ticks overrides this.
            axs[1].set_xticks(LOG_SCALE_X_TICK_LABELS)
            axs[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Set axis limits.
    if split:
        axs[0].set_xlim(-0.5, 0.5)
        axs[1].set_xlim(4.5, x_upper_lim)
        # if x_scale == 'log':
        #     axs[1].set_xlim(4.5, x_upper_lim)
        # else:
        #     axs[1].set_xlim(4.5, None)

    # Set y label.
    axs[0].set_ylabel(y_label, fontsize=fontsize, weight=fontweight)

    # Set y limits.
    axs[0].set_ylim(y_lim)
    if split:
        axs[1].set_ylim(y_lim)

    # Set y tick label fontsize.
    axs[0].tick_params(axis='x', which='major', labelsize=fontsize)
    axs[0].tick_params(axis='y', which='major', labelsize=fontsize)
    if split:
        axs[1].tick_params(axis='x', which='major', labelsize=fontsize)

    # Set title.
    title = title if title else region
    if split:
        axs[1].set_title(title, fontsize=fontsize, weight=fontweight)
    else:
        axs[0].set_title(title, fontsize=fontsize, weight=fontweight)

    # Add legend.
    if split:
        axs[1].legend(fontsize=fontsize, loc=legend_loc)
    else:
        axs[0].legend(fontsize=fontsize, loc=legend_loc)

    if split:
        # Hide axes' spines.
        axs[0].spines['right'].set_visible(False)
        axs[1].spines['left'].set_visible(False)
        axs[1].set_yticks([])

        # Add split between axes.
        plt.subplots_adjust(wspace=wspace)
        d_x_0 = .1
        d_x_1 = .006
        d_y = .03
        kwargs = dict(transform=axs[0].transAxes, color='k', clip_on=False)
        axs[0].plot((1 - (d_x_0 / 2), 1 + (d_x_0 / 2)), (-d_y / 2, d_y / 2), **kwargs)  # bottom-left diagonal
        axs[0].plot((1 - (d_x_0 / 2), 1 + (d_x_0 / 2)), (1 - (d_y / 2), 1 + (d_y / 2)), **kwargs)  # top-left diagonal
        kwargs = dict(transform=axs[1].transAxes, color='k', clip_on=False)
        axs[1].plot((-d_x_1 / 2, d_x_1 / 2), (-d_y / 2, d_y / 2), **kwargs)  # bottom-left diagonal
        axs[1].plot((-d_x_1 / 2, d_x_1 / 2), (1 - (d_y / 2), 1 + (d_y / 2)), **kwargs)  # top-left diagonal

def raw_data(
    region: str,
    model: str,
    metric: str,
    df: pd.DataFrame) -> np.ndarray:
    data_df = df[(df.metric == metric) & (df['model-type'] == model) & (df.region == region)]
    x = data_df['n-train']
    y = data_df['value']
    return x, y

def residuals(f):
    def inner(params, x, y, weights):
        y_pred = f(x, params)
        rs = y - y_pred
        if weights is not None:
            rs *= weights
        return rs
    return inner
