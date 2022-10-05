from fpdf import FPDF
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from hnas import config
from hnas.dataset.dicom import DicomDataset
from hnas import plotting
from hnas.utils import filterOnPatIDs
from hnas import types

def generate_dataset_regions_report(
    dataset: str,
    pat_ids: types.PatientIDs = 'all',
    regions: types.PatientRegions = 'all',
    report_name: str = None) -> None:
    """
    effect: Generates a PDF report of dataset segmentations.
    args:
        dataset: the dataset name.
    kwargs:
        pat_ids: the patients to show.
        regions: the regions to show.
        report_name: the name of the report.
    """
    # Get patients.
    ds = DicomDataset(dataset)
    pats = ds.list_patients()

    # Filter patients.
    pats = list(filter(filterOnPatIDs(pat_ids), pats))
    pats = list(filter(ds.filterOnRegions(regions), pats))

    # Get regions.
    if regions == 'all':
        regions = ds.list_regions().region.unique() 
    elif isinstance(regions, str):
        regions = [regions]

    # Create PDF.
    report = FPDF()
    report.set_font('Arial', 'B', 16)

    for region in tqdm(regions):
        for pat in tqdm(pats, leave=False):
            # Skip if patient doesn't have region.
            if not ds.patient(pat).has_region(region):
                continue

            # Add patient/region title.
            report.add_page()
            text = f"Region: {region}, Patient: {pat}"
            report.cell(0, 0, text, ln=1)

            # Get region centroid.
            summary = ds.patient(pat).region_summary(clear_cache=clear_cache, region=region).iloc[0].to_dict()
            centroid = (int(summary['centroid-voxels-x']), int(summary['centroid-voxels-y']), int(summary['centroid-voxels-z']))

            # Save orthogonal plots.
            views = ['sagittal', 'coronal', 'axial']
            origins = ((0, 20), (100, 20), (0, 120))
            for c, o, v in zip(centroid, origins, views):
                # Set figure.
                plotting.plot_patient_regions(pat, c, region=region, show=False, view=v)

                # Save temp file.
                filename = f"patient-{pat}-region-{region}-view-{v}.png"
                filepath = os.path.join(config.directories.temp, filename)
                plt.savefig(filepath)

                # Add image to report.
                report.image(filepath, *o, w=100, h=100)

                # Delete temp file.
                os.remove(filepath)

    # Save PDF.
    if report_name:
        filename = report_name
    else:
        filename = f"report-{dataset}.pdf"
    filepath = os.path.join(config.directories.files, filename) 
    report.output(filepath, 'F')
