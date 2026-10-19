"""Publish the reviewed report alongside the generated experiment artifacts."""
from pathlib import Path
import shutil


def write_report(path, statistics, knn_results, polynomial, poly_summary, folds, cv_summary):
    """Copy the report template after the experiment has produced its figures and tables."""
    template = Path(__file__).with_name('report_template.md')
    shutil.copyfile(template, path)
