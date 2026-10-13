"""Reproduce all four experiments from any working directory."""
from pathlib import Path
import shutil
import pandas as pd
from src.data_loader import load_election, election_summary, generate_regression
from src import problem_2_7, problem_2_8, problem_2_11, problem_2_12
from src.reporting import write_report

ROOT = Path(__file__).resolve().parent


def main():
    data_dir = ROOT/'data'
    tables, figures = ROOT/'results'/'tables', ROOT/'results'/'figures'
    for directory in (data_dir, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)
    election = {}
    for year in (2000, 2004):
        name = f'obesity_election_{year}.csv'
        source = ROOT.parent/name
        if source.exists():
            shutil.copyfile(source, data_dir/name)
        election[year] = load_election(data_dir/name)
    if set(election[2000].State_Code) != set(election[2004].State_Code):
        raise ValueError('State codes differ across years')
    statistics = pd.DataFrame([election_summary(election[year], year) for year in (2000, 2004)])
    statistics.to_csv(tables/'election_data_summary.csv', index=False)
    knn_results = [problem_2_7.run(election, tables, figures), problem_2_8.run(election, tables, figures)]
    knn_summary = pd.DataFrame([result[0] for result in knn_results])
    knn_summary.to_csv(tables/'knn_comparison.csv', index=False)
    regression = generate_regression()
    regression.to_csv(data_dir/'generated_example_2_6.csv', index=False)
    polynomial, poly_summary = problem_2_11.run(regression, tables, figures)
    folds, cv_summary, _ = problem_2_12.run(regression, tables, figures)
    write_report(ROOT/'report.md', statistics, knn_results, polynomial, poly_summary, folds, cv_summary)
    print('\nElection data validation and descriptive statistics\n', statistics.to_string(index=False))
    print('\nProblems 2.7 / 2.8 (error rates are proportions)\n', knn_summary.to_string(index=False))
    print('\nProblem 2.11\n', poly_summary.to_string(index=False))
    print('\nProblem 2.12 (MSE in squared y units)\n', cv_summary.to_string(index=False))
    print('\nGenerated report.md and results/tables, results/figures')


if __name__ == '__main__':
    main()
