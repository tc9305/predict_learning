"""Test mathematical definitions, data isolation and reproducibility."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_election, generate_regression
from src.polynomial_models import design_matrix, select_polynomial
from src.schwartz_criterion import schwartz
from src import knn_experiment, problem_2_7, problem_2_8, problem_2_12


@pytest.mark.parametrize('year', [2000, 2004])
def test_supplied_election_data(year):
    data = load_election(ROOT/'data'/f'obesity_election_{year}.csv')
    assert len(data) == 51
    assert data.iloc[0].State == 'Alabama'
    assert data.iloc[0].Obesity == {2000: 22.6, 2004: 27.7}[year]
    assert not data.isna().any().any()
    assert data.State_Code.eq('DC').sum() == 1
    if year == 2004:
        assert data.loc[data.State.eq('Hawaii'), 'Election_Result'].item() == 'R'


@pytest.mark.parametrize('fault', ['missing', 'duplicate_state', 'duplicate_code', 'percentage', 'label', 'row_count'])
def test_invalid_data_is_rejected(tmp_path, fault):
    data = pd.read_csv(ROOT/'data/obesity_election_2000.csv', header=None)
    if fault == 'missing': data.iloc[0, 0] = np.nan
    if fault == 'duplicate_state': data.iloc[0, 0] = data.iloc[1, 0]
    if fault == 'duplicate_code': data.iloc[0, 3] = data.iloc[1, 3]
    if fault == 'percentage': data.iloc[0, 1] = 'invalid%'
    if fault == 'label': data.iloc[0, 2] = 'X'
    if fault == 'row_count': data = data.iloc[1:]
    file = tmp_path/'invalid.csv'
    data.to_csv(file, index=False, header=False)
    with pytest.raises(ValueError): load_election(file)


def test_regression_generator_and_shared_sample():
    data = generate_regression()
    rng = np.random.default_rng(42)
    x, noise = rng.uniform(0, 1, 10), rng.normal(0, 0.5, 10)
    assert len(data) == 10
    np.testing.assert_array_equal(data.x, x)
    np.testing.assert_array_equal(data.noise, noise)
    np.testing.assert_array_equal(data.y, x*x+0.1*x+noise)
    saved = pd.read_csv(ROOT/'data/generated_example_2_6.csv')
    np.testing.assert_allclose(saved.y, data.y)
    oof = pd.read_csv(ROOT/'results/tables/problem_2_12_oof_predictions.csv')
    for family in ['trigonometric', 'algebraic']:
        np.testing.assert_allclose(oof[oof.family.eq(family)].sort_values('sample_id').actual_y, data.y)


def test_exact_basis_definitions():
    x = np.array([0.0, 0.13, 0.5, 1.0])
    a = design_matrix(x, 3, 'trigonometric')
    assert a.shape == (4, 4)
    np.testing.assert_allclose(a[:, 0], 1)
    for i in range(1, 4): np.testing.assert_allclose(a[:, i], np.cos(2*np.pi*i*x))
    b = design_matrix(x, 3, 'algebraic')
    for i in range(4): np.testing.assert_allclose(b[:, i], x**i)
    assert design_matrix(x, 0, 'trigonometric').shape == (4, 1)


def test_schwartz_formula_and_bounds():
    assert schwartz(0.25, 3, 10) == pytest.approx(0.25*(1+0.3*np.log(10)/0.7))
    for dof in [0, 10, 11]:
        with pytest.raises(ValueError): schwartz(0.25, dof, 10)


def test_ols_selection_range_ties_and_quadratic_recovery():
    x = np.linspace(0, 1, 10)
    for family in ['trigonometric', 'algebraic']:
        model, table = select_polynomial(x, np.zeros(10), family)
        assert model.m == 0  # every score is exactly zero
        assert list(table.m) == list(range(9))
        assert (table.DoF < table.n_train).all()
    from src.polynomial_models import fit_polynomial
    model = fit_polynomial(x, 2+3*x+4*x*x, 2, 'algebraic')
    np.testing.assert_allclose(model.coefficients, [2, 3, 4], atol=1e-12)


def test_loo_fits_exclude_each_validation_observation(monkeypatch):
    x = np.arange(5, dtype=float).reshape(-1, 1)
    y = np.array(['D', 'R', 'D', 'R', 'R'])
    original = knn_experiment.make_knn
    calls = []
    class AuditedModel:
        def __init__(self, k): self.model = original(k)
        def fit(self, train_x, train_y):
            self.training = train_x.copy()
            assert len(train_x) == 4
            self.model.fit(train_x, train_y)
            return self
        def predict(self, validation):
            assert len(validation) == 1
            assert validation[0, 0] not in self.training[:, 0]
            calls.append(validation[0, 0])
            return self.model.predict(validation)
    monkeypatch.setattr(knn_experiment, 'make_knn', AuditedModel)
    k, table, audit = knn_experiment.select_knn(x, y)
    assert len(calls) == 20
    assert list(table.k) == [1, 2, 3, 4]
    assert k == table.loc[table.loocv_error.eq(table.loocv_error.min()), 'k'].min()
    assert audit.groupby('k').validation_index.nunique().eq(5).all()


def test_knn_vote_tie_and_repeatability():
    model = knn_experiment.make_knn(2).fit(np.array([[0.], [2.]]), ['R', 'D'])
    assert model.predict([[1.]])[0] == 'D'
    data = load_election(ROOT/'data/obesity_election_2004.csv')
    model = knn_experiment.make_knn(3).fit(data[['Obesity']].to_numpy(), data.Election_Result)
    np.testing.assert_array_equal(model.predict([[22.9], [24.0]]), model.predict([[22.9], [24.0]]))


def test_problem_year_directions(monkeypatch):
    data = {2000: object(), 2004: object()}
    def capture(*args): return args[:5]
    monkeypatch.setattr(problem_2_7, 'run_knn', capture)
    monkeypatch.setattr(problem_2_8, 'run_knn', capture)
    assert problem_2_7.run(data, None, None) == (data[2004], data[2000], 2004, 2000, '2.7')
    assert problem_2_8.run(data, None, None) == (data[2000], data[2004], 2000, 2004, '2.8')


def test_cv_selection_sees_only_its_training_fold(monkeypatch, tmp_path):
    data = generate_regression()
    original = problem_2_12.select_polynomial
    seen = []
    def audited(x, y, family):
        assert len(x) == len(y) == 8
        indices = np.array([np.flatnonzero(data.x.to_numpy() == v).item() for v in x])
        np.testing.assert_array_equal(y, data.y.to_numpy()[indices])
        seen.append((family, tuple(indices)))
        model, table = original(x, y, family)
        assert list(table.m) == list(range(7))
        return model, table
    monkeypatch.setattr(problem_2_12, 'select_polynomial', audited)
    folds, summary, oof = problem_2_12.run(data, tmp_path, tmp_path)
    assert len(seen) == 10
    for i in range(0, 10, 2): assert seen[i][1] == seen[i+1][1]
    for family in ['trigonometric', 'algebraic']:
        subset = oof[oof.family.eq(family)]
        assert sorted(subset.sample_id) == list(range(10))
        for fold in folds.itertuples():
            train_indices = set(map(int, fold.train_indices.split(',')))
            validation = subset[subset.fold.eq(fold.fold)]
            assert len(validation) == 2
            assert train_indices.isdisjoint(validation.sample_id)
            assert set(validation.sample_id) == set(range(10))-train_indices
        row = summary[summary.family.eq(family)].iloc[0]
        assert row.mean_validation_mse == pytest.approx(row.oof_mse)


def test_artifact_metrics_and_report_consistency():
    tables = ROOT/'results/tables'
    comparison = pd.read_csv(tables/'knn_comparison.csv')
    report = (ROOT/'report.md').read_text(encoding='utf-8')
    for row in comparison.itertuples():
        prefix = 'problem_'+str(row.problem).replace('.', '_')
        loo = pd.read_csv(tables/f'{prefix}_loocv.csv')
        predictions = pd.read_csv(tables/f'{prefix}_test_predictions.csv')
        audit = pd.read_csv(tables/f'{prefix}_loocv_predictions.csv')
        assert len(loo) == 50 and len(audit) == 2550
        assert audit.n_train.eq(50).all() and audit.n_validation.eq(1).all()
        assert row.optimal_k == loo.loc[loo.loocv_error.eq(loo.loocv_error.min()), 'k'].min()
        assert row.test_error == pytest.approx(predictions.Wrong.mean())
        assert row.test_accuracy == pytest.approx(1-row.test_error)
        assert f'{row.test_error:.8f}' in report
    for family in ['trigonometric', 'algebraic']:
        scores = pd.read_csv(tables/f'problem_2_11_{family}_selection.csv')
        expected = (1+(scores.DoF/10)*np.log(10)/(1-scores.DoF/10))*scores.training_mse
        np.testing.assert_allclose(scores.schwartz_criterion, expected)
    assert len(list((ROOT/'results/figures').glob('*.png'))) == 15
    import re
    for relative in re.findall(r'\]\(([^)]+)\)', report):
        assert (ROOT/relative).exists(), relative


def artifact_hashes(root):
    paths = [root/'report.md', root/'data/generated_example_2_6.csv', *sorted((root/'results').rglob('*.csv')), *sorted((root/'results').rglob('*.png'))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_clean_project_reproduces_all_artifacts(tmp_path):
    project = tmp_path/'HW1'
    project.mkdir()
    shutil.copytree(ROOT/'src', project/'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'data', project/'data')
    shutil.copyfile(ROOT/'main.py', project/'main.py')
    subprocess.run([sys.executable, str(project/'main.py')], cwd=tmp_path, check=True, capture_output=True, timeout=240)
    assert artifact_hashes(project) == artifact_hashes(ROOT)
    before = artifact_hashes(project)
    subprocess.run([sys.executable, 'main.py'], cwd=project, check=True, capture_output=True, timeout=240)
    assert artifact_hashes(project) == before
