"""Generate the English typed solution from actual experiment outputs."""
import platform
import numpy as np
import pandas as pd
import sklearn
import matplotlib


def markdown_table(frame):
    def fmt(value):
        return f'{value:.8g}' if isinstance(value, (float, np.floating)) else str(value)
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---']*len(frame.columns)) + ' |',
                      *['| ' + ' | '.join(fmt(v) for v in row) + ' |'
                        for row in frame.itertuples(index=False, name=None)]])


def write_report(path, statistics, knn_results, polynomial, poly_summary, folds, cv_summary):
    parts = [r'''# Homework 1: Risk Minimization and Model Selection

## Experimental setup and source references

This typed solution uses the supplied election CSV files and a newly generated regression sample. It is an implementation and analysis aid: the student should verify assumptions, understand the calculations, and revise interpretations in their own words before submission.

All page references refer to PDF page numbers, matching the slide numbers. The source PDFs remain unchanged in the parent directory.

| Source | Relevant pages and methods |
| --- | --- |
| Homework1.pdf | pp. 1-2: all four problems, exact cosine-only basis and regression generator |
| intro_ML_algorithms_scikit_learn.pdf | p. 10: polynomial basis expansion; pp. 20-22: Euclidean nearest neighbors, uniform voting |
| lecture_set01.pdf | pp. 39-40: experimental procedure, preprocessing |
| lecture_set02.pdf | pp. 17-19: squared and 0-1 loss; p. 25: empirical risk minimization; pp. 35-38: complexity; p. 13: causality; pp. 15, 58: stationarity/i.i.d. |
| lecture_set03.pdf | pp. 12-13: basis expansions; pp. 25-32: bias-variance; p. 34: Schwartz; pp. 36-43: CV, LOO and assessment; pp. 47-48: fold-specific selection |

Lecture Set 2, pp. 36-37, illustrates $y=x^2+\epsilon$, whereas Homework1.pdf p. 2 explicitly specifies $y=x^2+0.1x+\epsilon$. The assignment takes precedence. The exact textbook ten observations are not provided; the new sample is not claimed to reproduce them.

## Election data validation

Each headerless CSV is loaded with `header=None` and four explicit column names. Both contain 51 complete observations, Alabama and DC, unique state names/codes, valid percentages and only R/D labels. State-code sets match across years. Obesity strings become numbers in percent units; `22.60%` becomes `22.60`. Only obesity is an input feature. Original row order and labels are retained.

The supplied 2004 file labels Hawaii as R. It is preserved rather than silently corrected. Its correspondence to historical records has not been independently verified; confirm this entry with the instructor. Results predict the supplied labels.
''', markdown_table(statistics)]
    for summary, predictions, cm in knn_results:
        problem = summary['problem']
        prefix = 'problem_' + problem.replace('.', '_')
        parts += [f'''\n## Problem {problem}: KNN classification

### Problem description and dataset / experimental setup

Predict R/D from obesity alone. Use 51 observations from {summary['training_year']} for training and 51 from {summary['test_year']} for external testing. Test features and labels never select k. This is cross-year transfer on the same states, rather than an independent sample of new states.

### Mathematical formulation and model selection procedure

Euclidean distance for the single feature is |x-x_i|. The nearest k observations vote uniformly. The 0-1 loss is 1 for an incorrect label and 0 otherwise; error rate averages this loss. For every k=1,...,50, 51 LOO fits each use 50 observations and predict exactly one held-out observation. Choose the smallest k attaining the minimum LOO error, then fit again on all training observations. Both even and odd k are included.

scikit-learn uses the explicit brute-force Euclidean backend, original CSV order, and one numerical thread. A class-vote tie selects D (first sorted class). Equal distances at the kth neighbor can cause membership to depend on row order and library implementation; fixed order and package versions are part of reproduction. No extra feature or artificial jitter is used.

### Experimental results

Optimal k = **{summary['optimal_k']}**, with tied best values {summary['tied_best_ks']}. Resampling / LOOCV error = **{summary['loocv_error']:.8f} ({summary['loocv_error']:.2%})**. Training error = **{summary['training_error']:.8f} ({summary['training_error']:.2%})**. Test error = **{summary['test_error']:.8f} ({summary['test_error']:.2%})**, or **{summary['test_wrong_count']}/51** incorrect labels. Test accuracy = **{summary['test_accuracy']:.8f} ({summary['test_accuracy']:.2%})**.

Confusion matrix entries are counts; rows are actual D/R and columns are predicted D/R.
''', markdown_table(pd.DataFrame({'Actual': ['D', 'R'], 'Predicted D': cm[:, 0], 'Predicted R': cm[:, 1]})),
                  '\nMisclassified test states:\n', markdown_table(predictions.loc[predictions.Wrong, ['State', 'Obesity', 'Election_Result', 'Predicted']]),
                  f'''\n### Figures and tables

![LOOCV scores](results/figures/{prefix}_loocv.png)
![Training observations](results/figures/{prefix}_training.png)
![Test predictions](results/figures/{prefix}_test.png)
![Confusion matrix](results/figures/{prefix}_confusion.png)

[LOO scores](results/tables/{prefix}_loocv.csv), [every LOO prediction](results/tables/{prefix}_loocv_predictions.csv), [training predictions](results/tables/{prefix}_training_predictions.csv), [test predictions](results/tables/{prefix}_test_predictions.csv), [misclassified states](results/tables/{prefix}_misclassified_states.csv), [confusion counts](results/tables/{prefix}_confusion_matrix.csv).

### Discussion / limitations

Test minus LOO error is {summary['test_error']-summary['loocv_error']:.8f}, or {(summary['test_error']-summary['loocv_error'])*100:.4f} percentage points. The selected k is supported by its LOO minimum in the specified search, not by a guarantee of optimal population risk. Small k gives more flexible local decisions but increases sensitivity to noise and ties; large k smooths decisions and can underfit. The minimized LOO score can be optimistic after selection over 50 candidates.

The test mistakes and overlapping classes show that obesity does not uniquely determine the supplied outcomes. Year-to-year shifts in obesity and labels can change the joint distribution. Geographic dependence and the reuse of the same states challenge i.i.d. and stationarity assumptions. Statistical dependency is not evidence that obesity causes voting outcomes (Lecture Set 2, p. 13).
''']
    comparison = pd.DataFrame([result[0] for result in knn_results])
    parts += ['\n## Cross-year comparison\n', markdown_table(comparison),
              f'''\nMean obesity is {statistics.iloc[0].obesity_mean:.6f}% in 2000 and {statistics.iloc[1].obesity_mean:.6f}% in 2004. Swapping years changes neighborhoods, label proportions, the LOO score landscape and the relationship between test values and the training range. Therefore Problem 2.8 reselects k independently. These are possible explanations of asymmetric transfer, not causal claims. [Comparison CSV](results/tables/knn_comparison.csv).

## Problem 2.11: polynomial regression with Schwartz selection

### Problem description and dataset / experimental setup

Use the same ten observations for both families. `np.random.default_rng(42)` first draws ten x values from Uniform(0,1), then ten independent noise values from Normal(0,0.5 standard deviation). Set y=x^2+0.1x+noise. Noise variance is 0.25. The [generated sample](data/generated_example_2_6.csv) discloses x, y, sample ID, noise and the reference function. Only x and y enter estimation and selection; reference values only support generation, visualization and interpretation.

### Mathematical formulation

Trigonometric: $f_m(x)=w_0+\\sum_{{i=1}}^m w_i\\cos(2\\pi i x)$, with no sine terms. Algebraic: $f_m(x)=\\sum_{{i=0}}^m w_i x^i$. Both have DoF=m+1. SVD-based `numpy.linalg.lstsq` minimizes squared training residuals without ridge/lasso or inversion of normal equations.

### Model selection procedure

For a training size n, evaluate m=0,...,n-2 to ensure DoF<n. Thus the full n=10 sample allows m=0,...,8, including the constant-only candidate. Compute $R_{{emp}}=n^{{-1}}\\sum_i(y_i-f_m(x_i))^2$, $p=(m+1)/n$, and the Lecture Set 3 p. 34 formula:

$$R_{{SC}}=\\left(1+\\frac{{p\\ln n}}{{1-p}}\\right)R_{{emp}}.$$

Minimize R_SC; exact score ties choose smaller m. The multiplier penalizes complexity increasingly as p approaches one. This is the multiplicative course formula rather than the common logarithmic BIC expression.

### Experimental results
''', markdown_table(poly_summary)]
    for family, (model, selection) in polynomial.items():
        parts += [f'\n#### {family.title()} selection\n', markdown_table(selection),
                  '\nSelected coefficients in basis order:\n',
                  markdown_table(pd.DataFrame({'coefficient': [f'w{i}' for i in range(model.m+1)], 'value': model.coefficients})),
                  f'\n[Selection CSV](results/tables/problem_2_11_{family}_selection.csv), [coefficients](results/tables/problem_2_11_{family}_coefficients.csv).\n']
    analytic_winner = poly_summary.loc[poly_summary.schwartz_criterion.idxmin(), 'family']
    largest = max(table.condition_number.max() for _, table in polynomial.values())
    parts += [f'''\n### Figures and tables

![Trigonometric full-data fit](results/figures/problem_2_11_trigonometric_fit.png)
![Algebraic full-data fit](results/figures/problem_2_11_algebraic_fit.png)
![Schwartz criterion](results/figures/problem_2_11_criterion.png)
![Training MSE](results/figures/problem_2_11_training_mse.png)

### Discussion / limitations: answer to Problem 2.11(b)

The selected **{analytic_winner}** model has the smaller full-data analytic score. That ranking alone cannot establish the best predictive family. Within a family, Schwartz selects complexity by trading training fit against parameter count. Across families, equal DoF does not imply equal approximation bias or uncertainty. The analytic score estimates rather than observes out-of-sample error. Independent assessment, as in Problem 2.12, is needed for an empirical predictive comparison and still does not prove universal superiority.

Cosine-only functions obey f(x)=f(1-x), whereas the generating function is asymmetric. This restriction helps interpret approximation bias after fitting; the reference function did not choose m. Higher degrees may reduce training loss while creating unstable predictions.

Maximum full-data candidate condition number: **{largest:.8g}**. Tables record every numerical rank and condition number; >1e8 is a diagnostic flag, not an exclusion or regularizer. SVD avoids the additional instability of normal-equation inversion, but poorly conditioned bases remain sensitive. Compare rank with nominal DoF: a rank-deficient design would weaken the nominal parameter-count interpretation.

## Problem 2.12: five-fold comparison of prediction accuracy

### Problem description and dataset / experimental setup

Use the exact same ten observations as Problem 2.11. `KFold(n_splits=5, shuffle=True, random_state=42)` defines one shared partition for both families. Every fold has eight training and two held-out validation observations. Every sample is held out once per family.

### Mathematical formulation and model selection procedure

For each family and fold, minimize Schwartz using only the eight training observations over m=0,...,6. Then predict the two held-out noisy responses and compute their average squared residual (validation MSE). Full-data selected m is never reused for assessment. Schwartz is the inner selection score; validation MSE is the outer performance score (Lecture Set 3, pp. 37, 42-43, 47-48).

Mean CV MSE averages the five fold MSEs. The sample standard deviation uses ddof=1, describing fold spread rather than a confidence interval. Overall out-of-fold (OOF) MSE averages the ten held-out squared errors. Equal fold sizes imply identical mean-fold and OOF MSE, apart from rounding. Evaluation compares predictions to observed noisy y, not noiseless reference values.

### Experimental results
''', markdown_table(folds[['fold', 'trigonometric_optimal_m', 'trigonometric_validation_mse',
                          'algebraic_optimal_m', 'algebraic_validation_mse']]), '\n', markdown_table(cv_summary)]
    winner = cv_summary.loc[cv_summary.mean_validation_mse.idxmin(), 'family']
    scores = cv_summary.set_index('family').mean_validation_mse
    delta = scores['trigonometric']-scores['algebraic']
    largest_fold = folds.loc[folds.algebraic_validation_mse.idxmax()]
    oof = pd.read_csv(path.parent/'results/tables/problem_2_12_oof_predictions.csv')
    subset = oof[oof.family.eq('algebraic') & oof.fold.eq(largest_fold['fold'])]
    point = subset.loc[subset.squared_error.idxmax()]
    parts += [f'''\nThe **{winner}** family has lower mean CV MSE in this experiment. Trigonometric minus algebraic mean CV MSE = **{delta:.8g}** squared y units.

The largest algebraic error is in fold **{int(largest_fold['fold'])}**, where the selected degree is **{int(largest_fold.algebraic_optimal_m)}**. Its training-only Schwartz score is **{largest_fold.algebraic_schwartz_criterion:.8g}**, but held-out MSE is **{largest_fold.algebraic_validation_mse:.8g}**; condition number is **{largest_fold.algebraic_condition_number:.8g}**. At held-out x={point.x:.8g}, observed y={point.actual_y:.8g}, predicted y={point.predicted_y:.8g}. This one unstable high-degree fit dominates the aggregate. The error is retained without clipping or adding unrequested regularization.

### Figures and tables

![Fold MSE](results/figures/problem_2_12_folds.png)
![Mean CV MSE](results/figures/problem_2_12_mean_mse.png)
![OOF predictions versus held-out responses](results/figures/problem_2_12_oof.png)

Fold and mean MSE plots use labeled logarithmic vertical axes to make both families visible. OOF plots retain the full prediction range.

[Fold results](results/tables/problem_2_12_folds.csv), [all training-fold selection scores](results/tables/problem_2_12_fold_selection.csv), [assignments](results/tables/problem_2_12_fold_assignments.csv), [OOF predictions](results/tables/problem_2_12_oof_predictions.csv), [summary](results/tables/problem_2_12_summary.csv).

### Discussion / limitations

Schwartz approximates risk from training fit and complexity; it does not observe held-out responses and cannot guarantee the smallest validation error. Complex fits amplify noise, especially with clustered training inputs or poorly conditioned monomial matrices; simple fits may underfit. This is the bias-variance trade-off (Lecture Set 3, pp. 25-32).

With just ten noisy observations, two-point fold errors vary strongly and a single unstable prediction may dominate. Fold training sets overlap, so fold errors are not independent. The comparison describes this sample, fixed partition and eight-observation fitting procedure, not all datasets or the exact population risk of the final ten-point models. Repeated assessment could explore sensitivity but is outside the requested experiment.

## Reproduction and verification

Run `python main.py` from HW1, or `python HW1/main.py` from the parent directory, after installing requirements. All samples, tables, figures and this report are regenerated. The automated tests check validation, noise construction, basis definitions, criterion bounds, LOO exclusion, tie handling, training-only fold selection, OOF coverage and artifact/report consistency. Deterministic artifact hashes are compared across repeated runs and a clean temporary project.

Runtime: Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}, Matplotlib {matplotlib.__version__}. CSV order and package versions are fixed for reproducibility.
''']
    path.write_text('\n'.join(parts), encoding='utf-8')
