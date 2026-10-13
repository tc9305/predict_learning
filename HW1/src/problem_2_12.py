import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .polynomial_models import FAMILIES, select_polynomial
from .plotting import plt, save


def run(data, tables, figures):
    x, y = data.x.to_numpy(), data.y.to_numpy()
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(x))
    rows, predictions, assignments, candidates = [], [], [], []
    for fold, (train, validation) in enumerate(folds, 1):
        row = {'fold': fold, 'n_train': len(train), 'n_validation': len(validation),
               'train_indices': ','.join(map(str, train)), 'validation_indices': ','.join(map(str, validation))}
        assignments.extend({'sample_id': int(i), 'validation_fold': fold} for i in validation)
        for family in FAMILIES:
            model, selection = select_polynomial(x[train], y[train], family)
            selection = selection.assign(fold=fold, family=family)
            candidates.append(selection)
            predicted = model.predict(x[validation])
            mse = float(np.mean((y[validation]-predicted)**2))
            row[f'{family}_optimal_m'] = model.m
            row[f'{family}_schwartz_criterion'] = float(selection.schwartz_criterion.min())
            row[f'{family}_validation_mse'] = mse
            row[f'{family}_condition_number'] = model.condition_number
            predictions.extend({'sample_id': int(i), 'fold': fold, 'family': family,
                                'x': x[i], 'actual_y': y[i], 'predicted_y': p,
                                'squared_error': (y[i]-p)**2, 'optimal_m': model.m}
                               for i, p in zip(validation, predicted))
        rows.append(row)
    fold_table, oof = pd.DataFrame(rows), pd.DataFrame(predictions).sort_values(['family', 'sample_id'])
    pd.DataFrame(assignments).sort_values('sample_id').to_csv(tables/'problem_2_12_fold_assignments.csv', index=False)
    pd.concat(candidates, ignore_index=True).to_csv(tables/'problem_2_12_fold_selection.csv', index=False)
    fold_table.to_csv(tables/'problem_2_12_folds.csv', index=False)
    oof.to_csv(tables/'problem_2_12_oof_predictions.csv', index=False)
    summary = pd.DataFrame([{'family': family,
                           'mean_validation_mse': fold_table[f'{family}_validation_mse'].mean(),
                           'std_fold_mse_ddof1': fold_table[f'{family}_validation_mse'].std(ddof=1),
                           'oof_mse': oof.loc[oof.family.eq(family), 'squared_error'].mean()}
                           for family in FAMILIES])
    summary.to_csv(tables/'problem_2_12_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for family in FAMILIES:
        ax.plot(fold_table.fold, fold_table[f'{family}_validation_mse'], marker='o', label=family.title())
    ax.set_yscale('log')
    ax.set(title='Problem 2.12: held-out fold errors', xlabel='Validation fold', ylabel='Validation MSE (squared y units, log scale)', xticks=range(1, 6))
    ax.legend(); save(fig, figures/'problem_2_12_folds.png')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary.family, summary.mean_validation_mse)
    ax.set_yscale('log')
    for i, value in enumerate(summary.mean_validation_mse):
        ax.annotate(f'{value:.6g}', (i, value), xytext=(0, 5), textcoords='offset points', ha='center')
    ax.set_ylim(summary.mean_validation_mse.min()/2, summary.mean_validation_mse.max()*3)
    ax.set(title='Problem 2.12: mean 5-fold validation error', xlabel='Model family', ylabel='Mean CV MSE (squared y units, log scale)')
    save(fig, figures/'problem_2_12_mean_mse.png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, family in zip(axes, FAMILIES):
        subset = oof[oof.family.eq(family)]
        ax.scatter(subset.actual_y, subset.predicted_y, label='Out-of-fold prediction')
        limits = [subset.actual_y.min(), subset.actual_y.max()]
        ax.plot(limits, limits, '--', color='gray', label='Perfect prediction')
        ax.set(title=family.title(), xlabel='Observed noisy y', ylabel='Out-of-fold predicted y',
               xlim=(limits[0]-0.05, limits[1]+0.05))
        ax.legend()
    fig.suptitle('Problem 2.12: predictions against held-out observations')
    save(fig, figures/'problem_2_12_oof.png')
    return fold_table, summary, oof
