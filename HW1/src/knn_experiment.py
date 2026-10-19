"""Shared implementation for Problems 2.7 and 2.8."""
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits
from .plotting import plt, save


def make_knn(k):
    # Pin backend, input order and thread count to make distance ties repeatable.
    return KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean',
                                algorithm='brute', n_jobs=1)


def select_knn(x, y):
    """Actual n leave-one-out fits for each k=1,...,n-1, using 0-1 loss."""
    x, y = np.asarray(x), np.asarray(y)
    rows, audit = [], []
    with threadpool_limits(limits=1):
        for k in range(1, len(y)):
            predictions = np.empty(len(y), dtype=y.dtype)
            for train, validation in LeaveOneOut().split(x):
                model = make_knn(k).fit(x[train], y[train])
                predictions[validation] = model.predict(x[validation])
            wrong = predictions != y
            rows.append({'k': k, 'wrong_count': int(wrong.sum()), 'loocv_error': float(wrong.mean())})
            audit.extend({'k': k, 'validation_index': i, 'n_train': len(y)-1,
                          'n_validation': 1, 'actual': y[i], 'predicted': predictions[i],
                          'wrong': bool(wrong[i])} for i in range(len(y)))
    table = pd.DataFrame(rows)
    best_position = table['loocv_error'].to_numpy().argmin()
    return int(table['k'].to_numpy()[best_position]), table, pd.DataFrame(audit)


def run_knn(train, test, train_year, test_year, problem, tables, figures):
    x, y = train[['Obesity']].to_numpy(), train.Election_Result.to_numpy()
    k, cv, audit = select_knn(x, y)
    with threadpool_limits(limits=1):
        model = make_knn(k).fit(x, y)
        train_prediction = model.predict(x)
        prediction = model.predict(test[['Obesity']].to_numpy())
    prefix = f'problem_{problem.replace(".", "_")}'
    result = test.copy()
    result['Predicted'] = prediction
    result['Wrong'] = prediction != test.Election_Result.to_numpy()
    train_result = train.copy()
    train_result['Predicted'] = train_prediction
    train_result['Wrong'] = train_prediction != y
    cv.to_csv(tables/f'{prefix}_loocv.csv', index=False)
    audit.to_csv(tables/f'{prefix}_loocv_predictions.csv', index=False)
    result.to_csv(tables/f'{prefix}_test_predictions.csv', index=False)
    result[result.Wrong].to_csv(tables/f'{prefix}_misclassified_states.csv', index=False)
    train_result.to_csv(tables/f'{prefix}_training_predictions.csv', index=False)
    cm = confusion_matrix(test.Election_Result, prediction, labels=['D', 'R'])
    pd.DataFrame(cm, index=['actual_D', 'actual_R'], columns=['predicted_D', 'predicted_R']).to_csv(tables/f'{prefix}_confusion_matrix.csv')
    summary = {'problem': problem, 'training_year': train_year, 'test_year': test_year,
               'optimal_k': k, 'loocv_error': float(cv.loocv_error.min()),
               'training_error': float(np.mean(train_prediction != y)),
               'test_error': float(result.Wrong.mean()), 'test_accuracy': float(1-result.Wrong.mean()),
               'test_wrong_count': int(result.Wrong.sum()),
               'tied_best_ks': ', '.join(map(str, cv.loc[cv.loocv_error.eq(cv.loocv_error.min()), 'k']))}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cv.k, cv.loocv_error, marker='.', label='LOOCV error')
    ax.axvline(k, color='red', linestyle='--', label=f'Selected k={k}')
    ax.set(title=f'Problem {problem}: {train_year} model selection', xlabel='Neighbors k', ylabel='0-1 error rate (proportion)')
    ax.legend(); save(fig, figures/f'{prefix}_loocv.png')
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in [('D', 'tab:blue'), ('R', 'tab:red')]:
        subset = train[train.Election_Result.eq(label)]
        ax.scatter(subset.Obesity, np.full(len(subset), label), c=color, alpha=0.7, label=label)
    ax.set(title=f'Problem {problem}: {train_year} training observations', xlabel='Obesity index (%)', ylabel='Election label')
    ax.legend(); save(fig, figures/f'{prefix}_training.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    actual = test.Election_Result.eq('R').astype(int)
    predicted = (prediction == 'R').astype(int)
    ax.scatter(test.Obesity, actual-0.04, marker='o', label='Actual')
    ax.scatter(test.Obesity, predicted+0.04, marker='x', label='Predicted')
    ax.scatter(test.loc[result.Wrong, 'Obesity'], predicted[result.Wrong]+0.04,
               facecolors='none', edgecolors='red', s=130, label='Incorrect prediction')
    ax.set(yticks=[0, 1], yticklabels=['D', 'R'], ylim=(-0.3, 1.3),
           title=f'Problem {problem}: {test_year} test predictions (k={k})', xlabel='Obesity index (%)', ylabel='Election label (markers offset)')
    ax.legend(loc='upper left'); save(fig, figures/f'{prefix}_test.png')
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['D', 'R']).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.grid(False); ax.set_title(f'Problem {problem}: {test_year} test counts')
    save(fig, figures/f'{prefix}_confusion.png')
    return summary, result, cm
