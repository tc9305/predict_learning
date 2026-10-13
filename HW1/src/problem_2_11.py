import numpy as np
import pandas as pd
from .polynomial_models import FAMILIES, select_polynomial
from .plotting import plt, save


def run(data, tables, figures):
    x, y = data.x.to_numpy(), data.y.to_numpy()
    grid = np.linspace(0, 1, 500)
    outputs, rows = {}, []
    for family in FAMILIES:
        model, selection = select_polynomial(x, y, family)
        selection.to_csv(tables/f'problem_2_11_{family}_selection.csv', index=False)
        best = selection.loc[selection.m.eq(model.m)].iloc[0]
        rows.append({'family': family, 'optimal_m': model.m, 'optimal_DoF': model.m+1,
                     'training_mse': best.training_mse, 'schwartz_criterion': best.schwartz_criterion,
                     'rank': model.rank, 'condition_number': model.condition_number})
        pd.DataFrame({'coefficient': [f'w{i}' for i in range(model.m+1)],
                      'value': model.coefficients}).to_csv(tables/f'problem_2_11_{family}_coefficients.csv', index=False)
        outputs[family] = (model, selection)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(x, y, label='Noisy observations', color='black')
        ax.plot(grid, grid**2+0.1*grid, '--', label='True function (reference only)')
        ax.plot(grid, model.predict(grid), label=f'{family.title()} OLS, m={model.m}')
        ax.set(title=f'Problem 2.11: {family} full-data fit', xlabel='x', ylabel='y / predicted y')
        ax.legend(); save(fig, figures/f'problem_2_11_{family}_fit.png')
    for metric, label, suffix in [('schwartz_criterion', 'Schwartz criterion (squared y units)', 'criterion'),
                                   ('training_mse', 'Training MSE (squared y units)', 'training_mse')]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for family, (_, selection) in outputs.items():
            ax.plot(selection.m, selection[metric], marker='o', label=family.title())
        ax.set(title=f'Problem 2.11: {label}', xlabel='Degree / maximum cosine index m', ylabel=label)
        ax.legend(); save(fig, figures/f'problem_2_11_{suffix}.png')
    summary = pd.DataFrame(rows)
    summary.to_csv(tables/'problem_2_11_summary.csv', index=False)
    return outputs, summary
