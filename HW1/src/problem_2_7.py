from .knn_experiment import run_knn


def run(data, tables, figures):
    return run_knn(data[2004], data[2000], 2004, 2000, '2.7', tables, figures)
