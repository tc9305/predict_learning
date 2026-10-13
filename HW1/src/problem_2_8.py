from .knn_experiment import run_knn


def run(data, tables, figures):
    return run_knn(data[2000], data[2004], 2000, 2004, '2.8', tables, figures)
