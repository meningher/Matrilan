import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy.stats import zscore


def create_features_matrix(in_dir: Path, window_folder: str, threshold=0.2):
    list_of_vecs = []
    csv_files_list = list(in_dir.joinpath(window_folder).rglob('*.csv'))
    for p in tqdm(csv_files_list):
        correlation_mat = np.loadtxt(p, delimiter=',')
        n_rows = correlation_mat.shape[0]
        correlation_vec = correlation_mat[np.tril_indices(n_rows, -1)]
        list_of_vecs.append(correlation_vec)

    features = np.array(list_of_vecs)
    average_vec = np.mean(list_of_vecs, axis=0)
    th_mask = np.where(abs(average_vec) > threshold)[0]
    features = np.delete(features, th_mask, axis=1)

    return features


def calc_subject_examplars(subject_features_mat, peaks_threshold=0.3, plot=False):
    features_z_score = zscore(subject_features_mat, axis=0)
    exemplar_mean = np.abs(features_z_score.mean(axis=1))
    if plot:
        plt.plot(exemplar_mean)
        plt.show()

    peaks_idx = np.argwhere(exemplar_mean >= peaks_threshold).ravel()

    return subject_features_mat[peaks_idx, :]
