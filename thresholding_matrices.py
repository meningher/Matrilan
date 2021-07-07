from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from mlinsights.mlmodel import KMeansL1L2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

if __name__ == '__main__':
    window_folder = '60_10_2'
    norm = 'L1'
    scaled = "scaled"
    in_dir = Path('/media/neuro/LivnyLab/Research/TBI_magneton/Analyses/MatriLan/matrices/Mat_180521/FC/Dynamic_Correlations/r_val/no_TH')
    list_of_vecs = []
    csv_files_list = list(in_dir.joinpath(window_folder).rglob('*.csv'))
    for p in tqdm(csv_files_list):
        correlation_mat = np.loadtxt(p, delimiter=',')
        n_rows = correlation_mat.shape[0]
        correlation_vec = correlation_mat[np.tril_indices(n_rows, -1)]
        list_of_vecs.append(correlation_vec)
    average_vec = np.mean(list_of_vecs, axis=0)

TH = 0.2
th_mask = np.where(abs(average_vec) > TH, 1, 0) # creating a binary mask to multiply all vectors
list_of_vecs_th = np.multiply(list_of_vecs, th_mask) # multiplying all vectors
idx = np.argwhere(np.all(list_of_vecs_th[..., :] == 0, axis=0))
list_of_vecs_squ = np.delete(list_of_vecs_th, idx, axis=1)
print(list_of_vecs_squ)


x = 4
