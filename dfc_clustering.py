import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from mlinsights.mlmodel import KMeansL1L2
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    window_folder = '60_10'
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
    features_mat = np.array(list_of_vecs)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_mat)    # fisher transformation
    sse = []
    K_range = range(1, 11)
    for k in tqdm(K_range):
        if norm == 'L1':
            kmeans = KMeansL1L2(n_clusters=k, norm='L1')
        else:
            kmeans = KMeans(n_clusters=k, random_state=42)
        # clusters = kmeans.fit_predict(scaled_features)
        if scaled == "scaled":
            kmeans = kmeans.fit(scaled_features)    #### select the features here ###
        else:
            kmeans = kmeans.fit(features_mat)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K_range, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    title = ('The Elbow Method showing the optimal k for', window_folder, norm, scaled)
    plt.title(title)
    #plt.show()
    full_path = ('/media/neuro/LivnyLab/Research/TBI_magneton/Analyses/MatriLan/matrices/Mat_180521/FC/Dynamic_Correlations/r_val/no_TH/' + window_folder + '_' + norm + '_' + scaled)
    plt.savefig(full_path)
