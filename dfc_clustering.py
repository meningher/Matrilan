import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from utils import create_features_matrix
from utils import calc_subject_exemplars
from mlinsights.mlmodel import KMeansL1L2
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    window_folder = '45_1'
    TH = 0.2
    norm = 'L1'
    scaled = "scaled"
    number_of_subjects = 50
    in_dir = Path('/media/neuro/LivnyLab/Research/TBI_magneton/Analyses/MatriLan/matrices/Mat_180521/FC/Dynamic_Correlations/r_val/no_TH')
    [features_mat, num_of_windows] = create_features_matrix(in_dir, window_folder, TH, number_of_subjects)
    all_subs_exemplars_mat = []
    for sub_num in range(number_of_subjects):
        one_subject_features_mat = features_mat[int(sub_num*num_of_windows):int((sub_num+1)*num_of_windows)]
        one_subject_exemplar_mat = calc_subject_exemplars(one_subject_features_mat, peaks_threshold=0.2, plot=False)
        all_subs_exemplars_mat.append(one_subject_exemplar_mat)
        if len(one_subject_exemplar_mat) > 0:
            print(f"\nsubject num {sub_num} has {len(one_subject_exemplar_mat)} exemplars")
        else:
            print(f"\nsubject num {sub_num} is empty!")

    # here I need to take all the matrices coming back from calc_subject_exemplars and append them to one matrix-->all_subs_features_map

    all_subs_exemplars_mat = np.concatenate(all_subs_exemplars_mat, axis=0)
    print(f'\n{all_subs_exemplars_mat.shape}\n')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_subs_exemplars_mat)    # fisher transformation
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
