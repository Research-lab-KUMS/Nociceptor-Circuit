"""
Adel Parvizi-Fard  **
analog chip modeling of pain receptors (Nociceptor)
E_2 --> 4 levels of pressure for two object stimuli
"""
# ######################################import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'qt')
from scipy.signal import find_peaks

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
#  ###################################################################  load data

n_pc = 3  # parameters for classifier
n_knn = 5
c_v = 5
trial = 10
stim = [0,1,2,3,4,5,6,7]  #0,1,2,3,
N_neuron = 3

time = 0.2  # trapezoidal start from 0.03 to 0.18 totally 200 us
spike_t = []
y = np.array([trial * [j] for j in range(len(stim))]).flatten()  # desired output

data_2 = np.zeros((trial * len(stim), N_neuron))
neuron=['/outN (V)', '/outN2 (V)', '/outN3 (V)']
acu_test=[]
std_test=[]
for time in np.linspace(0,0.2,20):
    for i,j in enumerate(stim):
        for tr in range(trial):
            df = pd.read_csv('data_output/Ex_2/3p/{}.csv'.format(j * trial + tr))
            for h in range(N_neuron):
                t = np.array(df['{}'.format(df.keys()[0])])
                sp = np.array(df[neuron[h]])
                qqw, _ = find_peaks(sp, height=1.1, distance=10)
                a = t[qqw] * 1000
                aa = a[a < time]
                data_2[i * trial + tr, h] = len(aa)

    data = data_2
    # ##  ######################
    classifier = KNeighborsClassifier(n_neighbors=n_knn)
    cv2 = StratifiedKFold(n_splits=5, random_state=79, shuffle=True)
    cv_results = cross_validate(classifier, data, y, cv=cv2, return_train_score=True)
    acu_test.append(cv_results['test_score'].mean())
    std_test.append(cv_results['test_score'].std())


plt.plot(acu_test)
#%% Ex_2 4 level of pressure - spike count - classifier (KNN(k=5)-SVM) time window

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# stim=[4,5,6,7]  # type_I 0,1,2,3,  type_II 4,5,6,7
stim=[0,1,2,3,4,5,6,7]  # type_I 0,1,2,3,  type_II 4,5,6,7
N_neuron=[1,2,3]  # Neuron 1,2,3
neuron=['/outN (V)', '/outN2 (V)', '/outN3 (V)']
trial=10
n_knn=5
train_s=[]
test_s=[]
data_2 = np.zeros((trial * len(stim), len(N_neuron)))
y = np.array([trial * [j] for j in range(len(stim))]).flatten()  # desired output
tim_linspace = np.linspace(0.00,0.2,20)
for tt in tim_linspace:
    for i1,i in enumerate(stim):
        for tr in range(trial):

            df = pd.read_csv('data_output/Ex_2/3p/{}.csv'.format(i * trial + tr))
            for h1,h in enumerate(N_neuron):
                t = np.array(df['{}'.format(df.keys()[0])])
                sp = np.array(df[neuron[h1]])
                qqw, _ = find_peaks(sp, height=1.1, distance=10)
                a = t[qqw] * 1000
                aa = a[a < tt]
                data_2[i1 * trial + tr, h1] = len(aa)
    # classifier = LogisticRegression(random_state=int(tt*300), solver='lbfgs', multi_class='multinomial')
    # classifier = DecisionTreeClassifier()
    # classifier = RandomForestClassifier(max_depth=2, random_state=0)
    # classifier = KNeighborsClassifier(n_neighbors=n_knn)
    classifier= SVC()  # gamma='auto'
    cv2 = StratifiedKFold(n_splits=5, random_state=int(tt*300), shuffle=True)
    cv_results = cross_validate(classifier, data_2, y, cv=cv2, return_train_score=True)
    test_s.append(cv_results['test_score'].mean())
    train_s.append(cv_results['train_score'].mean())
plt.plot(test_s)
plt.ylim([0,1.1])
print(test_s)
y_pred = cross_val_predict(classifier, data_2, y, cv=cv2)
conf_mat = confusion_matrix(y, y_pred)
print('confusion matrix is \n', conf_mat)

target_names = [str(i) for i in stim]
print(classification_report(y, y_pred, target_names=target_names))

#%%
pos = plt.imshow(conf_mat,cmap='hot_r')
plt.colorbar(pos)
#%% Ex_2 4 level of pressure - spike time (VRd) - classifier (KNN(k=5)-SVM) time window
from elephant.spike_train_dissimilarity import van_rossum_dist, victor_purpura_dist
import quantities as pq

tau=0.01* pq.ms

n_pc = 3  # parameters for classifier
n_knn = 5
c_v = 5
trial = 10
# stim = [0,1,4,5,8,9]
stim = np.arange(8)
# stim = np.arange(2)
N_neuron = 3
neuron=['/outN (V)', '/outN2 (V)', '/outN3 (V)']
train_s=[]
test_s=[]
time = 0.4  # trapezoidal start from 0.03 to 0.18 totally 200 us
spike_t = []
y = np.array([trial * [j] for j in stim]).flatten()  # desired output

tim_linspace = np.linspace(0.00,0.2,20)
for tt in [0.2]:
    spike_t=[]
    for i in stim:

        for tr in range(trial):
            df = pd.read_csv('data_output/Ex_2/3p/{}.csv'.format(i * trial+tr))
            for h in range(3):
                t = np.array(df['{}'.format(df.keys()[0])])
                sp = np.array(df[neuron[h]])
                qqw, _ = find_peaks(sp, height=1.1, distance=10)
                a = t[qqw] * 1000
                aa = a[a < tt]
                spike_t.append(aa)

    spike_t=np.array(spike_t)*pq.ms
    data_vrd=np.array([])
    # fig, ax=plt.subplots(1,3,figsize=(20,5))
    for j in range(3):
        dis=np.real(van_rossum_dist([spike_t[i*3+j] for i in range(0,80)],tau))   # if put [0,1] end of this line return one value
        # plt.figure(j)

        if j>0 :
            data_vrd=np.concatenate((data_vrd,dis),axis=1)
        else:
            data_vrd=dis
        # pos=ax[j].imshow(dis)
        # fig.colorbar(pos, ax=ax[j])
    data = decomposition.PCA(n_components=n_pc).fit_transform(data_vrd)
    # classifier = KNeighborsClassifier(n_neighbors=n_knn)
    classifier= SVC()  # gamma='auto'
    cv2 = StratifiedKFold(n_splits=5, random_state=int(tt*300), shuffle=True)
    cv_results = cross_validate(classifier, data, y, cv=cv2, return_train_score=True)
    test_s.append(cv_results['test_score'].mean())
    train_s.append(cv_results['train_score'].mean())
plt.plot(test_s)
plt.ylim([0,1.1])
print(test_s)
#%%
y_pred = cross_val_predict(classifier, data, y, cv=cv2)
conf_mat = confusion_matrix(y, y_pred)
pos = plt.imshow(conf_mat,cmap='hot_r')
plt.colorbar(pos)
# pos = plt.imshow(data_vrd,cmap='jet')
# plt.colorbar(pos)
#%%


