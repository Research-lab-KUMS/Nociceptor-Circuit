"""
Adel Parvizi-Fard  **
analog chip modeling of pain receptors (Nociceptor)
E_1 --> equal indentation
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

#%% test
neuron=['/outN (V)', '/outN2 (V)', '/outN3 (V)']
time = 0.4
spike_t=[]
suma=[]
fig_name=[1,2,3,4,5,6,7,8,9]

for j in range(8):
    df = pd.read_csv('data_output/EX_2/{}.csv'.format(j*10+5))
    plt.figure('plot_EX_1_ob_{}'.format(fig_name[j]))
    for h in range(3):
        t = np.array(df['{}'.format(df.keys()[0])])
        sp = np.array(df[neuron[h]])
        qqw, _ = find_peaks(sp, height=1.1, distance=10)
        a = t[qqw] * 1000
        aa = a
        # aa = a[a < time]
        spike_t.append(a)
        suma.append(len(aa))
        # plt.plot(aa, [h+1+j for r in range(len(aa))], '.k')
        plt.plot(aa, [h for r in range(len(aa))], '|b',markersize=40)
        plt.xlim([0,0.2])
        plt.ylim([-0.5,2.5])
    plt.savefig('plot_EX_1_ob_{}'.format(fig_name[j]),dpi=500,quality=100)
    plt.close()
# plt.figure('E3')
# plt.stem(suma)
# %%  Ex_1 equal indentation - spike count - classifier (KNN(k=5)) time window

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
stim = [4,5,6,7]  #0,1,2,3,
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
            df = pd.read_csv('data_output/Ex_1/7-NOC/{}.csv'.format(j * trial + tr))
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

# print(cv_results['test_score'].mean())
# print(cv_results['test_score'].std())
# print('train score is ', cv_results['train_score'], 'mean and std are ',
#       "{0:.2f}".format(cv_results['train_score'].mean()), "{0:.2f}".format(cv_results['train_score'].std()))
# print('test score is ', cv_results['test_score'], 'mean and std are ',
#       "{0:.2f}".format(cv_results['test_score'].mean()), "{0:.2f}".format(cv_results['test_score'].std()))
# y_pred = cross_val_predict(classifier, data, y, cv=cv2)
# conf_mat = confusion_matrix(y, y_pred)
# plt.scatter(data[:, 1], data[:, 2])
# print('confusion matrix is \n', conf_mat)
# target_names = [str(i) for i in range(stim)]
# print(classification_report(y, y_pred, target_names=target_names))
plt.plot(acu_test)
#%% Ex_1 equal indentation - spike count - classifier (KNN(k=5)-SVM) time window

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# stim=[4,5,6,7]  # type_I 0,1,2,3,  type_II 4,5,6,7
stim=[4,5,6,7]  # type_I 0,1,2,3,  type_II 4,5,6,7
N_neuron=[1,2,3]  # Neuron 1,2,3
neuron=['/outN (V)', '/outN2 (V)', '/outN3 (V)']
trial=10
n_knn=5
train_s=[]
test_s=[]
data_2 = np.zeros((trial * len(stim), len(N_neuron)))
y = np.array([trial * [j] for j in range(len(stim))]).flatten()  # desired output
tim_linspace = np.linspace(0.00,0.2,20)
for tt in [0.02]:
    for i1,i in enumerate(stim):
        for tr in range(trial):

            df = pd.read_csv('data_output/Ex_1/7-NOC/{}.csv'.format(i * trial + tr))
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
    classifier = KNeighborsClassifier(n_neighbors=n_knn)
    # classifier= SVC()  # gamma='auto'
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
#%% Ex_1 equal indentation - spike time (VRd) - classifier (KNN(k=5)-SVM) time window
from elephant.spike_train_dissimilarity import van_rossum_dist, victor_purpura_dist
import quantities as pq

tau=1* pq.ms

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
for tt in tim_linspace:
    spike_t=[]
    for i in stim:

        for tr in range(trial):
            df = pd.read_csv('data_output/Ex_1/7-NOC/{}.csv'.format(i * trial+tr))
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
# ##  ############################################################extra code

# pos = plt.imshow(data_vrd,cmap='jet')
# plt.colorbar(pos)

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn import manifold, datasets, decomposition, discriminant_analysis
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

data = decomposition.PCA(n_components=n_pc).fit_transform(data_vrd)  # PCA on 'data_1'
# data=data_vrd
# ##  ######################
knn = KNeighborsClassifier(n_neighbors=n_knn)
cv2=StratifiedKFold(n_splits=5, random_state=79, shuffle=True)
# cv = ShuffleSplit(n_splits=c_v, test_size=t_size, random_state=3)
cv_results = cross_validate(knn, data, y, cv=cv2, return_train_score=True)
print(cv_results['test_score'].mean())
print(cv_results['test_score'].std())
print('train score is ',cv_results['train_score'],'mean and std are ',"{0:.2f}".format(cv_results['train_score'].mean()),"{0:.2f}".format(cv_results['train_score'].std()))
print('test score is ',cv_results['test_score'],'mean and std are ',"{0:.2f}".format(cv_results['test_score'].mean()),"{0:.2f}".format(cv_results['test_score'].std()))

y_pred = cross_val_predict(knn, data, y, cv=cv2)
conf_mat = confusion_matrix(y, y_pred)
# print(afferent,n_fingers,'F')

print('confusion matrix is \n',conf_mat)

pos = plt.imshow(conf_mat,cmap='hot_r')
plt.colorbar(pos)
#%%  vectorization of spikes in time windows
t1=np.linspace(0,0.2,30)
vector=np.zeros((40,3,30))
for j in range(40):
    for k in range(3):
        for tt in range(29):
            qww=np.where((spike_t[j*3+k]>t1[tt]) & (spike_t[j*3+k]<t1[tt+1]))[0]
            vector[j,k,tt]=len(spike_t[j*3+k][qww])

#%% 3d plot of vectors

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
for j in [0,1,2]:

    trial=j
    ax.plot(vector[trial,0,:],vector[trial,1,:], vector[trial,2,:], label='parametric curve')
    ax.legend()
    plt.show()


#%% Plot ISI-Pressure /adel/wide-stim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython


get_ipython().run_line_magic('matplotlib', 'qt')
from scipy.signal import find_peaks

from elephant import statistics
#%%
dic={}

#%%
#                   P -- mix -- sutface
neuron=['/outN (V)', '/outN2 (V)', '/outN3 (V)']
time = 0.4
spike_t=[]
suma=[]
fi1=[1,2,3,4,5,6,7,8,9]
fi2=[i*2+10 for i in range(30)]
fi1=fi1+fi2
fi1=[i*4 for i in range(21)]

n_1_isi=[]
n_2_isi=[]
n_3_isi=[]
# obj="wide"
# obj="S-wide"
# obj="S-sharp"
obj="Sharp"
for j in fi1:
    # df = pd.read_csv('adel/dr_rahimi/{}/{}.csv'.format(obj,j))
    # df = pd.read_csv('adel/dr_rahimi/{}/{}.csv'.format(obj,j))
    # df = pd.read_csv('adel/dr_rahimi/{}/{}.csv'.format(obj,j))
    df = pd.read_csv('adel/dr_rahimi/{}/{}.csv'.format(obj,j))


    # plt.figure('plot_EX_1_ob_{}'.format(fig_name[j]))
    spike_t=[]
    for h in range(3):
        t = np.array(df['{}'.format(df.keys()[0])])
        sp = np.array(df[neuron[h]])
        qqw, _ = find_peaks(sp, height=1.1, distance=10)
        a = t[qqw] * 1000
        aa = a
        # aa = a[a < time]

        spike_t.append(a)
        suma.append(len(aa))
        # plt.plot(aa, [h+1+j for r in range(len(aa))], '.k')
        # plt.plot(aa, [h for r in range(len(aa))], '|b',markersize=40)
        # plt.xlim([0,0.2])
        # plt.ylim([-0.5,2.5])
    isi_neuron_1 = statistics.isi(spike_t[0]).mean()
    n_1_isi.append(isi_neuron_1)
    isi_neuron_2 = statistics.isi(spike_t[1]).mean()
    n_2_isi.append(isi_neuron_2)
    isi_neuron_3 = statistics.isi(spike_t[2]).mean()
    n_3_isi.append(isi_neuron_3)
bb=1
# color="red"
# mark='P'
# mark='*'
# mark='X'
mark='.'
dic['N_1_{}'.format(obj)]=n_1_isi
dic['N_2_{}'.format(obj)]=n_3_isi
dic['N_3_{}'.format(obj)]=n_2_isi
plt.figure("Nueron_1")
plt.plot(fi1,n_1_isi,'.k',marker=mark,alpha=bb)
plt.figure("Nueron_2")
plt.plot(fi1,n_3_isi,'.r',marker=mark,alpha=bb)
plt.figure("Nueron_3")
plt.plot(fi1,n_2_isi,'.b',alpha=bb,marker=mark)

#%%

from scipy.io import savemat
savemat("all_data.mat",dic)
#%%
import csv
with open('dict.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in dic.items():
       writer.writerow([key, value])
#%%

fig,ax=plt.subplots(3,1,sharex=True)
# ax[0].figure("Nueron_1")
ax[0].plot(fi1,n_1_isi,'.k')
# ax[0].xlabel("Pressure %")
# ax[0].set_ylabel("ISI")

# plt.figure("Nueron_2")
ax[1].plot(fi1,n_3_isi,'.r')
# plt.xlabel("Pressure %")
ax[1].set_ylabel("ISI")

# plt.figure("Nueron_3")
ax[2].plot(fi1,n_2_isi,'.b')
# ax[1].xlabel("Pressure %")
ax[2].set_xlabel("Pressure %")


    #plt.savefig('plot_EX_1_ob_{}'.format(fig_name[j]),dpi=500,quality=100)
    #plt.close()
# plt.figure('E3')
# plt.stem(suma)
#%%





