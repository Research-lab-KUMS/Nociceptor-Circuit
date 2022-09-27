# Adel Parvizi-Fard (adel.p1372@yahoo.com)


import os
import numpy as np
import xlrd

wb = xlrd.open_workbook('data_7_all_same_pressure.xls')
i=0
for j in range(80):
    os.mkdir('models/{}'.format(i+j))
    sheet = wb.sheet_by_index(i+j)
    # sheet.cell_value(0, 0)
    # print(sheet.row_values(0))
    for x in range(25):
        f = open("models/{}/{}.p".format(i+j,x), "w")
        data=(np.array(sheet.row_values(x))*1.8)/1023
        v=len(data)
        t=np.linspace(0,200e-6,v-20)
        for y in range(10,v-10):
            f.write('{:.6e} {:.6e}\n'.format(t[y-10],data[y]))
        f.close()
#%% for test DELETE
import os
import numpy as np
import xlrd

wb = xlrd.open_workbook('data/data_7_all_same_pressure.xls')
i=0
for j in [5,35]:
    os.mkdir('models/{}'.format(i+j))
    sheet = wb.sheet_by_index(i+j)
    # sheet.cell_value(0, 0)
    # print(sheet.row_values(0))
    for x in range(25):
        f = open("models/{}/{}.p".format(i+j,x), "w")
        data=(np.array(sheet.row_values(x))*1.8)/1023
        v=len(data)
        t1=np.linspace(0,2e-3,v-20)
        for y in range(10,v-10):
            f.write('{:.6e} {:.6e}\n'.format(t1[y-10],data[y]))
        f.close()


#%% revised in order to normalized pressure
import os
import numpy as np
import xlrd

wb = xlrd.open_workbook('data/data_7_all_same_pressure.xls')

coe=[3385/8745, 3385/8045, 3385/5545, 1, 3385/8245, 3385/10945, 3385/11045, 3385/7803]
for j in range(80):
    sheet = wb.sheet_by_index(j)
    print(sum(sheet.col_values(25)))

#%%


import xlutils.copy
wb2 = xlutils.copy.copy(wb)
wb2.save('data_7_all_same_pressure.xls')


#%%
import xlrd

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
wb = xlrd.open_workbook('robotic data/data.xls')

for j in range(8):

    sheet = wb.sheet_by_index(j*10+5)
    print(sum(sheet.col_values(25)))
    ax = plt.subplot(2, 4, j + 1)
    ax.imshow(np.array(sheet.col_values(27)).reshape(5, 5), vmin=0, vmax=1023, cmap='jet' ) #, interpolation='gaussian'
    plt.title(j)
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
from scipy.signal import find_peaks
suma=[]
for i in range(8):
    for x in range(2):
        df = pd.read_csv('notebooks/noc6/{}.csv'.format(i*10+x))
       

        for h in range(3):
            t = np.array(df['{}'.format(df.keys()[0])])
            sp=np.array(df['{}'.format(df.keys()[h+1])])
            qqw ,_= find_peaks(sp, height=1.1,distance=150)
            aa=t[qqw]*1000
            suma.append(len(aa))
            plt.plot(aa,[(i*2+x)*3+h+1 for j in range(len(aa))],'.k')
plt.figure(2)
plt.plot(suma,marker='*')

#%% classification
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, discriminant_analysis
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'qt')
ac_all = []
import numpy as np
import scipy.io
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

n_pc = 3
n_knn = 5
c_v = 5

N_neuron = [i for i in range(1, 24)]  ##   #(0-4 NC) #(5-11 SA)  #(12-23 RA)
trial = 2
stim = 8

data_2 = np.zeros((trial * stim, 4))
y = np.array([trial * [j] for j in
              range(stim)]).flatten()  # desired output 1-60  (1-20 is 0 ,20-40 is 1 and 40-60 is 2)
#  ###################################################################  load data
# plt.figure(2)
ex=1

for i in range(stim):
    for tr in range(trial):
        df = pd.read_csv('notebooks/NOC2/{}.csv'.format(i * trial + tr+5))
        for h in range(4):
            t = np.array(df['{}'.format(df.keys()[0])])
            sp = np.array(df['{}'.format(df.keys()[h + 1])])
            qqw, _ = find_peaks(sp, height=1.1, distance=150)
            data_2[i * trial + tr, h] = len(qqw)


#%% equal pressure

import os
import numpy as np
import xlrd

wb = xlrd.open_workbook('P_data_1.xls')
summm=[]
for j in range(16):
    sheet = wb.sheet_by_index(j*10+7)
    # sheet._cell_values=np.ndarray.tolist(np.array(sheet._cell_values)*coe[int(j/10)])  #coe[int(j/10)]
    print(sum(sheet.col_values(28)))
    summm.append(sum(sheet.col_values(25)))
summm=np.array(summm)
for j in range(4):
    summm[j*4:(j+1)*4]=summm[j*4+1]/summm[j*4:(j+1)*4]
for j in range(160):
    sheet = wb.sheet_by_index(j)
    sheet._cell_values=np.ndarray.tolist(np.array(sheet._cell_values)*summm[int(j/10)])  #coe[int(j/10)]

#%%

import xlutils.copy
wb2 = xlutils.copy.copy(wb)
wb2.save('P_data_1.xls')

#%%
import xlrd
import numpy as np
import matplotlib.pyplot as plt
wb = xlrd.open_workbook('P_data.xls')

for j in range(16):

    sheet = wb.sheet_by_index(j*10+2)
    
    ax = plt.subplot(4, 4, j + 1)
    ax.imshow(np.array(sheet.col_values(27)).reshape(5, 5), vmin=0, vmax=1024, cmap='jet' ) #, interpolation='gaussian'
    plt.title(j)
#%% data preparation ex_3
import os
import numpy as np
import xlrd
wb = xlrd.open_workbook('robotic data/data.xls')
target = 5000
final = []
for j in range(80):
    sheet = wb.sheet_by_index(j)

    final.append(np.array(sheet._cell_values))
    


#%% for test 1, 37 trial summation of col 25 is 5000  or 7000
i = 0
ma=[final[i].max() for i in range(80)]
ma=max(ma)
for jj,j in enumerate(range(80)):
    os.mkdir('models/{}'.format(i+j))
    sheet = final[jj]
    for x in range(25):
        f = open("models/{}/{}.p".format(i+j,x), "w")
        data=(sheet[x,:]*1.8)/ma
        v=len(data)
        t=np.linspace(0,200e-6,v-20)
        for y in range(10,v-10):
            f.write('{:.6e} {:.6e}\n'.format(t[y-10],data[y]))
        f.close()
#%%
