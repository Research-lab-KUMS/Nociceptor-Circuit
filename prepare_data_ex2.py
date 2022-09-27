import os
import numpy as np
import xlrd

#wb = xlrd.open_workbook('data/P_data_1.xls')
wb = xlrd.open_workbook('data/data_7_all_same_pressure.xls')
stepp=[0.01*i for i in range(100)]
for i in range(100):
    for j in [79]:
        print(int(i))

        os.mkdir('Adel/2_sharp/{}'.format(i))

        sheet = wb.sheet_by_index(j)
        # sheet.cell_value(0, 0)
        # print(sheet.row_values(0))
        for x in range(25):
            f = open("Adel/2_sharp/{}/{}.p".format(i,x), "w")
            data=((np.array(sheet.row_values(x))*1.8*4)/1023)*stepp[i]
            v=len(data)
            t=np.linspace(0,2e-4,v-20)
            for y in range(10,v-10):
                f.write('{:.6e} {:.6e}\n'.format(t[y-10],data[y]))
            f.close()

#%% revised in order to normalized pressure
import os
import numpy as np
import xlrd

wb = xlrd.open_workbook('data/data_7_all_same_pressure.xls')
# coe=[1,1,1,1.27,1,1,1,1.27]
# coe=1010/930
coe=[3385/8745, 3385/8045, 3385/5545, 1, 3385/8245, 3385/10945, 3385/11045, 3385/7803]
for j in range(80):
    sheet = wb.sheet_by_index(j)
    # sheet._cell_values=np.ndarray.tolist(np.array(sheet._cell_values)*coe[int(j/10)])  #coe[int(j/10)]
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
wb = xlrd.open_workbook('data/P_data_1.xls')
# coe=[1,1,1.6,2,7,1.3,1,1,1.4]
for j in range(12):

    sheet = wb.sheet_by_index(j*10+5)
    print(max(sheet.col_values(25)))
    ax = plt.subplot(3, 4, j + 1)
    ax.imshow(np.array(sheet.col_values(27)).reshape(5, 5), vmin=0, vmax=700, cmap='jet' ) #, interpolation='gaussian'
    plt.title(j)

#%% experiment 2    four objects 4 different pressure

# Writing to an excel
# sheet using Python
import numpy as np
import xlwt
from xlwt import Workbook

# Workbook is created
wb = Workbook()
shapes=[0,3,4,7]
# add_sheet is used to create sheet.
for p in range(4):
    for sh in shapes:
        for tr in range(10):
            sheet1 = wb.add_sheet('{}_P_{}_obj_{}_tr_{}'.format(p*40+shapes.index(sh) * 10 + tr,p+1,sh,tr))
            a = np.load('data/raw data_2/pressure{0}/sh{1}_tr{2}.npz'.format(p+1, sh, tr))
            b = a['layer_1']
            for i in range(25):
                for j in range(np.shape(b)[0]):
                    sheet1.write(i, j, b[j, i])

wb.save('data/P_data_2.xls')


#%% equal pressure

import os
import numpy as np
import xlrd

wb = xlrd.open_workbook('data/P_data_1.xls')
# coe=[1,1,1,1.27,1,1,1,1.27]
# coe=1010/930
# coe=[3385/8745, 3385/8045, 3385/5545, 1, 7803/8245, 7803/10945, 7803/11045, 1]
# coe=[[5844/10543,1,5432/5475,1], [4474/7187,1,6968/6151,1], [5333/7879,1,9008/7960,1], [5844/10543,1,10508/9251,1]]
summm=[]
for j in range(16):
    sheet = wb.sheet_by_index(j*10+4)
    # sheet._cell_values=np.ndarray.tolist(np.array(sheet._cell_values)*(2000/3600))  #coe[int(j/10)]
    print(sum(sheet.col_values(25)))
    summm.append(sum(sheet.col_values(25)))
# summm=np.array(summm)
# for j in range(4):
#     summm[j*4:(j+1)*4]=summm[j*4+1]/summm[j*4:(j+1)*4]
# for j in range(160):
#     sheet = wb.sheet_by_index(j)
#     sheet._cell_values=np.ndarray.tolist(np.array(sheet._cell_values)*summm[int(j/10)])  #coe[int(j/10)]

#%%

import xlutils.copy
wb2 = xlutils.copy.copy(wb)
wb2.save('data/P_data_2.xls')

#%%
import xlrd

import numpy as np
import matplotlib.pyplot as plt
wb = xlrd.open_workbook('P_data.xls')
# coe=[1,1,1.6,2,7,1.3,1,1,1.4]
for j in range(16):

    sheet = wb.sheet_by_index(j*10+2)
    # print(max(sheet.col_values(27)))
    ax = plt.subplot(4, 4, j + 1)
    ax.imshow(np.array(sheet.col_values(27)).reshape(5, 5), vmin=0, vmax=1024, cmap='jet' ) #, interpolation='gaussian'
    plt.title(j)
#%% data preparation
import os
import numpy as np
import xlrd
# wb = xlrd.open_workbook('data/data_7_all_same_pressure.xls')
wb = xlrd.open_workbook('data/P_data_1.xls')
final=[]
target=[1000, 3000, 5000, 7000]
trials=20
p_level=4
for i in range(p_level):
    for j in range(trials):
        sheet = wb.sheet_by_index(j)
        su = sum(sheet.col_values(25))
        # print(su)
        coe=target[i]/su
        final.append(np.array(sheet._cell_values)*coe)
        # print(sum(np.array(sheet._cell_values).flatten())/52)

#%%
i=0
ma=[final[i].max() for i in range(trials*p_level)]
ma=max(ma)
for jj,j in enumerate(np.arange(trials*p_level)):
    os.mkdir('models/{}'.format(i+j))
    sheet = final[jj]
    # sheet.cell_value(0, 0)
    # print(sheet.row_values(0))
    for x in range(25):
        f = open("models/{}/{}.p".format(i+j,x), "w")
        data=(sheet[x,:]*1.8)/ma
        v=len(data)
        t=np.linspace(0,200e-6,v-20)
        for y in range(10,v-10):
            f.write('{:.6e} {:.6e}\n'.format(t[y-10],data[y]))
        f.close()