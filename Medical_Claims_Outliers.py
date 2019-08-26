# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
from scipy import stats
from sklearn import linear_model
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.float_format = '{:.2f}'.format
import os
from sklearn import datasets,preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import squarify
import time,datetime,dateutil
from sklearn.svm import SVC,SVR
#-----------------------------------------------
print(__doc__)
##==============================================
Source_2017_H1_Sheet = 'JAN_JUNE_2017'
Source_2017_H2_Sheet = 'JULY_DEC_2017'
Source_2018_Sheet = '2018'
AfiyaCliams_itemized_2017_H1, AfiyaCliams_itemized_2017_H2, AfiyaCliams_itemized_2018 = pd.DataFrame() , pd.DataFrame(), pd.DataFrame()
# Load 2017 first Half sheet
%time AfiyaCliams_itemized_2017_H1 = pd.read_excel('../input/SPEND-TAKAFUL-2017-2018.xlsx',sheet_name=Source_2017_H1_Sheet)
AfiyaCliams_itemized_2017_H2 = pd.read_excel('../input/SPEND-TAKAFUL-2017-2018.xlsx',sheet_name=Source_2017_H2_Sheet)
AfiyaCliams_itemized_2018 = pd.read_excel('../input/SPEND-TAKAFUL-2017-2018.xlsx',sheet_name=Source_2018_Sheet)
#----------------------------------------

AfiyaCliams_itemized_2017_H1['POLICY_START_YEAR'] = 2017
AfiyaCliams_itemized_2017_H2['POLICY_START_YEAR'] = 2017
AfiyaCliams_itemized_2018['POLICY_START_YEAR'] = 2018

AfiyaCliams_itemized = AfiyaCliams_itemized_2017_H1.append(AfiyaCliams_itemized_2017_H2).append(AfiyaCliams_itemized_2018)

del AfiyaCliams_itemized_2017_H1, AfiyaCliams_itemized_2017_H2, AfiyaCliams_itemized_2018

AfiyaCliams_itemized_filtered = AfiyaCliams_itemized.drop(['UNDERWRITINGYEAR','FAMILYHEAD','FAMILYHEADNAME','BATCHNO','APPROVALNO','PROVIDERINVOICENO','PROVIDERINVOICEDATE'\
                          ,'REQUESTNATURE','EXGRATIA','COPAYDEDUCTIBLE','DISCOUNT','GROSSREQUESTEDAMT','PARAMOUNT','PAYMENTSTATUS'],axis=1)

#Exclude 0 value claims
AfiyaCliams_itemized_filtered = AfiyaCliams_itemized_filtered[AfiyaCliams_itemized_filtered.FINALAMT >= 1]
#AfiyaCliams_itemized.count()=1,557,562

#preparatios
AfiyaCliams_itemized_filtered.rename(columns={'MEMBERIDNO':'CARD_NUMBER'},inplace=True)
AfiyaCliams_itemized_filtered['TREATMENT_WEEKDAY'] = AfiyaCliams_itemized_filtered.TREATMENTDATE.dt.weekday_name

# exclude IPD
AfiyaCliams_itemized_filtered = AfiyaCliams_itemized_filtered[AfiyaCliams_itemized_filtered.IP_OP == 'OPD']

# exclude SERVICETYPE=='ROOMTYPE'
AfiyaCliams_itemized_filtered = AfiyaCliams_itemized_filtered[AfiyaCliams_itemized_filtered.SERVICETYPE != 'ROOMTYPE']

#Remove empty Speciality records
AfiyaCliams_itemized_filtered.dropna(subset=['SPECIALITYDEPARTMENT'],inplace=True)

#2017 filter
AfiyaCliams_itemized_filtered_2017 = AfiyaCliams_itemized_filtered[AfiyaCliams_itemized_filtered.POLICY_START_YEAR==2017]

#2018 filter
AfiyaCliams_itemized_filtered_2018 = AfiyaCliams_itemized_filtered[AfiyaCliams_itemized_filtered.POLICY_START_YEAR==2018]

groupByColList=['PROVIDERGROUP','PROVIDERNAME','PROVIDERTYPE',
                                                   'ATTENDINGDOCTORNAME','SPECIALITYDEPARTMENT','CLAIMNO']

df_2017 = AfiyaCliams_itemized_filtered[AfiyaCliams_itemized_filtered.POLICY_START_YEAR==2017].groupby(groupByColList,as_index=False).agg({'FINALAMT':'sum'})#as_index=False
groupByColList.pop(5)
df_2017 = df_2017.groupby(groupByColList, as_index=False).agg({'FINALAMT':['sum']})
df_2017.reset_index(level=0, inplace=True)
df_2017.columns = ['index','PROVIDERGROUP','PROVIDERNAME','PROVIDERTYPE','ATTENDINGDOCTORNAME','SPECIALITYDEPARTMENT','FINALAMT']

ProvGCatalogue = pd.Series(df_2017.PROVIDERGROUP.unique())
ProvGCatalogueIDs = pd.Series(range(1,ProvGCatalogue.size+1))

ProvTypeCatalogue = pd.Series(df_2017.PROVIDERTYPE.unique())
ProvTypeCatalogueIDs = pd.Series(range(1,ProvTypeCatalogue.size+1))

ProvNameCat = pd.Series(df_2017.PROVIDERNAME.unique())
ProvNameCatalogueIDs = pd.Series(range(1,ProvNameCat.size+1))

DoctNameCat = pd.Series(df_2017.ATTENDINGDOCTORNAME.unique())
DoctNameCatalogueIDs = pd.Series(range(1,DoctNameCat.size+1))

SpecialityCat = pd.Series(df_2017.SPECIALITYDEPARTMENT.unique())
SpecialityCatIDs = pd.Series(range(1,SpecialityCat.size+1))


ProvGCatDF = pd.DataFrame(data={'ID':ProvGCatalogueIDs,'PROVIDERGROUP':ProvGCatalogue})
ProvTypeCatDF = pd.DataFrame(data={'ID':ProvTypeCatalogueIDs,'PROVIDERTYPE':ProvTypeCatalogue})
ProvNameCatDF = pd.DataFrame(data={'ID':ProvNameCatalogueIDs,'PROVIDERNAME':ProvNameCat})
DoctNameCatDF = pd.DataFrame(data={'ID':DoctNameCatalogueIDs,'ATTENDINGDOCTORNAME':DoctNameCat})
SpecialityCatDF = pd.DataFrame(data={'ID':SpecialityCatIDs,'SPECIALITYDEPARTMENT':SpecialityCat})

df_2017 = df_2017.merge(ProvGCatDF,how='left',on='PROVIDERGROUP')
df_2017.rename(columns={'ID':'PROVIDERGROUP_ID'},inplace=True)
df_2017 = df_2017.merge(ProvTypeCatDF,how='left',on='PROVIDERTYPE')
df_2017.rename(columns={'ID':'PROVIDERTYPE_ID'},inplace=True)
df_2017 = df_2017.merge(ProvNameCatDF,how='left',on='PROVIDERNAME')
df_2017.rename(columns={'ID':'PROVIDERNAME_ID'},inplace=True)
df_2017 = df_2017.merge(DoctNameCatDF,how='left',on='ATTENDINGDOCTORNAME')
df_2017.rename(columns={'ID':'ATTENDINGDOCTORNAME_ID'},inplace=True)
df_2017 = df_2017.merge(SpecialityCatDF,how='left',on='SPECIALITYDEPARTMENT')
df_2017.rename(columns={'ID':'SPECIALITYDEPARTMENT_ID'},inplace=True)

df_2017.drop(columns=['index','PROVIDERGROUP','PROVIDERNAME','PROVIDERTYPE','ATTENDINGDOCTORNAME'
                      ,'SPECIALITYDEPARTMENT'],inplace=True)
                      
df_2017['FINALAMT'] = round(df_2017['FINALAMT'])


CostCatalogue = pd.Series(['< 100','101-300','301-500','501-700','701-900'
                           ,'901-1000','1001-1200','1201-1500','1501-1700','1701-2000','2001-3000','3001-4000'
                           ,'4001-5000','5001-7000','7001-10000'
                           ,'10001-15000','15001-20000','20001-30000','30001-50000'
                           ,'50001-70000','70001-100000','> 100000'])
CostCatalogueIDs = pd.Series(range(0,CostCatalogue.size))

CostCatDF = pd.DataFrame(data={'ID':CostCatalogueIDs,'CLAIMSCATEGORY':CostCatalogue})



df_2017['CLAIMSCATEGORY'] = np.nan

for i,row in df_2017.iterrows():
    if row['FINALAMT'] <= 100: 
        df_2017.loc[i,'CLAIMSCATEGORY']='< 100'
    elif (row['FINALAMT'] >100) & (row['FINALAMT'] <=300): 
        df_2017.loc[i,'CLAIMSCATEGORY']='101-300'
    elif (row['FINALAMT'] >300) & (row['FINALAMT'] <=500): 
        df_2017.loc[i,'CLAIMSCATEGORY']='301-500'
    elif (row['FINALAMT'] >500) & (row['FINALAMT'] <=700): 
        df_2017.loc[i,'CLAIMSCATEGORY']='501-700'
    elif (row['FINALAMT'] >700) & (row['FINALAMT'] <=900): 
        df_2017.loc[i,'CLAIMSCATEGORY']='701-900'
    elif (row['FINALAMT'] >900) & (row['FINALAMT'] <=1000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='901-1000'
    elif (row['FINALAMT'] >1000) & (row['FINALAMT'] <=1200): 
        df_2017.loc[i,'CLAIMSCATEGORY']='1001-1200'
    elif (row['FINALAMT'] >1200) & (row['FINALAMT'] <=1500): 
        df_2017.loc[i,'CLAIMSCATEGORY']='1201-1500'
    elif (row['FINALAMT'] >1500) & (row['FINALAMT'] <=1700): 
        df_2017.loc[i,'CLAIMSCATEGORY']='1501-1700'
    elif (row['FINALAMT'] >1700) & (row['FINALAMT'] <=2000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='1701-2000'
    elif (row['FINALAMT'] >2000) & (row['FINALAMT'] <=3000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='2001-3000'
    elif (row['FINALAMT'] >3000) & (row['FINALAMT'] <=4000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='3001-4000'
    elif (row['FINALAMT'] >4000) & (row['FINALAMT'] <=5000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='4001-5000'
    elif (row['FINALAMT'] >5000) & (row['FINALAMT'] <=7000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='5001-7000'
    elif (row['FINALAMT'] >7000) & (row['FINALAMT'] <=10000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='7001-10000'
    elif (row['FINALAMT'] >10000) & (row['FINALAMT'] <=15000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='10001-15000'
    elif (row['FINALAMT'] >15000) & (row['FINALAMT'] <=20000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='15001-20000'
    elif (row['FINALAMT'] >20000) & (row['FINALAMT'] <=30000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='20001-30000'
    elif (row['FINALAMT'] >30000) & (row['FINALAMT'] <=50000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='30001-50000'
    elif (row['FINALAMT'] >50000) & (row['FINALAMT'] <=70000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='50001-70000'
    elif (row['FINALAMT'] >70000) & (row['FINALAMT'] <=100000): 
        df_2017.loc[i,'CLAIMSCATEGORY']='70001-100000'
    else: 
        df_2017.loc[i,'CLAIMSCATEGORY']='> 100000'
    
#df_2017 = df_2017.merge(CostCatDF,how='left',on='CLAIMSCATEGORY')

#df_2017.rename(columns={'ID':'CLAIMSCATEGORY_ID'},inplace=True)
df_2017.drop(columns=['FINALAMT'],inplace=True)

X = df_2017.iloc[:,0:df_2017.shape[1]-1]
y = df_2017.CLAIMSCATEGORY

dd = pd.DataFrame({'CAT':y})
for i,row in dd.iterrows():
    if (row['CAT'] == '> 100000'):
        dd.loc[i,'CAT'] = '< 100'
y=dd['CAT']
#---------------------------------
yFD = pd.DataFrame({'CLAIMSCATEGORY':y})
yFD = yFD.merge(CostCatDF,how='left',on='CLAIMSCATEGORY')
yFD.rename(columns={'ID':'CLAIMSCATEGORY_ID'},inplace=True)
yFD.drop(columns=['CLAIMSCATEGORY'],inplace=True)
y = yFD['CLAIMSCATEGORY_ID']

##----------------------------------------------
def f1():
    pass

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# Example settings
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest", IsolationForest(behaviour='new',
                                         contamination=outliers_fraction,
                                         random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))]

# Define datasets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
datasets = [X]

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))

plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
rng = np.random.RandomState(42)

for i_dataset, X in enumerate(datasets):
    # Add outliers
    X = np.concatenate([X, rng.uniform(low=-6, high=6,
                       size=(n_outliers, 2))], axis=0)

    for name, algorithm in anomaly_algorithms:
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        # fit the data and tag outliers
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)

        # plot the levels lines and the points
        if name != "Local Outlier Factor":  # LOF does not implement predict
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()