# modified version of agrmol_tv_run.py
#by Alana Ginsburg 6-20-2024

import pandas as pd
import numpy as np



#which model variations do we want to run??
#do train test splits by year

#Means/percentiles: 
#Monthly splits: 
selperfn = ['p90_ndvi','p40_tmin','p20_tmin','p10_cmin','p60_cmax','p80_cmax','p90_cmax','p30_tmin','p90_tmax','p40_fldas','p20_esi','p60_esi']
from machinelearns6 import machinelearn as ml
from machinelearns6 import input_extraction

#Baseline five feature model for years 2001-2016
feature_names1 = ['ndvi', 'esi', 'tmax', 'tmin', 'cmax','cmin','fldas']


#modifying this to also use the stat array function, rfr, and ridge
#include all years 2001-2016
file1 = pd.read_csv('/gpfs/data1/cmongp1/ginsburga/sp24_2/kenya_maize_a3b.csv')
years = np.arange(2001, 2017, 1, dtype=int)
dek2 = input_extraction(feature_names1, 'yes', file1, False)
dek2.to_csv('kenya_maize_dek2.csv')

#extract only columns with features that have highest importances

#model with percentile features
#run model and get feature importances and statistics
#create empty dataframe to append feature importances to
fimp_mon = pd.DataFrame()

#columns =  ['ADM1_NAME', 'Year', 'yield'] + selperfn
#df_sm2 = df_sm[columns]


train_dek2 = pd.read_csv('/gpfs/data1/cmongp1/ginsburga/sp24_2/kenya_maize_dek2.csv')
train_dekd = train_dek2.dropna()
train_dek = train_dekd.drop_duplicates()

#xy = {}
#xy['X'] = train_dek[selperfn].dropna().values #convert features into numpy array
#xy['y'] = train_dek['yield'].dropna().values #convert into numpy array 
stat_mon = pd.DataFrame()

for y in years: #loop to get f_int in dataframe
    #define class
    ml_per = ml(train_dek, selperfn)
    #set maker- create your own test train splits
    train, test, xy, fn_p = ml_per.set_maker(ml_per.inputdf, 'year',y, selperfn, ts=0.2, p='n')
    
    #test_dek = input_extraction(feature_names1,'yes',file1)
    #test_dekd = test_dek.drop_duplicates(inplace=False)
    if y == 2019:
        for i in test_dekd.columns:
            if i.find("ndvi") == 4:
                test_dekd[i] = test_dekd[i]/10000
            else:
                pass
    #xy['X_test'] = test_dekd[selperfn].values
    #xy['y_test'] = test_dekd['yield'].values  
    #Dont need to seperate out a test set because we will just be using 2019 data
    #run rfr model 
    rfrdata, fir = ml_per.rfrmodel(test, xy, 81)
    rfrstats = ml_per.stat_array(rfrdata)
    
    #run merf model
    merfdata, fim = ml_per.merfmodel(train, test, xy, 75)
    merfstats = ml_per.stat_array(merfdata)
    
    
    #run xgb model
    xgbdata = ml_per.xgbreg(train, test,xy,15)
    xgbstats = ml_per.stat_array(xgbdata)
 
    xgbstats['Year'] = y
    stat_mon = pd.concat([stat_mon,rfrstats,xgbstats],axis=0)#merfstats
    #print(xgbstats['y_pred'],len(rfrdata.get('y_pred')))
stat_mon.to_csv('eos2001-2016xgbper2.csv', index=True, header=True)
