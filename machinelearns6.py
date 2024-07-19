#created on Jul 11 to run models on server
#machine learning class file
#copied machinelearn class from jupyter lab file
import pandas as pd
import numpy as np

class machinelearn:
    def __init__(self, inputdf, feature_names):
        self.inputdf = inputdf
        self.feature_names = feature_names
    
    def set_maker(self, inputdf, split_type, year, feature_names, ts=0.2, p='n'):
        if split_type == 'year':
            train = inputdf[inputdf['Year']!=year]
            test = inputdf[inputdf['Year']==year]
        elif split_type == 'function':
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(inputdf, test_size=ts)
        if p == 'y':
            coef_list = []
            for v in feature_names:
                count = 1
                names = ([])
                while count <= 10: #assembling list of columns without typing them in
                    t1 = 'p{}0'.format(count)
                    t2 = '_{}'.format(v)
                    t3 = t1+t2
                    names.append(t3)
                    count +=1
                coef_list.extend(names) 
            feature_names = coef_list
        elif p == 'n':
            train = train.reset_index()
            test = test.reset_index()
        #separate into X and y and create dictionary
        xy = {}
        xy['X'] = train[feature_names].dropna().values #convert features into numpy array
        try:
            xy['y'] = train['yield'].dropna().values #convert into numpy array
            xy['y_test'] = test['yield'].dropna().values
        except KeyError:
            train['yield'] = np.nan
            test['yield'] = np.nan
            xy['y'] = train['yield']
            xy['y_test'] = test['yield']
        xy['X_test'] = test[feature_names].dropna().values
        

        return train, test, xy, feature_names 
    
    def ridgemodel(self, test, xy, A):
        from sklearn.linear_model import Ridge
        #get X and y
        X = xy.get('X')
        y = xy.get('y')
        # Fit the model
        ridgemodel = Ridge(alpha=A)
        ridgemodel.fit(X, y)
        score = ridgemodel.score(X, y)
        #get X_test and y_test
        X_test = xy.get('X_test')
        y_test = xy.get('y_test')
        #use X_test to predict values for y in y_test
        y_pred = pd.DataFrame(ridgemodel.predict(X_test),columns=['yield'],index=test.index)
        
        #collect returned values into dictionary
        modeldata = {}
        modeldata['score'] = score
        modeldata['y_test'] = y_test
        modeldata['y_pred'] = y_pred
        modeldata['model'] = 'ridge'
        return modeldata
    
    def stat_array(self, modeldata):
        y_test = modeldata.get('y_test')
        y_pred = modeldata.get('y_pred')
        #y_pred = y_pred2['yield'].values
        i = modeldata.get('type')

        def list_to_string(array):
        	['{:.4f}'.format(x) for x in array]
        	ls = array.tolist()
        	sstring = "!".join(map(str, ls)) #single string
        	return sstring
        
        stat_array = pd.DataFrame(index=[i],columns=['MAPE', 'MAE', 'RMSE', 'R^2', 'y_test', 'y_pred'])
        try:
            from sklearn.metrics import mean_squared_error
            rmse = mean_squared_error(y_test, y_pred, squared=False) #updated
            
            from sklearn.metrics import mean_absolute_error
            mae = mean_absolute_error(y_test, y_pred)
            
        
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 #code from web statology.org https://www.statology.org/mape-python/
            
            #print stats
            stat_array.loc[i,'R^2'] = modeldata.get('score')
            stat_array.loc[i, 'RMSE'] = rmse
            stat_array.loc[i, 'MAPE'] = mape
            stat_array.loc[i, 'MAE'] = mae
        except ValueError:
            pass
        yt_s = [ y_test ] #list_to_string(y_test)
        yp_s = [ y_pred ] #list_to_string(y_pred)
        stat_array.loc[i, 'y_test'] = yt_s
        stat_array.loc[i, 'y_pred'] = yp_s
        return stat_array

    def rfrmodel(self, test, xy, t):
        #get X and y from xy
        X = xy.get('X')
        y = xy.get('y')
        
        # Instantiate a Random Forest Model
        from sklearn.ensemble import RandomForestRegressor
        rfrmodel = RandomForestRegressor(n_estimators=t)  # 100 is default value for n_estimators
        rfrmodel.fit(X, y)
        #score = rfrmodel.score(X,y)
        
        #get X_test and y_test from xy
        X_test = xy.get('X_test')
        y_test = xy.get('y_test')
        #predict y_test in y_pred
        y_preddf = pd.DataFrame(rfrmodel.predict(X_test),columns=['yield'],index=test.index)
        y_pred = y_preddf['yield'].values
        modeldata = {}
        try:
            from sklearn.metrics import r2_score
            score = r2_score(y_test,y_pred)
            #collect returned values into dictionary
            
            modeldata['score'] = score
        except ValueError:
            modeldata['score'] = np.NaN
        modeldata['y_test'] = y_test
        modeldata['y_pred'] = y_pred
        modeldata['type'] = 'rfr'
        feat_i = rfrmodel.feature_importances_
        return modeldata, feat_i
    
    def merfmodel(self, train, test, xy, mi):
        #create model inputs from train and test
        X = xy.get('X')
        y = xy.get('y')
        X_test = xy.get('X_test')
        y_test = xy.get('y_test')
        Z_train = np.ones((len(X), 1))
        Z_test = np.ones((len(X_test), 1)) #added after finding merf documentation
        
        #copied from google colab Copy of SepProjectCode.ipynb
        from merf.merf import MERF
        #from catboost import CatBoostRegressor
        #regr = CatBoostRegressor(iterations=500, silent=True)
        import xgboost as xgb
        regr = xgb.XGBRegressor(num_parallel_tree = 85)
        merfmodel = MERF(regr, max_iterations=mi)

        # Mixed Effects Random Forest Training
        merf_cluster = 'ADM1_NAME'
        clusters_train = train[merf_cluster] #names of regions??
        clusters_test = test[merf_cluster]
        merfmodel.fit(X, Z_train, clusters_train, y)
        
        #predict y_hat - called y_pred for conveinence
        y_predarray = merfmodel.predict(X_test, Z_test, clusters_test) #separated into two lines after meeting
        y_preddf = pd.DataFrame(y_predarray, columns=['yield'], index=test.index)
        y_pred = y_preddf['yield'].values
        #collect returned values into dictionary
        
        modeldata = {}
        try:
            from sklearn.metrics import r2_score
            score = r2_score(y_test, y_pred)
            modeldata['score'] = score
        except:
            pass
        modeldata['y_test'] = y_test
        modeldata['y_pred'] = y_pred
        modeldata['type'] = 'merf'
        feat_i = merfmodel.trained_fe_model.feature_importances_
        return modeldata, feat_i
    
    def xgbreg(self, train, test, xy,pt):
        import xgboost as xgb
        #pt: num_parallel_trees
        X_test = xy.get('X_test')
        y_test = xy.get('y_test')
        X_train = xy.get('X')
        y_train = xy.get('y')
        dvalid = xgb.DMatrix(X_test, y_test) #didn't work when ran with XGBRegressor, switched to DMatrix
        dtrain = xgb.DMatrix(X_train, y_train)

        params = {'objective':'reg:squarederror','booster':'gbtree','num_parallel_tree':pt}
    #'colsample_bynode': trial.suggest_float('colsample_bynode', 1e-8, 0.999, log=True)
        bst = xgb.train(params=params,dtrain=dtrain,evals=[(dvalid, 'eval')])
        preds = bst.predict(dvalid)

        #metrics
        from sklearn import metrics
        score = metrics.r2_score(y_test,preds)
        #collect returned values into dictionary
        modeldata = {}
        modeldata['score'] = score
        modeldata['y_test'] = y_test
        modeldata['y_pred'] = preds
        modeldata['type'] = 'xgb'
        return modeldata
    
    def feat_imp_graph(self, df, W_or_A, model):
        import seaborn as sns
        import matplotlib.pyplot as plt
        mod_df = df.loc[(df['Model']==model)]
        if W_or_A == 'Weight':
            fi = mod_df.groupby(['Feature Name']).agg(**{"Weight":('Importance', 'sum')}).reset_index()
            title_ = 'Summed Feature Importances of Various {} Runs'.format(model)
        if W_or_A == 'Average':
            fi = mod_df.groupby(['Feature Name']).agg(**{"Average":('Importance', 'mean')}).reset_index()
            title_ = 'Averaged Feature Importances of Various {} Runs'.format(model)
        fi2 = fi.sort_values(by=[W_or_A], ascending=False)
        ax =sns.barplot(x="Feature Name", y=W_or_A, data=fi2[:15])
        #plot top 15 values
        for item in ax.get_xticklabels():
            item.set_rotation(90)
        plt.title(title_)
        plt.show()
        return ax
        
#will create separate function for feature generation
def input_extraction(feature_names, percentiles, csv, mon=False):
    #create function that gives options on how our input data looks and eventually returns a usable dataset for models
    #inputs - feature_names: list of all features from main 5 that we want included, should be as list type
    #percentiles (options are 'yes' or 'no') : decides if we want features split into percentiles or just left as is
    #csv = pd.read_csv('/gpfs/data1/cmongp1/ginsburga/sp24_2/kenya_maize_a3b.csv')
    #mon - this input decides if you would like to create subseasonal forecasts excluding certain months in the growing season
    if mon != False:
        #mon is input dataset for year where predictions are being generated
        #exclusion of data is specifically for testing data only
        test = pd.read_csv('/gpfs/data1/cmongp1/ginsburga/sp24_2/maize_eo_2023.csv')
        #test = mon.copy()
        #test = csv3[csv3.Year.isin([2019,2023])]
        test['date']=pd.to_datetime(test['time'])
        test['month'] = pd.DatetimeIndex(test['date']).month
        csv = test[test['month']<mon]
    if mon == False:
        #csv2 = csv[csv['crop_cal'] != 0]
        pass
    csv.sort_values(['ADM1_NAME','date'],inplace=True)
    try:
        df = csv[['ADM1_NAME', 'Year', 'yield', 'ndvi', 'tmax', 'tmin', 'esi', 'cmax','cmin','fldas']]
    except KeyError:
        df = csv[['ADM1_NAME', 'Year', 'ndvi', 'tmax', 'tmin', 'esi', 'cmax','cmin','fldas']]
        df['yield'] = np.nan
    res = df.Year.isin([2019]).any().any()
    res2 = df.loc[(df.Year <=2016)].any().any()
    if (res | res2):
        df.dropna(subset=['yield'],inplace=True)
    else:
        pass
    #df = csv[['ADM1_NAME', 'Year', 'yield', 'ndvi', 'tmin', 'esi','fldas']]
    if percentiles == 'no':
        # subset to the following columns: Season, yield, ndvi
        df_ml = df.groupby(['ADM1_NAME', 'Year']).agg(**{"ndvi":('ndvi', 'max'), "esi":('esi', 'mean'), "tmax":('tmax', 'mean'),"tmin":('tmin', 'mean'), "cmax":("cmax","sum"),"cmin":("cmin",'sum'), "fldas":('fldas','mean'),"yield":('yield', 'mean')})
        #df_ml = df.groupby(['ADM1_NAME', 'Year']).agg(**{"ndvi":('ndvi', 'max'), "esi":('esi', 'mean'), "tmin":('tmin', 'mean'), "fldas":('fldas','mean'), "yield":('yield', 'mean')})
        include = ['yield'] + feature_names
        df = df_ml[include].reset_index()
        # Drop any rows where any information is missing
    elif percentiles == 'yes':
        #divvy up variable info into percentiles
        df2 = df.ffill()
        groups = df2.groupby(['ADM1_NAME', 'Year']) #does something to align these two columns
        frames = []
        feature_names = ['ndvi', 'esi', 'tmax', 'tmin', 'cmax','cmin','fldas']
#groupshead = groups[0:5,:]

        for key, vals in groups: # keys refers to group by features, vals = all of the data in the df
            df_o= pd.DataFrame()
            df_o2 = pd.DataFrame()
            percentiles = ['p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p100']
    #percentiles = ['p40']
            vals.loc[:, 'Fraction Season'] = range(1, len(vals) + 1)
            vals.loc[:, 'Fraction Season'] = vals.loc[:, 'Fraction Season'] * 100 / len(vals)
            for var in feature_names:
                if var in vals:
                    for f in percentiles:
                        perc = int(f[1:])
                        closest = vals['Fraction Season'].sub(perc).abs().idxmin(skipna=True) #subtract percentile
            #print([vals.loc[closest][var]])
            #split
                        df_o[f'{f}_{var}'] = [vals.loc[closest][var]]
                        df_o['ADM1_NAME'] = vals.iloc[0,0]
                        df_o['Year'] = vals.iloc[0,1]
                        df_o['yield'] = vals['yield'].unique()[0]
                        df_o2= pd.concat([df_o2, df_o],axis=0)
                    frames.append(df_o)  #format of percentiles is translated to an array

            df_full = pd.concat(frames)      
    df_full['Year'] = df_full['Year'].astype(int)  #code was returning key error likely because years were strings
    #print(df_full)
    if df_full['yield'].isna().sum() != 0:
        df_full.drop(columns=['yield'],inplace=True)
    else:
        pass
    df_full = df_full.query('2000 < Year < 2024') #modify frame
    df_full= df_full.reset_index()
    return df_full


