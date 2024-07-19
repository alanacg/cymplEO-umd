import pandas as pd
import numpy as np
import glob

#part 1 combining GEE files
#meant to work from blank file based on folders named by variable where data is located

#m = pd.read_csv('/gpfs/data1/cmongp1/ginsburga/sp24_2/ms1216.csv')
master= pd.DataFrame(columns=['time','ADM1_NAME','value','var'])
#put path name with * to wherever extracted GEE data organized by year are saved
#files = glob.glob('/gpfs/data1/cmongp1/ginsburga/sp24_2/newf/2021/*') #my path
files = glob.glob('ke_data/2006/*') #example
print([f[-15:-7] for f in files]) #find out what order the files are in, adjust indices based on length of path string

ele = 'nan'
var = []
while ele != 'done':   
    ele = input('enter names of var returned individually/in order. type done when finished.')
    var.append(ele)

vars = ['cmax','esi',
 'tmax',
 'cmin','cmax',
 'tmin',
 'fldas',
 'ndvi',
 'tmax',
 'cmin',
 'tmin']
#Transpose each file so that they are in long format with 1 column containing all admin 1 values
#vars = var[1:-2] #run if you used the input method to give file names
n = 0
mods = {}
for f in files:
	try:
		df1 = pd.read_csv(f)
	#try:
		df1.drop(columns=['.geo','system:index'],inplace=True) #remove meaningless columns
		df2 = df1.melt(id_vars=['time'],var_name='ADM1_NAME')
		df2['var'] = vars[n]
		df2.sort_values(by=['time','ADM1_NAME'],inplace=True)
		if vars[n] == 'cmin' or 'cmax' or 'tmin' or 'tmax':
			try:
				a = mods.get(vars[n])			#append same variable if necessary
				mods[vars[n]] = pd.concat([a,df2],axis=0,ignore_index=True)
			except:
				continue
		else:
			mods[vars[n]]=df2
		n+=1
	#master = pd.concat([m, df2])
	except IndexError:
		continue

print(mods.keys())

adding columns for date and year and filling
for v in vars:
	vdf = mods.get(v)
	vdf['date']=pd.to_datetime(vdf['time'])
	vdf['year']=np.NaN
	for i,j in vdf.iterrows():
		vdf.loc[i,'year'] = vdf.loc[i,'date'].year #year from datetime object
	#if v=='tmin' or v=='tmax': #could make these edits to concattenated version instead
	#	vdf['value']= vdf['value']-273.15
	if v=='ndvi':
		vdf['value']=vdf['value']/10000
	mods[v] = vdf

#extracting variable values from dictionaries and compiling all into 1 dataframe
vars_init = list(mods.keys())
dfc = mods.get('tmax')
for v in vars_init:
	#vdf = pd.read_csv('vars/full_{}_ke.csv'.format(v)) #code if files are fully organized in dir
	vdf2 = mods.get(v)
	adm1 = pd.unique(vdf2['ADM1_NAME']).tolist()
	dts = pd.unique(vdf2['time']).tolist()
	print(v)
	for a in adm1:
		y = 365
		for d in dts:
        	#identify index of admin 2 region in admins file
			try:
				ind = vdf2.loc[(vdf2['time'] == d)&(vdf2['ADM1_NAME']==a)].index
				#grab the admin 1 region for that district
				value = vdf2.loc[ind, 'value'].tolist()
				inputt = value[0]
			except IndexError:
				y-=1
        #get index of d
			place = dfc.loc[(dfc['time'] == d)&(dfc['ADM1_NAME']==a)].index
			#print(d,a, place, v)
        #list admin 1 region
			dfc.loc[place, v] = inputt

#Part 3: adding yield values and crop cal values to 365 day variables file (yield from UCSB/UAH SERVIR)
files = glob.glob('/gpfs/data1/cmongp1/ginsburga/sp24_2/vars/full_eo/*')
filescc = {}
for f in files:
    print(f)
    df = pd.read_csv(f)
    df['date']=pd.to_datetime(df['time'])
    #f = glob.glob.read_csv('vars/full*.csv')
    #ndvi = pd.read_csv('vars/full_ndvi_ke.csv')
    #adm1 = pd.unique(df['ADM1_NAME']).tolist()
	
    df['crop_cal'] = np.ones(len(df['date']))
    for i,j in df.iterrows():
        m = df.loc[i,'date'].month
        if (m >= 9) or (m< 3):
            df.loc[i,'crop_cal'] = 0
        else:
            pass
    df.rename(columns={'year':'Year'}, inplace=True)
    y = df.loc[0,'Year']
    filescc[y] = df

dfc['date']=pd.to_datetime(dfc['time'])
yld = pd.read_csv('/gpfs/data1/cmongp1/ginsburga/sp24_2/fewsnet_yield_c2001.csv')
adm1 = pd.unique(dfc['ADM1_NAME']).tolist()
yrs = np.arange(2001,2017,1)
yrs.append(2019)
#add yield column from other file into the dataframe
for a in adm1:
	for y in yrs:
	try:
		ind = yld.loc[(yld['Year'] == y)&(yld['admin_1']==a)].index
		#grab the year for that district
		value = yld.loc[ind, 'value'].tolist()
		inputt = value[0]
	except IndexError:
		#y-=1
		continue
    #get index of d
	place = dfc.loc[(dfc['year'] == y)&(dfc['ADM1_NAME']==a)].index
    #list admin 1 region
	dfc.loc[place, 'yield'] = inputt	
print('done with yield')

dfc['crop_cal'] = np.ones(len(df['date']))
for i,j in dfc.iterrows():
    m = dfc.loc[i,'date'].month
    if (m >= 9) or (m< 3):
        dfc.loc[i,'crop_cal'] = 0
    else:
        pass
dfc.rename(columns={'year':'Year'}, inplace=True)
print('done with crop_cal')

dfc.to_csv('/gpfs/data1/cmongp1/ginsburga/sp24_2/kenya_maize_yr.csv') #change path