# Guide to umd-agroml
### Steps to modeling: 
1. [Step 1: Prepare RS Data](#prepareRS)
   - [(a) Extraction of input features via Google Earth Engine](#subparagraph1)
   - [(b) Python package requirements and installation](#subparagraph2)
   - [(c) Scripts to combine GEE Data and add yield](#subparagraph3)
2. [Step 2: Prepare ML models](#prepareML)
  - (a) [Select Models To Use](#subparagraph4)
  - (b) Convert EO data into descriptive, model-compatible features
  - (c) Perform feature importance
  - (d) Perform hyperparameter tuning
3. [Run ML models](#runmodels)
4. [Analyze Model Output](#analysis)
  - Analyze CSV output metrics
  - create graphics

## Step 1: Prepare Remote Sensing Data <a name="prepareRS"></a>

### Part (a) - Extraction of input features via Google Earth Engine <a name="subparagraph1"></a>

This model was created with the intention of relying exclusively on inputs derived from remote sensing (RS) products, to assess their feasibility in producing results without requiring extensive field campaigns. This data has been sourced in the past exclusively using Google Earth Engine (GEE) scripts, however methods are available for extracting data not available on GEE as well. The variables chosen in this example cover both meterological influences to crop growth, and metrics of ecological conditions related to crop growth. These include air temperature, precipitation, soil moisture, evaporative stress index (ESI) and normalized differential vegetation index (NDVI). 

Copy repository from this link: https://code.earthengine.google.com/?accept_repo=users/acgins/inputdata_keadm1 (not shared 5/27)
  Needed files:
  * Shapefile of regional boundaries (Administrative 1 County-level for Kenya)
  * Shapefile of crop mask (maize for this model)
    
### Part (b) - Python package requirements and installation <a name="subparagraph2"></a>
Package requirements:
- pandas
- numpy
- scikit-learn
- <a href="https://manifoldai.github.io/merf/">pip install merf</a>
- <a href="https://xgboost.readthedocs.io/en/stable/install.html#conda">pip install xgboost</a>
- pip install glob2

Most code for this model can be executed as scripts without an IDE and just a code editor. However, the scripts have been organized into Jupyter Notebooks for this repository, and can be ran with various local platforms such as Jupyter lab/notebook via Anaconda Navigator or Visual Studio Code. Find more information about Jupyter notebooks <a href="https://docs.jupyter.org/en/latest/">here</a>. If using one of these platforms, all python packages needed can be organized within a conda environment. See x file for a list of some required package installation commands.

These notebooks can also be adapted to be run with Google CoLab.

### Part (c) - Scripts to combine GEE Data and Yield <a href="modvars_share.py">modvars_share.py</a>  
  1. Extracting each variable, concattenating multiple variables
  2. Transposing data frame and adding yield and crop calendar info
     
## Step 2: Prepare scripts to run machine learning algorithms <a name="prepareML"></a>
<p> Scripts or notebooks that run models randomforestregressor, xgboost, or merf use functions from the machinelearn class defined in machinelearns6.py. Once the class is instantialized, functions for each of these models can be run, with room for modification to hyperparameters and training and testing data, interpreted in the form of dictionaries of numpy arrays. </p>

### Part (a) - Select machine learning models to use <a name="subparagraph3"></a>

Find regression models, best to employ those in python. Refer to literature.

### Part (b) - Convert EO data into model-compatible features

The model will generate 1 yield prediction based on each year and unique administrative region, in this case we rely on county-level (KE Admin 1), that all data is available for. As the highest resolution of data for a feature is daily, we need to find the best way to describe each spread with 1 or more aggregation metrics. This example creates a feature for monthly averages of each variable. Percentiles may also be effective.
