# Guide to umd-agroml
### Steps to modeling: 
1. [Step 1: Prepare RS Data](#prepareRS)
   (a) - Extraction of input features via Google Earth Engine
   (b) - Scripts to combine GEE Data and add yield
2. [Step 2: Prepare ML models](#prepareML)
  (a) [Select Models To Use](#subparagraph3)
- Convert EO data into descriptive, model-compatible features
- Perform feature importance
- Perform hyperparameter tuning
3. [Run ML models](#runmodels)
4. [Analyze Model Output](#analysis)
- Analyze CSV output metrics
- create graphics

## Step 1: Prepare Remote Sensing Data <a name="prepareRS"></a>

### Part (a) - Extraction of input features via Google Earth Engine

This model was created with the intention of relying exclusively on inputs derived from remote sensing (RS) products, to assess their feasibility in producing results without requiring extensive field campaigns. This data has been sourced in the past exclusively using Google Earth Engine (GEE) scripts, however methods are available for extracting data not available on GEE as well. The variables chosen in this example cover both meterological influences to crop growth, and metrics of ecological conditions related to crop growth. These include air temperature, precipitation, soil moisture, evaporative stress index (ESI) and normalized differential vegetation index (NDVI). 

Copy repository from this link: https://code.earthengine.google.com/?accept_repo=users/acgins/inputdata_keadm1 (not shared 5/27)
  Needed files:
  * Shapefile of regional boundaries (Administrative 1 County-level for Kenya)
  * Shapefile of crop mask (maize for this model)
    
### Part (b) - Scripts to combine GEE Data and Yield
  1. modvars.py (extracting each variable, concattenating multiple variables)
  2. Transposing data frame and adding yield and crop calendar info
     
## Step 2: Prepare scripts to run machine learning algorithms <a name="prepareML"></a>

### Part (a) - Select machine learning models to use <a name="subparagraph3"></a>

Find regression models, best to employ those in python. Refer to literature.

### Part (b) - Convert EO data into model-compatible features

The model will generate 1 yield prediction based on each year and unique administrative region, in this case we rely on county-level (KE Admin 1), that all data is available for. As the highest resolution of data for a feature is daily, we need to find the best way to describe each spread with 1 or more aggregation metrics. This example creates a feature for monthly averages of each variable. Percentiles may also be effective.
