'''import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from columnflexpart.utils import select_boundary, optimal_lambda
from columnflexpart.classes.flexdataset import FlexDataset
from functools import partial
import geopandas as gpd
import cartopy.crs as ccrs
from typing import Optional, Literal, Union, Callable, Iterable, Any
from pathlib import Path
import bayesinverse
from columnflexpart.classes.inversion import InversionBioclass
from columnflexpart.utils.utils import optimal_lambda
from columnflexpart.classes.inversionCO import InversionBioclassCO
#from columnflexpart.utils.utilsCO import optimal_lambda
from matplotlib.colors import LogNorm 
from matplotlib.dates import DateFormatter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib import colors as mcolors'''
import pandas as pd
import pytest
import datetime 
import tqdm
import os
import xarray as xr

def select_relevant_times(date_min: datetime.datetime, date_max: datetime.datetime, path_to_CO2_predictions : str, path_to_CO_predictions: str): 
      '''returns: cropped predictionsCO and predicitons CO2 datasets for which measurements will be loaded later'''
      assert date_max >= date_min 
      predictionsCO2 = pd.read_pickle(path_to_CO2_predictions+'predictions.pkl')
      predictionsCO = pd.read_pickle(path_to_CO_predictions+'predictions3_CO.pkl') 
      mask = (predictionsCO2['time']>=date_min)&(predictionsCO2['time']<=date_max) # selbe maske, da CO und CO2 genau gleich sortiert 
      predictionsCO2_cut = predictionsCO2[mask]
      predictionsCO_cut = predictionsCO[mask]
      if date_min.date() < predictionsCO_cut['time'].min().date():
           print('Given date min '+str(date_min.date())+' is not in predictions (Minimum : '+str(predictionsCO_cut['time'].min())+')')
      if date_max.date() > predictionsCO_cut['time'].max().date():
           print('Given date max '+str(date_max.date())+' is not in predictions (Maximum : '+str(predictionsCO_cut['time'].max())+')')
      return predictionsCO2_cut, predictionsCO_cut

def get_footprints_K_matrix(footprint_paths: list[str]): 
    file = os.path.join(result_row["directory"], "Footprint_total_inter_time.nc")
    if not os.path.exists(file):
        continue        
    footprint: xr.DataArray = xr.load_dataarray(file)
    footprint = footprint.expand_dims("measurement").assign_coords(dict(measurement=[i]))# HERAUSFINDEN WAS MEASUREMENT ID MACHT!!!!!!!
    if not self.data_outside_month:
        footprint = footprint.where((footprint.time >= self.start) * (footprint.time < self.stop), drop=True)
    else:
        min_time = footprint.time.min().values.astype("datetime64[D]") if footprint.time.min() < min_time else min_time

    footprint = select_boundary(footprint, self.boundary)
    footprint = self.coarsen_data(footprint, "sum", None)
    footprints.append(footprint)
    # merging and conversion from s m^3/kg to s m^2/mol
    footprints = xr.concat(footprints, dim = "measurement")/100 * 0.029 #!!!!!!!!!!!!!!!!!!!!!
    # extra time coarsening for consistent coordinates 
    self.min_time = min_time
    if not self.time_coarse is None:
        footprints = footprints.coarsen({self.time_coord:self.time_coarse}, boundary=self.coarsen_boundary).sum()
    footprints = footprints.where(~np.isnan(footprints), 0)
    return footprints


def get_footprint_and_measurement(path_to_CO2_predictions : str, path_to_CO_predictions: str, date_min : datetime.datetime, date_max: datetime.datetime):

    #Returns pandas Dataframes with relevant times only - I SHOULD APPEND ALL PREDICITONS AND PREDICIONSCO FILES TO HAVE ONE WITH ALL TIME STEPS PER SPECIES 
    predictionsCO2_cut, predictionsCO_cut = select_relevant_times(date_min, date_max, path_to_CO2_predictions , path_to_CO_predictions)

    concentrationsCO2 = [] # measurement - background
    concentration_errsCO2 = []
    concentrationsCO = []
    concentration_errsCO = []
    measurement_id = []
    footprint_paths = []
    footprints = []
    
    #Loads Measurements, Measurement Uncertainty, Backgrounds, footprints paths for CO2
    for i, result_row in tqdm(predictionsCO2_cut.iterrows(), desc="Loading CO2 measurements", total=predictionsCO2_cut.shape[0]):
        concentrationsCO2.append(result_row["xco2_measurement"] - result_row["background"])
        concentration_errsCO2.append(result_row["measurement_uncertainty"])
        measurement_id.append(i)
        footprint_paths.append(result_row["directory"])
    
    #Loads Measurements, Measurement Uncertainty, Backgrounds, footprints paths for CO
    for i, result_row in tqdm(predictionsCO_cut.iterrows(), desc="Loading CO measurements", total=predictionsCO_cut.shape[0]):
        concentrationsCO.append(result_row["xco2_measurement"] - result_row["background"])
        concentration_errsCO.append(result_row["measurement_uncertainty"])
        measurement_id.append(i) # HERAUSFINDEN WAS MEASUREMENT ID MACHT!!!!!!!

    concentrations = xr.DataArray(
        data = concentrationsCO2+concentrationsCO,# append CO2 and CO
        dims = "measurement",
        coords = dict(measurement = measurement_id) # !!!!!!!!!
    )
    concentration_errs  = xr.DataArray(
        data = concentration_errsCO2+concentration_errsCO,# append CO2 and CO
        dims = "measurement",
        coords = dict(measurement = measurement_id) # !!!!!!!!!
    )

    footprints = get_footprints_K_matrix(footprint_paths)
    
    return footprints, concentrations, concentration_errs


    #print(predictionsCO2_cut.columns)
    #print(predictionsCO_cut)
    #return 1 


get_footprint_and_measurement('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/','/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/', 
                              datetime.datetime(year = 2019, month=12, day= 15), datetime.datetime(year = 2019, month = 12, day = 31))

'''
min_time = self.start
        concentrations = []
        concentration_errs = []
        measurement_id = []
        footprints = []
        for i, result_row in tqdm(self.results.iterrows(), desc="Loading footprints", total=self.results.shape[0]):
            concentrations.append(result_row["xco2_measurement"] - result_row[data_key])
            concentration_errs.append(result_row["measurement_uncertainty"])
            measurement_id.append(i)
            file = os.path.join(result_row["directory"], "Footprint_total_inter_time.nc")
            if not os.path.exists(file):
                continue        
            footprint: xr.DataArray = xr.load_dataarray(file)
            footprint = footprint.expand_dims("measurement").assign_coords(dict(measurement=[i]))
            if not self.data_outside_month:
                footprint = footprint.where((footprint.time >= self.start) * (footprint.time < self.stop), drop=True)
            else:
                min_time = footprint.time.min().values.astype("datetime64[D]") if footprint.time.min() < min_time else min_time

            footprint = select_boundary(footprint, self.boundary)
            footprint = self.coarsen_data(footprint, "sum", None)
            footprints.append(footprint)
        self.min_time = min_time
        concentrations = xr.DataArray(
            data = concentrations,
            dims = "measurement",
            coords = dict(measurement = measurement_id)
        )
        concentration_errs  = xr.DataArray(
            data = concentration_errs,
            dims = "measurement",
            coords = dict(measurement = measurement_id)
        )
        # merging and conversion from s m^3/kg to s m^2/mol
        footprints = xr.concat(footprints, dim = "measurement")/100 * 0.029
        # extra time coarsening for consistent coordinates 
        if not self.time_coarse is None:
            footprints = footprints.coarsen({self.time_coord:self.time_coarse}, boundary=self.coarsen_boundary).sum()
        footprints = footprints.where(~np.isnan(footprints), 0)
        return footprints, concentrations, concentration_errs'''


#for CO: 
#Inversion = InversionBioclassCO(
#    result_path="/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions3_CO.pkl",#predictions_GFED_cut_tto_AU.pkl",
#    month="2019-12", 
#    flux_path="/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/Regridded_1x1/",
#    bioclass_path= mask, #'OekomaskAU_AKbased_2",#Flexpart_version8_all1x1
 #   time_coarse = None,
#    boundary=[110.0, 155.0, -45.0, -10.0],
#    data_outside_month=False
#)


