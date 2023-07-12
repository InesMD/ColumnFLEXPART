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
import numpy as np 
import pytest
import datetime 
import tqdm
import os
import xarray as xr
from columnflexpart.utils import select_boundary
from columnflexpart.classes import InversionBioclass

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
    footprints = []
    for i, filepath in tqdm(footprint_paths.iterrows(), desc="Loading footprints", total=len(footprint_paths)):
        file = os.path.join(filepath, "Footprint_total_inter_time.nc")
        if not os.path.exists(file):
            continue        
        footprint: xr.DataArray = xr.load_dataarray(file)
        footprint = footprint.expand_dims("measurement").assign_coords(dict(measurement=[i]))
        #if not self.data_outside_month:
        #    footprint = footprint.where((footprint.time >= self.start) * (footprint.time < self.stop), drop=True)
        #else:
        min_time = footprint.time.min().values.astype("datetime64[D]") if footprint.time.min() < min_time else min_time
        footprint = select_boundary(footprint, [110.0, 155.0, -45.0, -10.0])
        footprint = self.coarsen_data(footprint, "sum", None)
        footprints.append(footprint)
    
    # merging and conversion from s m^3/kg to s m^2/mol
    footprints = xr.concat(footprints, dim = "measurement")/100 * 0.029 

    zero_block = xr.zeros_like(footprints)
    zero_block_left = zero_block.assign_coords(measurement = ((zero_block.measurement+ zero_block.measurement.values.max() + 1)))
    block_left = xr.concat([footprints,zero_block_left], dim = "measurement")

    zero_block_right = zero_block.assign_coords(bioclass = ((zero_block.bioclass+ zero_block.bioclass.values.max() + 1)))
    block_bottom_right = footprints.assign_coords(measurement = ((footprints.measurement+ footprints.measurement.values.max() + 1)), 
                                                  bioclass = ((footprints.bioclass+ footprints.bioclass.values.max() + 1)) )
    block_right = xr.concat([zero_block_right,block_bottom_right], dim = "measurement")

    total_matrix = xr.concat([block_left, block_right], dim = "bioclass")
    # extra time coarsening for consistent coordinates 
    #self.min_time = min_time
    if not self.time_coarse is None:
        total_matrix = total_matrix.coarsen({self.time_coord:self.time_coarse}, boundary=self.coarsen_boundary).sum()
    total_matrix = total_matrix.where(~np.isnan(total_matrix), 0) 

    return total_matrix


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
    
    m = len(measurement_id)
    print('Measurment number single tracers: '+str(m))

    #Loads Measurements, Measurement Uncertainty, Backgrounds, footprints paths for CO
    for i, result_row in tqdm(predictionsCO_cut.iterrows(), desc="Loading CO measurements", total=predictionsCO_cut.shape[0]):
        concentrationsCO.append(result_row["xco2_measurement"] - result_row["background"])
        concentration_errsCO.append(result_row["measurement_uncertainty"])
        measurement_id.append(m+i) 

    concentrations = xr.DataArray(
        data = concentrationsCO2+concentrationsCO,# append CO2 and CO
        dims = "measurement",
        coords = dict(measurement = measurement_id) 
    )
    concentration_errs  = xr.DataArray(
        data = concentration_errsCO2+concentration_errsCO,# append CO2 and CO
        dims = "measurement",
        coords = dict(measurement = measurement_id) 
    )

    footprints = get_footprints_K_matrix(footprint_paths)
    
    return footprints, concentrations, concentration_errs

def get_flux_CO2():
     #CO2 
    flux_files = []
    for date in np.arange(self.min_time.astype("datetime64[D]"), self.stop):
        date_str = str(date).replace("-", "")
        flux_files.append(self.flux_path.parent / (self.flux_path.name + f"{date_str}.nc"))
    flux = xr.open_mfdataset(flux_files, drop_variables="time_components").compute()
    assert not (bio_only and no_bio), "Choose either 'bio_only' or 'no_bio' not both"
    if bio_only:
        flux = flux.bio_flux_opt + flux.ocn_flux_opt
    elif no_bio:
        flux = flux.fossil_flux_imp
    else:
        flux = flux.bio_flux_opt + flux.ocn_flux_opt + flux.fossil_flux_imp + flux.fire_flux_imp
    flux = select_boundary(flux, self.boundary)
    flux_mean = self.coarsen_data(flux, "mean", self.time_coarse)
    flux_err = self.get_flux_err(flux, flux_mean)
    return flux_mean, flux_err

def get_flux_CO(): 
#CO: 
    total_flux = xr.DataArray()
    first_flux = True
    for year in range(2019,2020):
        cams_flux_bio = xr.DataArray()
        cams_flux_ant = xr.Dataset()
        for sector in ['AIR_v1.1', 'BIO_v3.1', 'ANT_v4.2']:
            if sector == 'ANT_v4.2':
                cams_file = "/CAMS-AU-"+sector+"_carbon-monoxide_"+str(year)+"_sum_regr1x1.nc"
                flux_ant_part = xr.open_mfdataset(
                    str(self.flux_path)+cams_file,
                    combine="by_coords",
                    chunks="auto",
                    )
                flux_ant_part = flux_ant_part.assign_coords({'year': ('time', [year*i for i in np.ones(12)])})
                flux_ant_part = flux_ant_part.assign_coords({'MonthDate': ('time', [datetime(year=year, month= i, day = 1) for i in np.arange(1,13)])})
                cams_flux_ant = flux_ant_part

            elif sector == 'BIO_v3.1': 
                cams_file = "/CAMS-AU-"+sector+"_carbon-monoxide_2019_regr1x1.nc"   
                #print(str(self.flux_path)+cams_file)         
                flux_bio_part = xr.open_mfdataset(
                str(self.flux_path)+cams_file,
                combine="by_coords",
                chunks="auto",
                )      
                flux_bio_part = flux_bio_part.emiss_bio.mean('TSTEP')
                flux_bio_part = flux_bio_part.assign_coords(dict({'time': ('time',[0,1,2,3,4,5,6,7,8,9,10,11] )}))
                flux_bio_part = flux_bio_part.assign_coords({'year': ('time', [year*i for i in np.ones(12)])})
                flux_bio_part = flux_bio_part.assign_coords({'MonthDate': ('time', [datetime(year=year, month= i, day = 1) for i in np.arange(1,13)])})
                cams_flux_bio = flux_bio_part                
        total_flux_yr = (cams_flux_ant['sum'] + cams_flux_bio)/0.02801 # from kg/m^2/s to molCO/m^2/s 
        if first_flux:
                first_flux = False
                total_flux = total_flux_yr
        else:
            total_flux = xr.concat([total_flux, total_flux_yr], dim = 'time')

    # select flux data for time period given: 
    total_flux = total_flux.where(total_flux.MonthDate >= pd.to_datetime(self.start), drop = True )
    total_flux = total_flux.where(total_flux.MonthDate <= pd.to_datetime(self.stop), drop = True )

    #cams_flux = cams_flux[:, :, 1:]
    flux = select_boundary(total_flux, self.boundary)

    # create new flux dataset with value assigned to every day in range date start and date stop 
    dates = [pd.to_datetime(self.start)]
    date = dates[0]
    while date < pd.to_datetime(self.stop):
        date += datetime.timedelta(days = 1)
        dates.append(pd.to_datetime(date))
    fluxes = xr.DataArray(data = np.zeros((len(dates), len(flux.latitude.values), len(flux.longitude.values))), dims = ['time', 'latitude', 'longitude'])
    count = 0
    for i,dt in enumerate(dates): 
        for m in range(len(flux.MonthDate.values)): # of moths 
            if dt.month == pd.to_datetime(flux.MonthDate[m].values).month: 
            #    if bol == True: 
                fluxes[i,:,:] =  flux[m,:,:]#/0.02801 
    
    fluxes = fluxes.assign_coords({'time': ('time', dates), 'latitude': ('latitude', flux.latitude.values), 'longitude': ('longitude', flux.longitude.values)})
    flux_mean = self.coarsen_data(fluxes, "mean", self.time_coarse)
    flux_err = self.get_flux_err(fluxes, flux_mean)

    # Error of mean calculation
    return flux_mean, flux_err


def get_flux(): # to be wrapped together # flux mean has coords bioclass and week 
    
    flux_meanCO2, flux_errCO2 = get_flux_CO2()
    flux_meanCO, flux_errCO = get_flux_CO() # err checken!!!!!!!!!!!!!!
    andere Koordinaten 
    xr.concat
    return flux_mean, flux_err
   

    















#get_footprint_and_measurement('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/','/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/', 
#                              datetime.datetime(year = 2019, month=12, day= 15), datetime.datetime(year = 2019, month = 12, day = 31))









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


