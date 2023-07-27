from columnflexpart.classes import InversionBioclass
from columnflexpart.classes import Inversion
from typing import Optional, Literal, Union, Callable, Iterable, Any
from pathlib import Path
from columnflexpart.utils import select_boundary
import pandas as pd
import xarray as xr
import datetime
import bayesinverse
import os
import tqdm
import numpy as np 

Pathlike = Union[Path, str]
Boundary = Optional[tuple[float, float, float, float]]
Timeunit = Literal["week", "day"]
Coarsefunc = Union[Callable, tuple[Callable, Callable], str, tuple[str, str]]
Concentrationkey = Literal["background", "background_inter", "xco2", "xco2_inter"]


class CoupledInversion(InversionBioclass):# oder von InversionBioClass? 
    def __init__(self, 
        result_pathCO: Pathlike,
        result_pathCO2: Pathlike,
        flux_pathCO, 
        flux_pathCO2,
        bioclass_path: Pathlike,
        month: str,
        date_min: datetime.datetime,
        date_max = datetime.datetime, 
        time_coarse: Optional[int] = None, 
        coarsen_boundary: str = "pad",
        time_unit: Timeunit = "week",
        boundary: Boundary = None,
        concentration_key: Concentrationkey = "background_inter",
        data_outside_month: bool = False
       ): 
        
        self.pathCO = result_pathCO
        self.pathCO2 = result_pathCO2
        self.n_eco: Optional[int] = None
        self.bioclass_path = bioclass_path
        self.bioclass_mask = select_boundary(self.get_bioclass_mask(), boundary)
        self.spatial_valriables = ["bioclass"]
        self.start = np.datetime64(month).astype("datetime64[D]")
        self.stop = (np.datetime64(month) + np.timedelta64(1, "M")).astype("datetime64[D]")
        self.date_min = date_min 
        self.date_max = date_max 
        self.flux_pathCO = Path(flux_pathCO)
        self.flux_pathCO2 = Path(flux_pathCO2)
        self.time_coarse = time_coarse
        self.coarsen_boundary = coarsen_boundary
        self.time_unit = time_unit
        self.boundary = boundary
        self.concentration_key = concentration_key
        self.data_outside_month = data_outside_month

        #nötig? 
        self.min_time = None
        self.fit_result: Optional[tuple] = None
        self.predictions: Optional[xr.DataArray] = None
        self.predictions_flat: Optional[xr.DataArray] = None
        self.prediction_errs: Optional[xr.DataArray] = None
        self.prediction_errs_flat: Optional[xr.DataArray] = None
        self.l_curve_result: Optional[tuple] = None
        self.alpha: Optional[Iterable[float]] = None
        self.reg: Optional[bayesinverse.Regression] = None
        ####

        self.time_coord, self.isocalendar = self.get_time_coord(self.time_unit)
        self.footprints, self.concentrations, self.concentration_errs = self.get_footprint_and_measurement(self.concentration_key)
        self.F = self.get_F_matrix()
        self.flux, self.flux_errs = self.get_flux()

        self.coords = self.footprints.stack(new=[self.time_coord, *self.spatial_valriables]).new
        self.footprints_flat = self.footprints.stack(new=[self.time_coord, *self.spatial_valriables])

        self.flux_flat = self.flux.stack(new=[self.time_coord, *self.spatial_valriables])
        self.flux_errs_flat = self.flux_errs.stack(new=[self.time_coord, *self.spatial_valriables])

    def select_relevant_times(self): 
        '''returns: cropped predictionsCO and predicitons CO2 datasets for which measurements will be loaded later'''
        assert self.date_max >= self.date_min 
        predictionsCO2 = pd.read_pickle(self.pathCO2)#+'predictions.pkl')
        predictionsCO = pd.read_pickle(self.pathCO)#+'predictions3_CO.pkl') 
        mask = (predictionsCO2['time']>=self.date_min)&(predictionsCO2['time']<=self.date_max+datetime.timedelta(days=1)) # selbe maske, da CO und CO2 genau gleich sortiert 
        predictionsCO2_cut = predictionsCO2[mask].reset_index()
        predictionsCO_cut = predictionsCO[mask].reset_index()
        #print(predictionsCO['time'].max())
        if self.date_min.date() < predictionsCO_cut['time'].min().date():
            print('Given date min '+str(self.date_min.date())+' is not in predictions (Minimum : '+str(predictionsCO_cut['time'].min())+')')
        if self.date_max.date() > predictionsCO_cut['time'].max().date():
            print('Given date max '+str(self.date_max.date())+' is not in predictions (Maximum : '+str(predictionsCO_cut['time'].max())+')')
        return predictionsCO2_cut, predictionsCO_cut
    
    def get_footprints_H_matrix(self,footprint_paths: list[str]): 
        footprints = []
        i = 0
        min_time = self.start
        for filepath in tqdm(footprint_paths, desc="Loading footprints"):#, total=len(footprint_paths)):
            file = os.path.join(filepath, "Footprint_total_inter_time.nc")
            if not os.path.exists(file):
                continue        
            footprint: xr.DataArray = xr.load_dataarray(file)
            footprint = footprint.expand_dims("measurement").assign_coords(dict(measurement=[i]))
            if not self.data_outside_month:
                footprint = footprint.where((footprint.time >= self.start) * (footprint.time < self.stop), drop=True)
                #print([pd.Timestamp(x).to_pydatetime() for x in footprint.time.values])
                #footprint = footprint.where([(pd.Timestamp(x).to_pydatetime() >= self.start) for x in footprint.time.values]*footprint.time , drop=True)
                #footprint = footprint.where(([pd.Timestamp(x).to_pydatetime() < self.stop for x in  footprint.time.values]), drop = True)
            #else:
            #print(footprint.time.min().values)
            min_time = footprint.time.min().values.astype("datetime64[D]") if footprint.time.min() < min_time.astype("datetime64[D]") else min_time.astype("datetime64[D]")
            footprint = select_boundary(footprint, self.boundary)
            footprint = self.coarsen_data(footprint, "sum", None)
            footprints.append(footprint)
            i += 1
        
        # merging and conversion from s m^3/kg to s m^2/mol
        footprints = xr.concat(footprints, dim = "measurement")/100 * 0.029 
        footprints2 = footprints.copy(deep = True)
        footprints2 = footprints2.assign_coords(bioclass = ((footprints.bioclass+ footprints.bioclass.values.max() + 1)))
        double_footprints = xr.concat([footprints,footprints2],dim = 'bioclass') # is upper left corner

        zero_block = xr.zeros_like(double_footprints)
        zero_block_left = zero_block.assign_coords(measurement = ((zero_block.measurement+ zero_block.measurement.values.max() + 1)))
        block_left = xr.concat([double_footprints,zero_block_left], dim = "measurement")

        zero_block_right = zero_block.assign_coords(bioclass = ((zero_block.bioclass+ zero_block.bioclass.values.max() + 1)))
        block_bottom_right = double_footprints.assign_coords(measurement = ((double_footprints.measurement+ double_footprints.measurement.values.max() + 1)), 
                                                    bioclass = ((double_footprints.bioclass+ double_footprints.bioclass.values.max() + 1)) )
        block_right = xr.concat([zero_block_right,block_bottom_right], dim = "measurement")

        total_matrix = xr.concat([block_left, block_right], dim = "bioclass")
  

        # extra time coarsening for consistent coordinates 
        self.min_time = min_time
        if not self.time_coarse is None:
            total_matrix = total_matrix.coarsen({self.time_coord:self.time_coarse}, boundary=self.coarsen_boundary).sum()
        total_matrix = total_matrix.where(~np.isnan(total_matrix), 0) 

        return total_matrix

    def get_footprint_and_measurement(self, concentration_key):#path_to_CO2_predictions : str, path_to_CO_predictions: str, date_min : datetime.datetime, date_max: datetime.datetime):

        #Returns pandas Dataframes with relevant times only - I SHOULD APPEND ALL PREDICITONS AND PREDICIONSCO FILES TO HAVE ONE WITH ALL TIME STEPS PER SPECIES 
        predictionsCO2_cut, predictionsCO_cut = self.select_relevant_times()
        #print(predictionsCO2_cut.shape)
        #print(predictionsCO_cut)

        concentrationsCO2 = [] # measurement - background
        concentration_errsCO2 = []
        concentrationsCO = []
        concentration_errsCO = []
        measurement_id = []
        footprint_paths = []
        footprints = []
        #print(predictionsCO2_cut.shape[0])
        #Loads Measurements, Measurement Uncertainty, Backgrounds, footprints paths for CO2
        for i, result_row in tqdm(predictionsCO2_cut.iterrows(), desc="Loading CO2 measurements", total=predictionsCO2_cut.shape[0]):
            concentrationsCO2.append((result_row["xco2_measurement"] - result_row[concentration_key])*1e-6)
            concentration_errsCO2.append(result_row["measurement_uncertainty"]*1e-6)
            measurement_id.append(i)
            footprint_paths.append(result_row["directory"])
        
        m = len(measurement_id)

        #Loads Measurements, Measurement Uncertainty, Backgrounds, footprints paths for CO
        for i, result_row in tqdm(predictionsCO_cut.iterrows(), desc="Loading CO measurements", total=predictionsCO_cut.shape[0]):
            concentrationsCO.append((result_row["xco2_measurement"] - result_row[concentration_key])*1e-9)
            concentration_errsCO.append(result_row["measurement_uncertainty"]*1e-9)
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
        # sollte passen (footprints - H matrix)
        footprints = self.get_footprints_H_matrix(footprint_paths)
        
        return footprints, concentrations, concentration_errs
    
    def get_F_matrix(): 

