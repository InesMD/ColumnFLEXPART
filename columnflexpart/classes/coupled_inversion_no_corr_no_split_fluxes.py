import pandas as pd
import numpy as np 
import pytest
import datetime 
from tqdm.auto import tqdm
import os
import xarray as xr
from columnflexpart.utils import select_boundary
from columnflexpart.classes import InversionBioclass
from columnflexpart.classes import Inversion

from typing import Optional, Literal, Union, Callable, Iterable, Any
from pathlib import Path
import bayesinverse

Pathlike = Union[Path, str]
Boundary = Optional[tuple[float, float, float, float]]
Timeunit = Literal["week", "day"]
Coarsefunc = Union[Callable, tuple[Callable, Callable], str, tuple[str, str]]
Concentrationkey = Literal["background", "background_inter", "xco2", "xco2_inter"]

class CoupledInversion(InversionBioclass):# oder von InversionBioClass? 
    # INIT VERFOLLSTÄMNDIGEN UND ALLES TESTEN!!!!!!!!
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

    def get_footprints_K_matrix(self,footprint_paths: list[str]): 
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
        print(footprints)
        zero_block = xr.zeros_like(footprints)
        zero_block_left = zero_block.assign_coords(measurement = ((zero_block.measurement+ zero_block.measurement.values.max() + 1)))
        block_left = xr.concat([footprints,zero_block_left], dim = "measurement")

        zero_block_right = zero_block.assign_coords(bioclass = ((zero_block.bioclass+ zero_block.bioclass.values.max() + 1)))
        block_bottom_right = footprints.assign_coords(measurement = ((footprints.measurement+ footprints.measurement.values.max() + 1)), 
                                                    bioclass = ((footprints.bioclass+ footprints.bioclass.values.max() + 1)) )
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

        footprints = self.get_footprints_K_matrix(footprint_paths)
        return footprints, concentrations, concentration_errs

    def get_flux_CO2(self,bio_only=False, no_bio=False):
        #CO2 
        flux_files = []
        for date in np.arange(self.min_time.astype("datetime64[D]"), self.stop):
            date_str = str(date).replace("-", "")
            flux_files.append(self.flux_pathCO2.parent / (self.flux_pathCO2.name + f"{date_str}.nc"))
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
        #print('CO2 flux')
        #print(flux_mean.mean())
        return flux_mean, flux_err

    def get_flux_CO(self): 
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
                        str(self.flux_pathCO)+cams_file,
                        combine="by_coords",
                        chunks="auto",
                        )
                    flux_ant_part = flux_ant_part.assign_coords({'year': ('time', [year*i for i in np.ones(12)])})
                    flux_ant_part = flux_ant_part.assign_coords({'MonthDate': ('time', [datetime.datetime(year=year, month= i, day = 1) for i in np.arange(1,13)])})
                    cams_flux_ant = flux_ant_part

                elif sector == 'BIO_v3.1': 
                    cams_file = "/CAMS-AU-"+sector+"_carbon-monoxide_2019_regr1x1.nc"   
                    #print(str(self.flux_path)+cams_file)         
                    flux_bio_part = xr.open_mfdataset(
                    str(self.flux_pathCO)+cams_file,
                    combine="by_coords",
                    chunks="auto",
                    )      
                    flux_bio_part = flux_bio_part.emiss_bio.mean('TSTEP')
                    flux_bio_part = flux_bio_part.assign_coords(dict({'time': ('time',[0,1,2,3,4,5,6,7,8,9,10,11] )}))
                    flux_bio_part = flux_bio_part.assign_coords({'year': ('time', [year*i for i in np.ones(12)])})
                    flux_bio_part = flux_bio_part.assign_coords({'MonthDate': ('time', [datetime.datetime(year=year, month= i, day = 1) for i in np.arange(1,13)])})
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

        #print('CO flux')
        #print(flux_mean.mean())

        # Error of mean calculation
        return flux_mean, flux_err


    def get_flux(self): # to be wrapped together # flux mean has coords bioclass and week 
        
        flux_meanCO2, flux_errCO2 = self.get_flux_CO2()
        flux_meanCO, flux_errCO = self.get_flux_CO() # err checken!!!!!!!!!!!!!!

        #adapt coordinates and concatenate fluxes
        flux_meanCO = flux_meanCO.assign_coords(bioclass = ((flux_meanCO.bioclass+ flux_meanCO.bioclass.values.max() + 1)))
        flux_mean = xr.concat([flux_meanCO2, flux_meanCO], dim = "bioclass")
    
        flux_errCO = flux_errCO.assign_coords(bioclass = ((flux_errCO.bioclass+ flux_errCO.bioclass.values.max() + 1)))
        flux_err = xr.concat([flux_errCO2, flux_errCO], dim = "bioclass")
        return flux_mean, flux_err
    

    def get_regression(
        self,
        x: Optional[np.ndarray] = None,
        yerr: Optional[Union[np.ndarray, xr.DataArray, float, list[float]]] = None, 
        xerr: Optional[Union[np.ndarray, list]] = None,
        with_prior: Optional[bool] = True,
        alpha: Optional[float] = None,
        ) -> bayesinverse.Regression:
        """Constructs a Regression object from class attributes and inputs 

        Args:
            x (Optional[np.ndarray], optional): Alternative values for prior. Defaults to None.
            yerr (Optional[Union[np.ndarray, xr.DataArray, float, list[float]]], optional): Alternative values for y_covariance. Defaults to None.
            xerr (Optional[Union[np.ndarray, list]], optional): Alternative values for x_covariances. Defaults to None.
            with_prior (Optional[bool], optional): Switch to use prior. Defaults to True.
            alpha (Optional[float]): Regulatization value

        Returns:
            bayesinverse.Regression: Regression model
        """        
        concentration_errs = self.concentration_errs.values
        if not yerr is None:
            if isinstance(yerr, float):
                yerr = np.ones_like(concentration_errs) * yerr
            elif isinstance(yerr, xr.DataArray):
                if yerr.size == 1:
                    yerr = np.ones_like(concentration_errs) * yerr.values
                else:
                    yerr = yerr.values
            concentration_errs = yerr
        
        flux_errs = self.flux_errs_flat
        if not xerr is None:
            if isinstance(xerr, float):
                xerr = np.ones_like(flux_errs) * xerr
            if isinstance(xerr, xr.DataArray):
                if xerr.size == 1:
                    xerr = np.ones_like(flux_errs) * xerr.values
                elif xerr.shape == self.flux_errs.shape:
                    xerr = xerr.stack(new=[self.time_coord, *self.spatial_valriables]).values        
                else:
                    xerr.values
            flux_errs = xerr

        if not x is None:
            x_prior = x
        else:
            x_prior = self.flux_flat.values

        if with_prior:
            #print(len(concentration_errs.values))
            print('x covariance')
            print(flux_errs**2)
            print('y_covariance')
            print(concentration_errs**2)
            self.reg = bayesinverse.Regression(
                y = self.concentrations.values, 
                K = self.footprints_flat.values, 
                x_prior = x_prior, 
                x_covariance = flux_errs**2, 
                y_covariance = concentration_errs**2, 
                alpha = alpha
            )
        else:
            self.reg = bayesinverse.Regression(
                y = self.concentrations.values, 
                K = self.footprints_flat.values,
                alpha = alpha
            )
        return self.reg

    def fit(
        self, 
        x: Optional[np.ndarray] = None,
        yerr: Optional[Union[np.ndarray, xr.DataArray, float, list[float]]] = None, 
        xerr: Optional[Union[np.ndarray, list]] = None,
        with_prior: Optional[bool] = True,
        alpha: float = None,
        ) -> xr.DataArray:
        """Uses bayesian inversion to estimate emissions.

        Args:
            yerr (Optional[list], optional): Can be used instead of loaded error of measurement. Defaults to None.
            xerr (Optional[list], optional): Can be used instead of error of the mean of the fluxes. Defaults to None.
            with_prior (Optional[bool]): Wether to use a prior or not. True strongly recommended. Defaults to True.
            alpha (Optional[float]): Value to weigth xerr against yerr. Is used as in l curve calculation. Defaults to 1.

        Returns:
            xr.DataArray: estimated emissions
        """        
        _ = self.get_regression(x, yerr, xerr, with_prior, alpha)
        
        self.fit_result = self.reg.fit()
        self.predictions_flat = xr.DataArray(
            data = self.fit_result[0],
            dims = ["new"],
            coords = dict(new=self.coords)
        )
        self.predictions = self.predictions_flat.unstack("new")
        self.prediction_errs_flat = xr.DataArray(
            data = np.sqrt(np.diag(self.get_posterior_covariance())),
            dims = ["new"],
            coords = dict(new=self.coords)
        )
        self.prediction_errs = self.prediction_errs_flat.unstack("new")
        return self.predictions


    # for calculation errors : 
        
    def get_land_ocean_error(self, factor):
        CO2flux_errs = self.flux_errs.where(self.flux_errs.bioclass < (self.flux_errs.bioclass.max()+1)/2, drop = True)
        CO2err = xr.ones_like(CO2flux_errs)* CO2flux_errs.mean()
        CO2err = CO2err.where(CO2err.bioclass != 0, CO2flux_errs.mean() * factor)
        #print('CO2 err')
        #print(CO2flux_errs.mean())

        COflux_errs = self.flux_errs.where(self.flux_errs.bioclass >= (self.flux_errs.bioclass.max()+1)/2, drop = True)
        COerr = xr.ones_like(COflux_errs)* COflux_errs.mean()
        COerr = COerr.where(COerr.bioclass != (self.flux_errs.bioclass.max()+1)/2, COflux_errs.mean() * factor)
        #print('CO err')
        #print(COflux_errs.mean())
        err = xr.concat([CO2err,COerr], dim = "bioclass")

        self.flux_errs = err
        return err
    
    def get_land_ocean_error_same_scaling(self, factor):
        # da kommt was ganz komsiches raus!!!!!!!!!! 
        CO2flux_errs = self.flux_errs.where(self.flux_errs.bioclass < (self.flux_errs.bioclass.max()+1)/2, drop = True)
        CO2flux_mean = self.flux.where((self.flux.bioclass < (self.flux.bioclass.max()+1)/2)&(self.flux.bioclass >0), drop = True).mean()
        CO2err = xr.ones_like(CO2flux_errs)*CO2flux_mean 
        CO2err = CO2err.where(CO2err.bioclass != 0, CO2flux_mean * factor)
        print('CO2 err')
        print(CO2err.mean())
        print('CO2')
        print(CO2flux_mean)

        COflux_errs = self.flux_errs.where(self.flux_errs.bioclass >= (self.flux_errs.bioclass.max()+1)/2, drop = True)
        COflux_mean = self.flux.where((self.flux.bioclass > (self.flux.bioclass.max()+1)/2), drop = True).mean()
        COerr = xr.ones_like(COflux_errs)*COflux_mean
        COerr = COerr.where(COerr.bioclass != (self.flux_errs.bioclass.max()+1)/2, COflux_mean * factor)
        print('CO err')
        print(COerr.mean())
        print('CO')
        print(COflux_mean)

        #print(COflux_errs.mean())
        err = xr.concat([CO2err,COerr], dim = "bioclass")

        self.flux_errs = err
        print('total_error')
        print(err)
        return err
    
    
    def calc_errors_flat_area_weighted_scaled_to_mean_flux(self, area_bioreg): 
        flux_mean = self.flux
        flat_errors = np.ones((int(len(flux_mean.bioclass.values)/2), len(flux_mean.week.values)))
        area_bioreg[0] = area_bioreg[0]*10000000 # ocean smaller
        final_errorCO = np.ones(flat_errors.shape)
        final_errorCO2 = np.ones(flat_errors.shape)
        for w in range(len(flux_mean.week.values)): 
            area_weighted_errors = flat_errors[:,w]/area_bioreg
            # checken ob richtiger bereich ausgewählt wurde !!!!!!!!!!!!!!!!!!
            scaling_factorCO2 = flux_mean[1:int((len(flux_mean.bioclass.values))/2), w].mean()/area_weighted_errors[1:].mean()
            scaling_factorCO = flux_mean[int((len(flux_mean.bioclass.values))/2)+1:(len(flux_mean.bioclass.values)), w].mean()/area_weighted_errors[1:].mean()
            final_errorCO[:,w] = scaling_factorCO.values*area_weighted_errors
            final_errorCO2[:,w] = scaling_factorCO2.values*area_weighted_errors
            #print('Week: '+str(w))
            #print('Mean error CO:'+str(np.mean(scaling_factorCO.values*area_weighted_errors)))
            #print('Mean error CO2:'+str(np.mean(scaling_factorCO2.values*area_weighted_errors)))
            #print('Mean flux CO: '+str(flux_mean[int((len(flux_mean.bioclass.values))/2):(len(flux_mean.bioclass.values)), w].mean()))
            #print('Mean flux CO2: '+str(flux_mean[1:int((len(flux_mean.bioclass.values))/2), w].mean()))

        final_error = np.concatenate([final_errorCO2, final_errorCO])
        err_scaled = xr.DataArray(data=final_error, coords=dict({ 'bioclass': ('bioclass',flux_mean.bioclass.values),# [0,1,2,3,4,5,6]),
                                                                    'week': ('week',flux_mean.week.values)}))
        self.flux_errs = err_scaled
        return err_scaled
    
    def get_prior_flux_errors(self, non_equal_region_size, area_bioreg):
        # area bioreg: size of one inversion (CO or CO2, not both together)
        # prior errors
        if non_equal_region_size == True: 
            print('per region')
            err = self.calc_errors_flat_area_weighted_scaled_to_mean_flux(area_bioreg)
            print('maximum error value: '+str(err.max()))
        else: 
            print('not per region')
            err = self.get_land_ocean_error(1/100000) # adapt CO2 and CO errors separately check 
            print(err)
            print('maximum error value: '+str(err.max()))
        print('Initlaizing done')
        return err 




    def calculate_averaging_kernel(self, class_num, week_num): 
        ak = self.get_averaging_kernel()

        ak_sum_CO2 = np.zeros((week_num*class_num,class_num))
        ak_sum_CO = np.zeros((week_num*class_num,class_num))   
        for i in range(0,class_num):#ecosystem number 
            list_indices = list(i+class_num*np.arange(0,week_num))
            for idx in list_indices: 
                #print(idx)
                ak_sum_CO2[:,i] += ak[0:class_num*week_num,idx]#ak[:,i]+ak[:,i+7]+ak[:,i+2*7]+ak[:,i+3*7]+ak[:,i+4*7]+ak[:,i+5*7]

            list_indicesCO = list(i+class_num*np.arange(week_num,2*week_num))
            for idx in list_indicesCO: 
                ak_sum_CO[:,i] += ak[class_num*week_num:2*class_num*week_num,idx]#ak[:,i]+ak[:,i+7]+ak[:,i+2*7]+ak[:,i+3*7]+ak[:,i+4*7]+ak[:,i+5*7]
        
        ak_finalCO2 = np.zeros((class_num,class_num))
        ak_finalCO = np.zeros((class_num,class_num))
        for i in range(0,class_num):
            list_indices = list(i+class_num*np.arange(0,week_num))#0,week_num))
            #list_indicesCO = list(i+class_num*np.arange(week_num,2*week_num))

            for idx in list_indices:
                ak_finalCO2[i] += ak_sum_CO2[idx]#(ak_sum[i+7*2]+ak_sum[i+7*3]+ak_sum[i+7*4]+ak_sum[i+7*5])/4
                ak_finalCO[i] += ak_sum_CO[idx]#(ak_sum[i+7*2]+ak_sum[i+7*3]+ak_sum[i+7*4]+ak_sum[i+7*5])/4
            ak_finalCO2[i] = ak_finalCO2[i]/len(list_indices)
            ak_finalCO[i] = ak_finalCO[i]/len(list_indices)
        
        return ak_finalCO2, ak_finalCO, ak_sum_CO2, ak_sum_CO
        















#get_footprint_and_measurement('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/','/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/', 
#                              datetime.datetime(year = 2019, month=12, day= 15), datetime.datetime(year = 2019, month = 12, day = 31))




'''

Inversion = CoupledInversion(
    result_pathCO = "/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions3_CO.pkl" ,
    result_pathCO2 = "/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions.pkl" ,
    flux_pathCO = "/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/Regridded_1x1/", 
    flux_pathCO2 = "/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Flux/CT2022.flux1x1.",
    bioclass_path = "/home/b/b382105/ColumnFLEXPART/resources/Ecosystems_AK_based_split_with_21_and_20_larger.nc",
    month = '2019-12',
    date_min = datetime.datetime(year = 2019, month = 12, day = 1),
    date_max = datetime.datetime(year= 2019, month = 12, day = 31),
    boundary=[110.0, 155.0, -45.0, -10.0], 
       ) 

'''

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


