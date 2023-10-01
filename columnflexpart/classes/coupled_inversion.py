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
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt
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
        area_bioreg,
        week: str, 
        non_equal_region_size = True,  
        date_max = datetime.datetime, 
        time_coarse: Optional[int] = None, 
        coarsen_boundary: str = "pad",
        time_unit: Timeunit = "week",
        boundary: Boundary = None,
        concentration_key: Concentrationkey = "background_inter",
        data_outside_month: bool = False,
        meas_err_CO: float = None,# ppb
        meas_err_CO2: float = None, #ppm
        prior_err_CO_fire: float = 1, 
        prior_err_CO2_fire: float = 1,
       ): 
        

        self.meas_err_CO = meas_err_CO
        self.meas_err_CO2 = meas_err_CO2
        self.prior_err_CO_fire = prior_err_CO_fire
        self.prior_err_CO2_fire = prior_err_CO2_fire
        self.pathCO = result_pathCO
        self.pathCO2 = result_pathCO2
        self.n_eco: Optional[int] = None
        self.bioclass_path = bioclass_path
        self.bioclass_mask = select_boundary(self.get_bioclass_mask(), boundary)
        self.gridded_mask = select_boundary(self.get_gridded_mask(), boundary)
        self.spatial_valriables = ["bioclass"]
        self.final_spatial_variable = ["final_regions"]
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
        self.area_bioreg = area_bioreg
        self.number_of_reg = len(area_bioreg)
        self.non_equal_region_size = non_equal_region_size
        self.week = week

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
        self.footprints,self.footprints_eco, self.concentrations, self.concentration_errs = self.get_footprint_and_measurement(self.concentration_key)
        self.flux_grid, self.flux_eco = self.get_flux() ### brauche ich erstmal nicht, ich bekomme ja Scaling Faktoren raus, also der "prior" ist 1 überall
        self.flux_grid_flat = self.flux_grid.stack(new=[self.time_coord, *self.final_spatial_variable])
        self.flux_eco_flat = self.flux_eco.stack(new=[self.time_coord, *self.final_spatial_variable])
        #self.flux_eco_co2_fire, self.flux_eco_co2_bio, self.flux_eco_co_fire, self.flux_eco_co_bio = self.split_eco_flux()
        self.F = self.get_F_matrix()
        self.Cprior = None
        self.rho_prior = None
        self.K = self.multiply_H_and_F()
        #self.coords = self.K.stack(new=[self.time_coord, *self.spatial_valriables]).new
        #print(self.K.week == self.week)
        self.footprints_flat = self.K[:,self.K.week == self.week,:].squeeze()#.stack(new=[self.time_coord, *self.final_spatial_variable])
        #print(self.footprints_flat)
        self.flux_errs_flat = self.get_prior_covariace_matrix(self.non_equal_region_size, self.area_bioreg)# self.flux_errs.stack(new=[self.time_coord, *self.spatial_valriables])
        #print(self.flux_errs_flat)
        self.coords = self.flux_eco.rename(dict(final_regions = 'new')).new#assign_coords(new = ('new', self.flux_eco.final_regions.values)).new

    def get_gridded_mask(self) -> xr.DataArray:
        mask = xr.load_dataset("/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1")["bioclass"]
        if "Lat" in mask.coords:
            mask = mask.rename(dict(Lat="latitude", Long="longitude"))
        self.n_eco = len(np.unique(mask.values))
        return mask

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
    

    '''
    def select_times_one_week_only(self): 
        #'''#returns: cropped predictionsCO and predicitons CO2 datasets for which measurements will be loaded later'''
    '''
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
    '''
    
    def get_footprints_H_matrix(self,footprint_paths: list[str]): 
        footprints = []
        footprints_eco = []
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
            footprint_eco = self.coarsen_data(footprint, "sum", None)
            footprints_eco.append(footprint_eco)
            footprint = select_boundary(footprint, self.boundary)
            footprint = self.coarsen_data_to_grid(footprint, "sum", None)
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
        #print('footprints')
        #print(total_matrix)
        footprints_eco = xr.concat(footprints_eco, dim = "measurement")/100 * 0.029 
        #print('footprints_eco')
        #print(footprints_eco)
        return total_matrix, footprints_eco

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
            concentrationsCO2.append((result_row["xco2_measurement"] - result_row[concentration_key]))
            concentration_errsCO2.append(result_row["measurement_uncertainty"])
            measurement_id.append(i)
            footprint_paths.append(result_row["directory"])
        
        m = len(measurement_id)

        #Loads Measurements, Measurement Uncertainty, Backgrounds, footprints paths for CO
        for i, result_row in tqdm(predictionsCO_cut.iterrows(), desc="Loading CO measurements", total=predictionsCO_cut.shape[0]):
            concentrationsCO.append((result_row["xco2_measurement"] - result_row[concentration_key]))
            concentration_errsCO.append(result_row["measurement_uncertainty"])
            measurement_id.append(m+i) 

        concentrations = xr.DataArray(
            data = concentrationsCO2+concentrationsCO,# append CO2 and CO
            dims = "measurement",
            coords = dict(measurement = measurement_id) 
        )
        ########### added to see influence of concentration errors ############
        if self.meas_err_CO != None: 
            concentration_errsCO = list(np.ones(len(concentration_errsCO2))*self.meas_err_CO)# ppb
        if self.meas_err_CO2 != None: 
            concentration_errsCO2 = list(np.ones(len(concentration_errsCO2))*self.meas_err_CO2)# 2 ppm

        ####################################################################### 

        concentration_errs  = xr.DataArray(
            data = concentration_errsCO2+concentration_errsCO,# append CO2 and CO
            dims = "measurement",
            coords = dict(measurement = measurement_id) 
        )
        # sollte passen (footprints - H matrix)
        footprints, footprints_eco = self.get_footprints_H_matrix(footprint_paths)
        
        return footprints,footprints_eco, concentrations, concentration_errs
    
    def get_grid_cell_indices_for_final_regions(self): 
        #print(self.bioclass_mask.values[:].flatten())
        #print(set(self.bioclass_mask.values[:].flatten()))
        indices_dict = dict()
        indices_dict['0'] = np.zeros(1)
        for reg in set(self.bioclass_mask.values[:].flatten()):
            if reg >0:
                mask_for_reg = self.bioclass_mask.where(self.bioclass_mask == reg, drop = True)
                gridded_masked_for_reg = self.gridded_mask.where(mask_for_reg == reg, drop = True)
                indices_dict[str(int(reg))] = gridded_masked_for_reg.values.flatten()[~np.isnan(gridded_masked_for_reg.values.flatten())]

        #print(indices_dict)
        return indices_dict
    
    def construct_F_matrix(self, xF1_matrix): 
        '''
        Constructs F matrix (4 blocks with F1 on "diagonal", otherwise zero blocks)
        '''
        # es fehlt:  multiplikation mit 10 -6 etc, multiplication mit flux pro Co fire pro Woche etc 
        # pro final region und week einmla mit flux mutliplizieren
        zero_block = xr.zeros_like(xF1_matrix)
        zero_block_left = zero_block.assign_coords(bioclass = ((xF1_matrix.bioclass+ xF1_matrix.bioclass.values.max() + 1)))
        block_left = xr.concat([xF1_matrix,zero_block_left], dim = "bioclass")

        zero_block_right = zero_block.assign_coords(final_regions = ((xF1_matrix.final_regions+ xF1_matrix.final_regions.values.max()+1)))
        xF2_matrix = xF1_matrix.assign_coords(final_regions = (xF1_matrix.final_regions+ xF1_matrix.final_regions.values.max()+1),
                                               bioclass = (xF1_matrix.bioclass+ xF1_matrix.bioclass.values.max() + 1 ))
        block_right = xr.concat([zero_block_right, xF2_matrix], dim = "bioclass")

        upper_left_block = xr.concat([block_left, block_right], dim = "final_regions")

        double_zero_block = xr.zeros_like(upper_left_block)
        lower_left_block = double_zero_block.assign_coords(bioclass = (double_zero_block.bioclass+ double_zero_block.bioclass.values.max() + 1 ))
        upper_right_block = double_zero_block.assign_coords(final_regions = ((double_zero_block.final_regions+ double_zero_block.final_regions.values.max()+1)))

        lower_right_block = upper_left_block.assign_coords(bioclass = (double_zero_block.bioclass+ double_zero_block.bioclass.values.max() + 1 ),
                                                           final_regions = ((double_zero_block.final_regions+ double_zero_block.final_regions.values.max()+1)))
        
        upper_block = xr.concat([upper_left_block, upper_right_block], dim = 'final_regions')
        lower_block = xr.concat([lower_left_block, lower_right_block], dim = 'final_regions')
        
        F_matrix = xr.concat([upper_block, lower_block], dim='bioclass') 

        flux_grid_with_int = self.flux_grid.rename({'final_regions': 'bioclass'})#.assign_coords(bioclass = ('bioclass',self.flux_grid.final_regions.values.astype(int)))
        #np.set_printoptions(threshold=np.Inf)
        #print(np.array(flux_grid_with_int.bioclass.values[:]))
        #print(np.array(flux_grid_with_int.drop_duplicates(dim = 'bioclass').bioclass.values[:]))
        #print(F_matrix.drop_duplicates(dim = 'bioclass'))
        F_matrix = flux_grid_with_int*F_matrix      

        return F_matrix


    def get_F_matrix(self): 
        indices_dict = self.get_grid_cell_indices_for_final_regions()
        F1_matrix =  np.zeros((int(self.gridded_mask.values.max())+1,(int(self.bioclass_mask.values.max())+1), len(self.flux_eco.week.values)))
        #print(self.flux)
        for i in indices_dict: 
            #print(i)
            #column = np.zeros((int(self.gridded_mask.values.max())+1,1))
            for index in indices_dict[i]: 
                F1_matrix[int(index),int(i), :] = 1/len(indices_dict[i])#*self.flux_grid[int(index),:]
        #print(F1_matrix)
        #plt.imshow(F1_matrix[:,:,:])
        #plt.savefig('/home/b/b382105/test/ColumnFLEXPART/columnflexpart/scripts/F_matrix.png')
        xF1_matrix = xr.DataArray(data = F1_matrix,dims = ["bioclass", "final_regions", "week"], coords = dict(bioclass = (["bioclass"], np.arange(0,int(self.gridded_mask.values.max())+1)),
                                                                                                      final_regions =(["final_regions"], np.arange(0,int(self.bioclass_mask.values.max())+1)), 
                                                                                                      week = (["week"], self.flux_eco.week.values)) )
        
        F_matrix = self.construct_F_matrix(xF1_matrix)
       
        #fig = plt.figure(figsize = (10,10))
        #im = plt.imshow(F_matrix[300:,:], cmap = 'Greys')
        #fig.colorbar(im)
        #plt.savefig('/home/b/b382105/test/ColumnFLEXPART/columnflexpart/scripts/F_matrix_test.png')
        #plt.close()

        return F_matrix
    
    def multiply_H_and_F(self): 
        K = xr.dot(self.footprints,self.F, dims = "bioclass")
        #print(self.flux_grid)
        return K 
    
    def coarsen_data_to_grid(
        self,
        xarr: xr.DataArray,
        coarse_func: Union[Callable, tuple[Callable, Callable, Callable], str, tuple[str, str, str]],
        time_coarse: Optional[int] = None
        ) -> xr.DataArray:
        if not isinstance(coarse_func, (list, tuple)):
            coarse_func = [coarse_func]*3
        time_class = xarr["time"].dt
        if self.isocalendar:
            time_class = time_class.isocalendar()
        xarr = xarr.groupby(getattr(time_class, self.time_coord))
        xarr = self.apply_coarse_func(xarr, coarse_func[0])
        xarr.name = "data"
        combination = xr.merge([xarr, self.gridded_mask]).groupby("bioclass")
        xarr = self.apply_coarse_func(combination, coarse_func[1])["data"]
        
        if not time_coarse is None:
            xarr = xarr.coarsen({self.time_coord: time_coarse}, boundary=self.coarsen_boundary)
            xarr = self.apply_coarse_func(xarr, coarse_func[2])
        return xarr

    def coarsen_and_cut_flux_and_get_err(self, flux): 
        flux = select_boundary(flux, self.boundary)
        flux_mean = self.coarsen_data_to_grid(flux, "mean", self.time_coarse)
        #print('flux_mean')
        #print(flux_mean)
        #plt.figure()
        #img = flux_mean.plot()
        ##bar = plt.colorbar(img)
        #plt.savefig('/home/b/b382105/test/ColumnFLEXPART/columnflexpart/scripts/flux_mean.png')
        #plt.close()
        flux_mean_eco = self.coarsen_data(flux, "mean", self.time_coarse)
        #print('flux_mean_eco')
        #print(flux_mean_eco)
        #plt.figure()
        #img = flux_mean_eco.plot()
        #bar = plt.colorbar(img)
        plt.savefig('/home/b/b382105/test/ColumnFLEXPART/columnflexpart/scripts/flux_mean_eco.png')
        plt.close()
        #flux_err = self.get_flux_err(flux, flux_mean)
        return flux_mean, flux_mean_eco

    def get_flux_CO2(self):
        #CO2 
        flux_files = []
        for date in np.arange(self.min_time.astype("datetime64[D]"), self.stop):
            date_str = str(date).replace("-", "")
            flux_files.append(self.flux_pathCO2.parent / (self.flux_pathCO2.name + f"{date_str}.nc"))
        flux = xr.open_mfdataset(flux_files, drop_variables="time_components").compute()
        #print(flux.bio_flux_opt)

        flux_bio_fossil = flux.bio_flux_opt + flux.ocn_flux_opt + flux.fossil_flux_imp
        flux_fire = flux.fire_flux_imp
        
        flux_fire_mean, flux_fire_eco = self.coarsen_and_cut_flux_and_get_err(flux_fire)
        fire_flux_to_plot = flux_fire_mean.where(flux_fire_mean.week == self.week, drop = True)
        spatial_flux = self.map_on_grid(fire_flux_to_plot*12*10**6)
        spatial_flux.plot(x = 'longitude', y = 'latitude')
        plt.savefig('/home/b/b382105/test/ColumnFLEXPART/columnflexpart/scripts/fire_flux_mean.png')
        flux_bio_fossil_mean, flux_bio_fossil_eco = self.coarsen_and_cut_flux_and_get_err(flux_bio_fossil)

        flux_bio_fossil_mean = flux_bio_fossil_mean.assign_coords(bioclass = (flux_fire_mean.bioclass+ flux_fire_mean.bioclass.values.max() + 1 ).astype(int)).rename(dict(bioclass = 'final_regions'))
        flux_mean = xr.concat([flux_fire_mean.rename(dict(bioclass = 'final_regions')), flux_bio_fossil_mean], dim = 'final_regions')

        flux_bio_fossil_eco = flux_bio_fossil_eco.assign_coords(bioclass = (flux_fire_eco.bioclass+ flux_fire_eco.bioclass.values.max() + 1 ).astype(int)).rename(dict(bioclass = 'final_regions'))
        flux_mean_eco = xr.concat([flux_fire_eco.rename(dict(bioclass = 'final_regions')), flux_bio_fossil_eco], dim = 'final_regions')
       
        flux_mean = flux_mean * 10**6
        flux_mean_eco = flux_mean_eco * 10**6
        #flux_bio_fossil_err = flux_bio_fossil_err.assign_coords(bioclass = (flux_fire_err.bioclass+ flux_fire_err.bioclass.values.max() + 1 )).rename(dict(bioclass = 'final_regions'))
        #flux_err = xr.concat([flux_fire_err.rename(dict(bioclass = 'final_regions')), flux_bio_fossil_err], dim = 'final_regions')
        #print(flux_mean)
        return flux_mean, flux_mean_eco#, flux_err

    def get_flux_CO(self): # NOCH NICHT ANGEPASST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #CO: 
        total_flux = xr.DataArray()
        first_flux = True
        for year in range(2019,2020): # HARDCODED UNTIL NOW !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
            #total_flux_yr = (cams_flux_ant['sum'] )/0.02801 # from kg/m^2/s to molCO/m^2/s 
            cams_flux_bio_yr = cams_flux_bio/0.02801
            if first_flux:
                    first_flux = False
                    total_flux = total_flux_yr
                    bio_flux = cams_flux_bio_yr
            else:
                total_flux = xr.concat([total_flux, total_flux_yr], dim = 'time')
                bio_flux =  xr.concat([bio_flux, cams_flux_bio_yr], dim = 'time')

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

        cams_flux_bio = bio_flux
        '''
        ########################### added to test bio flux as prior
        cams_flux_bio = cams_flux_bio.where(cams_flux_bio.MonthDate >= pd.to_datetime(self.start), drop = True )
        cams_flux_bio= cams_flux_bio.where(cams_flux_bio.MonthDate <= pd.to_datetime(self.stop), drop = True )

        #cams_flux = cams_flux[:, :, 1:]
        bio_flux = select_boundary(bio_flux, self.boundary)

        # create new flux dataset with value assigned to every day in range date start and date stop 
        dates = [pd.to_datetime(self.start)]
        date = dates[0]
        while date < pd.to_datetime(self.stop):
            date += datetime.timedelta(days = 1)
            dates.append(pd.to_datetime(date))
        bio_fluxes = xr.DataArray(data = np.zeros((len(dates), len(bio_flux.latitude.values), len(flux.longitude.values))), dims = ['time', 'latitude', 'longitude'])
        count = 0
        for i,dt in enumerate(dates): 
            for m in range(len(bio_flux.MonthDate.values)): # of moths 
                if dt.month == pd.to_datetime(bio_flux.MonthDate[m].values).month: 
                #    if bol == True: 
                    bio_fluxes[i,:,:] =  bio_flux[m,:,:]#/0.02801 
        
        bio_fluxes = bio_fluxes.assign_coords({'time': ('time', dates), 'latitude': ('latitude', bio_flux.latitude.values), 
                                               'longitude': ('longitude', bio_flux.longitude.values)})


        '''
        ############################################################

        flux_mean_bio_ant, flux_bio_ant_eco = self.coarsen_and_cut_flux_and_get_err(fluxes)
        #print(flux_mean_bio_ant)
        #flux_mean_bio, flux_mean_bio_eco = self.coarsen_and_cut_flux_and_get_err(bio_fluxes)

        ######## Flat prior for CO fire: ############################
        #flux_fire = xr.ones_like(flux_mean_bio_ant).rename(dict(bioclass = 'final_regions'))*flux_mean_bio_ant.mean()*0.5
        #flux_fire[0] = flux_fire[0]*10**-7
        #flux_fire_eco = xr.ones_like(flux_bio_ant_eco).rename(dict(bioclass = 'final_regions'))*flux_mean_bio_ant.mean()*0.5
        #flux_fire_eco[0] = flux_fire_eco[0]*10**-7
        ###########################################################
        #Co prior like CO2:  in other function


        #flux_fire_eco = flux_fire_eco.assign_coords(bioclass = (flux_fire_eco.bioclass).astype(int)).rename(dict(bioclass = 'final_regions'))
        #flux_fire = flux_fire.assign_coords(bioclass = (flux_fire.bioclass).astype(int)).rename(dict(bioclass = 'final_regions'))
        #print(flux_fire_eco)
        #print(flux_mean_bio_ant.mean())

        flux_mean_bio_fossil = flux_mean_bio_ant.assign_coords(bioclass = (flux_mean_bio_ant.bioclass+ flux_mean_bio_ant.bioclass.values.max() + 1 ).astype(int)).rename(dict(bioclass = 'final_regions'))
        flux_bio_fossil_eco = flux_bio_ant_eco.assign_coords(bioclass = (flux_bio_ant_eco.bioclass+ flux_bio_ant_eco.bioclass.values.max() + 1 ).astype(int)).rename(dict(bioclass = 'final_regions'))
      
        #flux_err_bio_ant2 = flux_err_bio_ant.assign_coords(bioclass = (flux_mean_bio_ant.bioclass+ flux_mean_bio_ant.bioclass.values.max() + 1 )).rename(dict(bioclass = 'final_regions'))
        #was used for flat prior 
        #flux_mean = xr.concat([flux_fire, 
        #                       flux_mean_bio_fossil], dim = 'final_regions')
        #flux_mean = flux_mean *10**9
        

        #flux_mean_eco = xr.concat([flux_fire_eco, 
        #                       flux_bio_fossil_eco], dim = 'final_regions')
        #flux_err = xr.concat([flux_err_bio_ant.rename(dict(bioclass = 'final_regions')), 
        #                      flux_err_bio_ant2], dim = 'final_regions')
        #flux_mean_eco = flux_mean_eco*10**9
        
        #print('CO flux')
        #print('CO flux mean')
        #print(flux_mean)

        # Error of mean calculation
        return flux_mean_bio_fossil*10**9, flux_bio_fossil_eco*10**9#flux mean , flux_mean_eco

    def get_CO_fire_flux_like_CO2(self, flux_meanCO2_grid, flux_meanCO2_eco):
        fire_grid_CO2 = flux_meanCO2_grid.where(flux_meanCO2_grid.final_regions <= int(flux_meanCO2_grid.final_regions.values.max()/2), drop = True)
        #print('fire grid CO2')
        #print(fire_grid_CO2)
        fire_eco_CO2 = flux_meanCO2_eco.where(flux_meanCO2_eco.final_regions <= int(flux_meanCO2_eco.final_regions.values.max()/2), drop = True)
        factor_scaling_CO2_to_CO = (14.4*28.01/44.01)**-1 *10**3 # from Van de Velde Paper (14.4gCo2/gCO -> 14.4 *44/28 molCO2/molCO)and scaling ppm to ppb
        return fire_grid_CO2*factor_scaling_CO2_to_CO,  fire_eco_CO2*factor_scaling_CO2_to_CO


    def get_flux(self): # to be wrapped together # flux mean has coords bioclass and week 
        
        flux_meanCO2_grid, flux_meanCO2_eco = self.get_flux_CO2()
        ################# uncomment for flat prior ##############
        #flux_meanCO_grid, flux_meanCO_eco = self.get_flux_CO() # err checken!!!!!!!!!!!!!! # ADAPTED _ NEEDS TO BNE CO!!!!!!!!!!!!!!!!!!!!!!!!
        

        ################## comment for flat prior ###############
        COflux_mean_bio_fossil_grid, COflux_bio_fossil_eco = self.get_flux_CO()
        COfire_flux_grid, COfire_flux_eco = self.get_CO_fire_flux_like_CO2(flux_meanCO2_grid, flux_meanCO2_eco)
        flux_meanCO_grid  = xr.concat([COfire_flux_grid, 
                               COflux_mean_bio_fossil_grid], dim = 'final_regions')
        #print('flux mean CO grid')
        #print(flux_meanCO_grid)
        #print('flux mean CO2 grid')
        #print(flux_meanCO2_grid)
        flux_meanCO_eco  = xr.concat([COfire_flux_eco, 
                               COflux_bio_fossil_eco], dim = 'final_regions')

        #np.set_printoptions(threshold=np.Inf)
        #print(np.array(flux_meanCO2_grid.final_regions.values[:]))
        #print(np.array(flux_meanCO_grid.final_regions.values[:]))
        
        #adapt coordinates and concatenate fluxes
        flux_meanCO_grid = flux_meanCO_grid.assign_coords(final_regions = ((flux_meanCO_grid.final_regions+ flux_meanCO_grid.final_regions.values.max() + 1).astype(int)))
        flux_mean_grid = xr.concat([flux_meanCO2_grid, flux_meanCO_grid], dim = "final_regions")
        #flux_mean_grid = flux_mean_grid.assign_coords(final_regions = ('final_regions', flux_mean_grid.final_regions.values.astype(int)))

        flux_meanCO_eco = flux_meanCO_eco.assign_coords(final_regions = ((flux_meanCO_eco.final_regions+ flux_meanCO_eco.final_regions.values.max() + 1).astype(int)))
        flux_mean_eco = xr.concat([flux_meanCO2_eco, flux_meanCO_eco], dim = "final_regions")
        ################################################################################################


        #flux_errCO = flux_errCO.assign_coords(final_regions = ((flux_errCO.final_regions+ flux_errCO.final_regions.values.max() + 1)))
        #flux_err = xr.concat([flux_errCO2, flux_errCO], dim = "final_regions")


        #print('flux_mean_grid')
        #print(flux_mean_grid)
        #print('flux_mean_eco')
        #print(flux_mean_eco)


        return flux_mean_grid, flux_mean_eco#, flux_err
    

   
    # brauche ich das??????????????????????????????????????????
    def get_prior_scaling_factors(self):
        return xr.ones_like(self.flux_eco)
    

    def calc_errors_flat_area_weighted_scaled_to_mean_flux(self, area_bioreg): 
        flux_mean = self.flux
        flat_errors = np.ones((int((self.bioclass_mask.max()+1)), len(flux_mean.week.values)))
        area_bioreg[0] = area_bioreg[0]*10000000 # ocean smaller
        final_errorCO = np.ones(flat_errors.shape)
        final_errorCO2 = np.ones(flat_errors.shape)
        for w in range(len(flux_mean.week.values)): 
            area_weighted_errors = flat_errors[:,w]/area_bioreg
            # checken ob richtiger bereich ausgewählt wurde !!!!!!!!!!!!!!!!!!
            scaling_factorCO2 = flux_mean[1:int((len(flux_mean.final_regions.values))/4), w].mean()/area_weighted_errors[1:].mean()
            scaling_factorCO = flux_mean[int((len(flux_mean.final_regions.values))/2)+1:int((len(flux_mean.final_regions.values))*3/4), w].mean()/area_weighted_errors[1:].mean()
            final_errorCO[:,w] = scaling_factorCO.values*area_weighted_errors # only for fire
            final_errorCO2[:,w] = scaling_factorCO2.values*area_weighted_errors # only for fire
            #print('Week: '+str(w))
            #print('Mean error CO:'+str(np.mean(scaling_factorCO.values*area_weighted_errors)))
            #print('Mean error CO2:'+str(np.mean(scaling_factorCO2.values*area_weighted_errors)))
            #print('Mean flux CO: '+str(flux_mean[int((len(flux_mean.bioclass.values))/2):(len(flux_mean.bioclass.values)), w].mean()))
            #print('Mean flux CO2: '+str(flux_mean[1:int((len(flux_mean.bioclass.values))/2), w].mean()))

        #final_error = np.concatenate([final_errorCO2, final_errorCO])
        errCO_scaled = xr.DataArray(data=final_errorCO, coords=dict({ 'bioclass': ('bioclass',np.arange(0,len(area_bioreg))),# [0,1,2,3,4,5,6]),
                                                                    'week': ('week',flux_mean.week.values)}))
        errCO2_scaled = xr.DataArray(data=final_errorCO2, coords=dict({ 'bioclass': ('bioclass',np.arange(0,len(area_bioreg))),# [0,1,2,3,4,5,6]),
                                                                    'week': ('week',flux_mean.week.values)}))
        
        #self.flux_errs = err_scaled
        return errCO2_scaled, errCO_scaled

    def get_Cprior_matrix(self): 
        # ist momentatn ja gleich pro Woche, daher nur für final regions, ohne Wochenkoordinate konsturiert
        #print(self.flux.final_regions.shape)
        n_reg = int(self.bioclass_mask.values.max()) 
        #print(n_reg)
        #print(Cprior)
        #print(Cprior.shape)
        Cprior = np.zeros((self.flux_eco.final_regions.shape[0], self.flux_eco.final_regions.shape[0]))
        # set 1 at "block diagonal" - fire CO2 and fire Co2 are max correlated
        Cprior[:n_reg+1, :n_reg+1] = 1
        Cprior[n_reg+1:2*(n_reg+1),n_reg+1 :2*(n_reg+1)] = 1
        Cprior[2*(n_reg+1):3*(n_reg+1), 2*(n_reg+1):3*(n_reg+1)] = 1
        Cprior[3*(n_reg+1):4*(n_reg+1),3*(n_reg+1) :4*(n_reg+1)] = 1

        # correlate the fires (Co and Co2) with 0.7 (ant and bi onot correlated)
        Cprior[:n_reg+1,2*(n_reg+1) :3*(n_reg+1)] = 0.7
        Cprior[2*(n_reg+1) :3*(n_reg+1), :n_reg+1] = 0.7
        #print(Cprior[np.where(Cprior!=0)])
        # DImensions cannot be called the same, check with mutliplication that names of coordinates etc fit !!!!!!!!!!!!!!!!!
        '''
        if len(self.flux_eco.week.values) == 6: 
            C = np.block([[Cprior, Cprior, Cprior, Cprior, Cprior,Cprior], 
                     [Cprior, Cprior, Cprior, Cprior, Cprior,Cprior],
                     [Cprior, Cprior, Cprior, Cprior, Cprior,Cprior],
                     [Cprior, Cprior, Cprior, Cprior, Cprior,Cprior],
                     [Cprior, Cprior, Cprior, Cprior, Cprior,Cprior],
                     [Cprior, Cprior, Cprior, Cprior, Cprior,Cprior]])
        elif len(self.flux_eco.week.values) == 5: 
            C = np.block([[Cprior, Cprior, Cprior, Cprior, Cprior],
                         [Cprior, Cprior, Cprior, Cprior, Cprior],
                         [Cprior, Cprior, Cprior, Cprior, Cprior],
                         [Cprior, Cprior, Cprior, Cprior, Cprior],
                         [Cprior, Cprior, Cprior, Cprior, Cprior]])
        elif len(self.flux_eco.week.values) ==4 : 
            C = np.block([[Cprior, Cprior, Cprior, Cprior],
                         [Cprior, Cprior, Cprior, Cprior],
                         [Cprior, Cprior, Cprior, Cprior],
                         [Cprior, Cprior, Cprior, Cprior]])
        else: 
            raise ValueError('Cprior matrix cannot be constructed because number of weeks is not 4,5 or 6')
        '''
        #plt.figure()
        #plt.imshow(Cprior)#Cprior.plot(x = 'final_regions', y = 'final_regions')
        #plt.savefig('/home/b/b382105/test/ColumnFLEXPART/columnflexpart/scripts/C_prior.png')
        #plt.close()
        #print(Cprior)
        #print(Cprior[np.where(Cprior!=0)])
        #print(C.shape)
        #print(self.flux_eco_flat.new.values)
        #Cprior = xr.DataArray(data = C ,dims = ["final_regions", "final_regions2"],
        #                       coords = dict(final_regions =(["final_regions"], self.flux_eco_flat.new.values), 
        #                                     final_regions2 =(["final_regions2"], self.flux_eco_flat.new.values)))
        
        Cprior = xr.DataArray(data = Cprior ,dims = ["final_regions", "final_regions2"],
                            coords = dict(final_regions =(["final_regions"], self.footprints_flat.final_regions.values), 
                                        final_regions2 =(["final_regions2"],  self.footprints_flat.final_regions.values)))
        #print(Cprior.isnull())
        #plt.imshow(Cprior)
        #plt.savefig('/home/b/b382105/test/ColumnFLEXPART/columnflexpart/scripts/Cprior.png')
        self.Cprior = Cprior
        return Cprior
    

    def get_rho_prior_matrix(self, area_weighted_errorsCO2, area_weighted_errorsCO): # EINHEITEN?!?!?!#gerht gerade nicht weil input 2d (verschieden pro Woche)
        # area weighted errors are only for the real regions number e.g 30 not the splitted, concatenated ones
        #print(area_weighted_errorsCO.values.shape[0])
        bio_fossil_CO = np.ones(area_weighted_errorsCO.values.shape[0])*area_weighted_errorsCO.values[0] # adapt according to errors of fire block!!!!!!!!!!!!!
        bio_fossil_CO[0] = bio_fossil_CO[0]/10000000 
        bio_fossil_block_CO = np.diag(bio_fossil_CO)

        bio_fossil_CO2 = np.ones_like(area_weighted_errorsCO2)*area_weighted_errorsCO2.values[0] # adapt according to errors of fire block!!!!!!!!!!!!!
        bio_fossil_CO2[0] = bio_fossil_CO2[0]/10000000 
        bio_fossil_block_CO2 = np.diag(bio_fossil_CO2)

        fire_blockCO = np.diag(area_weighted_errorsCO.values) 
        fire_blockCO2 = np.diag(area_weighted_errorsCO2.values) 
        zero_block = np.zeros_like(fire_blockCO)
   
        off_diag_fire = np.sqrt(area_weighted_errorsCO* area_weighted_errorsCO2)
        off_diag_fire_block = np.diag(off_diag_fire)

        total_left_block = np.concatenate((fire_blockCO2, zero_block, off_diag_fire_block, zero_block), axis = 0)
        total_middle_left_block = np.concatenate((zero_block, bio_fossil_block_CO2, zero_block, zero_block), axis = 0)
        total_middle_ríght_block = np.concatenate((off_diag_fire_block, zero_block, fire_blockCO, zero_block), axis = 0)
        total_right_block = np.concatenate((zero_block, zero_block, zero_block, bio_fossil_block_CO), axis = 0)

        total_data = np.concatenate((total_left_block, total_middle_left_block, total_middle_ríght_block, total_right_block), axis = 1)
        rho_prior = xr.DataArray(data = total_data, dims = ["final_regions", "final_regions2"],
                               coords = dict(final_regions =(["final_regions"], self.flux_eco_flat.new.values), final_regions2 =(["final_regions2"], self.flux_eco_flat.new.values)))
   
        return rho_prior

    def get_prior_flux_errors(self, non_equal_region_size, area_bioreg):
        # area bioreg: size of one inversion (CO or CO2, not both together)
        # prior errors
        if non_equal_region_size == True:
            errCO2, errCO = self.calc_errors_flat_area_weighted_scaled_to_mean_flux(area_bioreg)
            errCO2 = errCO2.stack(new=[self.time_coord, *self.spatial_valriables])
            errCO = errCO.stack(new=[self.time_coord, *self.spatial_valriables])
        else: 
            #print('not per region') # FÜR GEGRIDDEDE AUCH NOCH FLATTEN?!
            errCO2, errCO = self.get_land_ocean_error(1/100000) # adapt CO2 and CO errors separately check 

        return errCO2, errCO 
    
    def get_non_area_weighted_rho_matrix(self,  CO2fire_error: float, COfire_error: float):
        '''
        return rho matrix with the given values on the diagonal for CO and CO2 fire. 1 corresponds to a 100% error.
        '''
        len_diag_flattened = int(self.bioclass_mask.values.max()+1)

        fireCO_diag = np.ones(len_diag_flattened) * COfire_error
        fireCO_diag[0] = fireCO_diag[0]/10000000 
        fireCO = np.diag(fireCO_diag**2)
    
        fireCO2_diag = np.ones(len_diag_flattened) * CO2fire_error
        fireCO2_diag[0] = fireCO2_diag[0]/10000000 
        fireCO2 = np.diag(fireCO2_diag**2)

        off_diag_fire = fireCO2_diag*fireCO_diag
        off_diag_fire_block = np.diag(off_diag_fire)

        bio_diag =  np.ones(len_diag_flattened)/10000000 
        bio = np.diag(bio_diag**2)

        zero_block = np.zeros_like(bio)

        total_left_block = np.concatenate((fireCO2, zero_block, off_diag_fire_block, zero_block), axis = 0)
        total_middle_left_block = np.concatenate((zero_block, bio, zero_block, zero_block), axis = 0)
        total_middle_ríght_block = np.concatenate((off_diag_fire_block, zero_block, fireCO, zero_block), axis = 0)
        total_right_block = np.concatenate((zero_block, zero_block, zero_block, bio), axis = 0)

        data_weekly = np.concatenate((total_left_block, total_middle_left_block, total_middle_ríght_block, total_right_block), axis = 1)
        
        rho_prior = xr.DataArray(data = data_weekly, dims = ["final_regions", "final_regions2"],
                               coords = dict(final_regions =(["final_regions"], self.footprints_flat.final_regions.values),
                                              final_regions2 =(["final_regions2"], self.footprints_flat.final_regions.values)))
        
        self.rho_prior = rho_prior
        return rho_prior

    def get_prior_covariace_matrix(self, non_equal_region_size, area_bioreg): 

        rho = self.get_non_area_weighted_rho_matrix(self.prior_err_CO2_fire,self.prior_err_CO_fire)
        Cprior = self.get_Cprior_matrix()
        prior_cov = rho * Cprior
        return prior_cov # hat Dimension flattened week, final_regions x flattened week, final regions, 
        #  die heißen final_regions und final_regions2, in FP etc heißt die Dimension 'new'
    
###################### muss noch angepasst werden!!!!!!!!!!!!!!!!!!!!!!!!!
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
            x_prior = np.ones(len(self.area_bioreg)*4)
        #print(x_prior)
        #print(self.footprints_flat)
        #print(flux_errs)
        if with_prior:
            #print(len(concentration_errs.values))
            self.reg = bayesinverse.Regression(
                y = self.concentrations.values, 
                K = self.footprints_flat.values, 
                x_prior = x_prior, 
                x_covariance = flux_errs,#**2, 
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
        """Uses bayesian inversion to estiamte emissions.

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
        #self.predictions = self.predictions_flat.unstack("new")
        self.prediction_errs_flat = xr.DataArray(
            data = np.sqrt(np.diag(self.get_posterior_covariance())),
            dims = ["new"],
            coords = dict(new=self.coords)
        )# is std deviation 
        #self.prediction_errs = self.prediction_errs_flat.unstack("new")
        return #self.predictions
    
    def map_on_gridded_grid(self, xarr: xr.DataArray) -> xr.DataArray:#[self.time_coord]
        mapped_xarr = xr.DataArray(
        data = np.zeros(
            (
                # len(xarr[self.time_coord]),
                len(self.gridded_mask.longitude),
                len(self.gridded_mask.latitude)
            )
        ),
        coords = {
            #self.time_coord: xarr[self.time_coord], 
            "longitude": self.gridded_mask.longitude, 
            "latitude": self.gridded_mask.latitude
        }
        )
        for bioclass in xarr.bioclass:
            mapped_xarr = mapped_xarr + (self.gridded_mask == bioclass) * xarr.where(xarr.bioclass == bioclass, drop=True)
            mapped_xarr = mapped_xarr.squeeze(drop=True)
        return mapped_xarr

