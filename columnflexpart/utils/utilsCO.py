from pathlib import Path
from typing import Optional, Union, Any
import numpy as np
import pandas as pd
import yaml
import os
import xarray as xr
import geopandas as gpd
from datetime import date, datetime, timedelta
import dask
import bayesinverse

################
# GENERAL THINGS
################

def pressure_factor(
    h,
    Tb = 288.15,
    hb = 0,
    R = 8.3144598,
    g = 9.80665,
    M = 0.0289644,
    ):
    """Calculate factor of barrometric height formula as described here: https://en.wikipedia.org/wiki/Barometric_formula

    Args:
        h (fleat): height for factor calculation [m]
        Tb (float, optional): reference temperature [K]. Defaults to 288.15.
        hb (float, optional): height of reference [m]. Defaults to 0.
        R (float, optional): universal gas constant [J/(mol*K)]. Defaults to 8.3144598.
        g (float, optional): gravitational acceleration [m/s^2]. Defaults to 9.80665.
        M (float, optional): molar mass of Earth's air [kg/mol]. Defaults to 0.0289644.

    Returns:
        float: fraction of pressure at height h compared to height hb
    """    
    factor = np.exp(-g * M * (h - hb)/(R * Tb))
    return factor

def inv_pressure_factor(
    factor: Union[float, np.ndarray],
    Tb: float = 288.15,
    hb: float = 0,
    R: float = 8.3144598,
    g: float = 9.80665,
    M: float = 0.0289644,
    ) -> Union[float, np.ndarray]:
    """Calculate height form inverse of barrometric height formula as described here: https://en.wikipedia.org/wiki/Barometric_formula

    Args:
        factor (fleat): height for factor calculation [m]
        Tb (float, optional): reference temperature [K]. Defaults to 288.15.
        hb (float, optional): height of reference [m]. Defaults to 0.
        R (float, optional): universal gas constant [J/(mol*K)]. Defaults to 8.3144598.
        g (float, optional): gravitational acceleration [m/s^2]. Defaults to 9.80665.
        M (float, optional): molar mass of Earth's air [kg/mol]. Defaults to 0.0289644.

    Returns:
        float: height h at fraction of pressure relative to hb
    """    
    h = (R * Tb) / (-g * M) *  np.log(factor) + hb
    return h

def config_total_parts(config_path: str, output_path: str, total_parts: str):
    """Create a config file for prepare_release.py with different total particle number.

    Args:
        config_path (str): Path of original config.
        output_path (str): Path for manipulated config.
        total_parts (str): Number of particles to insert.
    """    
    with open(config_path) as f:
        config = dict(yaml.full_load(f))
    
    config["n_total"] = total_parts

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved to {output_path}.")

def yyyymmdd_to_datetime64(date_string: str) -> np.datetime64:
    """Convert string of form yyyymmdd to np.datetime64.

    Args:
        date_string (str): String of date to convert 

    Returns:
        np.datetime64: Converted date
    """    
    date = np.datetime64(f"{date_string[:4]}-{date_string[4:6]}-{date_string[6:]}")
    return date

def yyyymmdd_to_datetime(date_string: str) -> datetime:
    """Convert string of form yyyymmdd to datetime.datetime.

    Args:
        date_string (str): String of date to convert 

    Returns:
        datetimt.datetime: Converted date
    """    
    date = yyyymmdd_to_datetime64(date_string)
    date = date.astype(datetime)
    date = datetime.combine(date, datetime.min.time())
    return date

def hhmmss_to_timedelta64(time_string: str) -> np.timedelta64:
    """Convert string of form hhmmss to np.timedelta64.

    Args:
        time_string (str): String of time to convert 

    Returns:
        np.timedelta64: Converted time
    """    
    time = (np.timedelta64(time_string[:2], "h")
        + np.timedelta64(time_string[2:4], "m") 
        + np.timedelta64(time_string[4:], "s"))
    return time

def hhmmss_to_timedelta(time_string: str) -> timedelta:
    """Convert string of form hhmmss to datetime.timedelta.

    Args:
        time_string (str): String of time to convert 

    Returns:
        datetime.timedelta: Converted time
    """    
    time = hhmmss_to_timedelta64(time_string)
    time = time.astype(timedelta)
    return time

def datetime64_to_yyyymmdd_and_hhmmss(time: np.datetime64) -> tuple[str, str]:
    """Convert np.datetime64 to strings of type yyyymmdd and hhmmss 

    Args:
        time (np.datetime64): time to convert

    Returns:
        tuple[str, str]: String for date and time
    """    
    string = str(time)
    string = string.replace("-", "").replace(":", "")
    date, time = string.split("T")
    return date, time

def datetime_to_yyyymmdd_and_hhmmss(time: datetime) -> tuple[str, str]:
    """Convert datetime.datetime to strings of type yyyymmdd and hhmmss 

    Args:
        time (datetime.datetime): time to convert

    Returns:
        tuple[str, str]: String for date and time
    """    
    time = np.datetime64(time).astype("datetime64[s]")
    date, time = datetime64_to_yyyymmdd_and_hhmmss(time)
    return date, time

def in_dir(path: str, string: str) -> bool:
    """Checks if a sting is part of the name of a file in a directory.

    Args:
        path (str): Path of directory
        string (str): String to search for.

    Returns:
        bool: Whether string was found or not
    """    
    for file in os.listdir(path):
        if string in file:
            return True
    return False

def to_tuple(convert_arg: list[str], convert_ind: list[int]) -> tuple:
    """Decorator for functions that are cached but can get np.ndarrays or lists as input

    Args:
        convert_arg (list[str]): keywords for arguments to transform    
        convert_ind (list[int]): positions of arguments to transform
    
    Returns:
        tuple: converted arguments
    """    
    def to_tuple_inner(function):
        def new_function(*args, **kwargs):
            args = list(args)
            for i, arg in enumerate(args):
                if i in convert_ind:
                    if not (isinstance(arg, int) or arg is None):
                        args[i] = tuple(arg)
            for i, (key, value) in enumerate(kwargs.items()):
                if key in convert_arg:
                    if not (isinstance(value, int) or value is None):
                        kwargs[key] = tuple(value)
            args = tuple(args)
            return function(*args, **kwargs)

        return new_function

    return to_tuple_inner


def val_to_list(types: list[type], val: Any, expand_to: Optional[int]=None):
    """Tranforms values to list if needed.

    Args:
        types (list[type])): Classes to be converted convered to list
        val (Any): Vlaue to convert
        expand_to (int, optional): Expands lenght of list. Defaults to None.

    Returns:
        list: Value put in list.
    """    
    if not isinstance(type(types), list):
        types = [types]
    if type(val) in types:
        val = [val]
    val = list(val)
    if expand_to is not None and len(val) == 1:
        val = val * expand_to
    return val

############################
# LESS GENERAL
############################


def detrend_hawaii(dataframe: pd.DataFrame, variable: str, time_variable:str = "time") -> pd.DataFrame:
    """Subtracts absolute measured concentrations from data.

    Args:
        dataframe (pd.DataFrame): Dataframe that holds variable to be detrended
        variable (str): Key of data to detrend
        time_variable (str, optional): Key of variable that holds times of the rows. Defaults to "time".

    Returns:
        pd.DataFrame: Dataframe with added variable_detrend column.
    """    

    noaa_data = pd.read_csv(Path(__file__).parent / "data/co2_annmean_gl.csv", delimiter=",", header=0).sort_values("year")
    noaa_data.insert(1, "year_corrected", noaa_data.year.to_numpy(dtype="str").astype("datetime64") + np.timedelta64(6, "M"))

    dataframe.sort_values(time_variable)
    reference = noaa_data.year_corrected.values.min().astype("datetime64")
    time_data = (dataframe[time_variable].values.astype("datetime64") - reference).astype(float)
    time_noaa = (noaa_data.year_corrected.values.astype("datetime64") - reference).astype(float)
    correction = np.interp(x = time_data, xp = time_noaa, fp = noaa_data["mean"].values)

    dataframe.insert(1, f"{variable}_detrend", dataframe[variable] - correction)
    return dataframe


#############################
# GEOGRAPHICAL APPLICATIONS
#############################
def select_boundary(
    data: Union[xr.DataArray, xr.Dataset], 
    boundary: Optional[tuple[float, float, float, float]]
    ) -> Union[xr.DataArray, xr.Dataset]:
    """Selects data within a boundary in lungitude and latitude

    Args:
        data (Union[xr.DataArray, xr.Dataset]): xarray object that hold data
        boundary (Optional[tuple[float, float, float, float]]): Boundary to cut out in [Left, right, lower, upper]

    Returns:
        Union[xr.DataArray, xr.Dataset]: Data within selected boundaries.
    """    
    if not boundary is None:
        data = data.isel(
            longitude = (data.longitude >= boundary[0]) * (data.longitude <= boundary[1]),
            latitude = (data.latitude >= boundary[2]) * (data.latitude <= boundary[3]) 
        )
    return data


def select_extent(xarr, lon1, lon2, lat1, lat2):
    """Select extent of xarray.DataArray with geological data

    Args:
        xarr (xarray.DataArray): ...returns
        lon1 (float): left
        lon2 (float): right
        lat1 (float): lower
        lat2 (float): upper

    Returns:
        xarray.DataArray: cut xarray
    """
    xarr = xarr.isel(longitude=(lon1 <= xarr.longitude) * (xarr.longitude <= lon2))
    xarr = xarr.isel(latitude=(lat1 <= xarr.latitude) * (xarr.latitude <= lat2))
    return xarr

def xr_to_gdf(xarr, *data_variables, crs="EPSG:4326"):
    """Convert xarray.DataArray to GeoDataFrame

    Args:
        xarr (DataArray): To be converted
        *data_variables (str): data variables to be transferred to GeoDataFrame
        crs (str, optional): Coordinate reference system for GeoDataFrame. Defaults to "EPSG:4326".

    Returns:
        GeoDataFrame: Output
    """
    df = xarr.to_dataframe().reset_index()
    gdf = gpd.GeoDataFrame(
        df[data_variables[0]],
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs=crs,
    )
    for i, dv in enumerate(data_variables):
        if i == 0:
            continue
        gdf[dv] = df[dv]
    return gdf

def add_country_names(gdf):
    """Add country names to GeaDataFrame

    Args:
        gdf (GeoDataFrame): to add country names to

    Returns:
        GeoDataFrame: with added names
    """
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = gpd.GeoDataFrame(world[["name"]], geometry=world.geometry)
    gdf = gpd.sjoin(gdf, world, how="left").drop(columns=["index_right"])
    return gdf


def country_intersections(gdf, country, crs="EPSG:4326"):
    """Cut GeoDataFrame w.r.t. one country into country, , not country (rest), other countries and ocean

    Args:
        gdf (GeoDataFrame): GeoDataFrame to split up
        country (str): Country to single out
        crs (str, optional): Coordinate reference system. Defaults to "EPSG:4326".

    Returns:
        dict: Dict of GeoDataFrames
    """
    assert (
        country in gdf.name.values
    ), f"No country found with name {country} in GeoDataFrame"
    ov_count = gdf[gdf.name == country]
    ov_rest = gdf[gdf.name != country]
    ov_other = gdf[(gdf.name != country) & (gdf.name.notnull())]
    ov_sea = gdf[gdf.name.isnull()]
    ret = dict(country=ov_count, rest=ov_rest, other_countries=ov_other, sea=ov_sea)
    return ret

##############################
# ONLY ENHANCEMENT CALUCLATION
##############################
#load_ct_data
def load_cams_data(
    cams_path: str,
    startdate: Union[np.datetime64, datetime],
    enddate: Union[np.datetime64, datetime],
) -> xr.DataArray:
    """Loads CT2019B flux data

    Args:
        ct_file (str): String describing file names of CT2019B flux data before time stamp, e.g: '/path/to/data/CT2019B.flux1x1.
        startdate (Union[np.datetime64, datetime]): Start of time frame to load data of 
        enddate (Union[np.datetime64, datetime]): End of time frame to load data of 

    Returns:
        xr.DataArray: Loaded flux data
    """    
    import matplotlib.pyplot as plt

    total_flux = xr.DataArray()
    first_flux = True
    for year in range(2019,2021):
        cams_flux_bio = xr.DataArray()
        cams_flux_ant = xr.Dataset()
        cams_flux_air = xr.Dataset()
        for sector in ['AIR_v1.1', 'BIO_v3.1', 'ANT_v4.2']:
            if sector == 'ANT_v4.2':
                cams_file = 'CAMS-AU-'+sector+'_carbon-monoxide_'+str(year)+'_sum_regr1x1.nc'
                flux_ant_part = xr.open_mfdataset(
                    cams_path+cams_file,
                    combine="by_coords",
                    chunks="auto",
                    )
                flux_ant_part = flux_ant_part.assign_coords({'year': ('time', [year*i for i in np.ones(12)])})
                flux_ant_part = flux_ant_part.assign_coords({'MonthDate': ('time', [datetime(year=year, month= i, day = 1) for i in np.arange(1,13)])})
                cams_flux_ant = flux_ant_part

            elif sector == 'BIO_v3.1': 
                cams_file = 'CAMS-AU-'+sector+'_carbon-monoxide_2019_regr1x1.nc'            
                flux_bio_part = xr.open_mfdataset(
                cams_path+cams_file,
                combine="by_coords",
                chunks="auto",
                )
                #datetime_var = []
                #for h_step in range(8): 
                #    print(h_step*3)
                #    print(flux_bio_part.emiss_bio.isel({'TSTEP':np.arange(h_step*3,h_step*3+3,1)}).mean('TSTEP'))
                #datetime_var.append(datetime(year = year, month = m, day = )for m in [1,2,3,4,5,6,7,8,9,10,11,12])
                    
                flux_bio_part = flux_bio_part.emiss_bio.mean('TSTEP')
                #print(flux_bio_part)
                flux_bio_part = flux_bio_part.assign_coords(dict({'time': ('time',[0,1,2,3,4,5,6,7,8,9,10,11] )}))
                flux_bio_part = flux_bio_part.assign_coords({'year': ('time', [year*i for i in np.ones(12)])})
                flux_bio_part = flux_bio_part.assign_coords({'MonthDate': ('time', [datetime(year=year, month= i, day = 1) for i in np.arange(1,13)])})
                cams_flux_bio = flux_bio_part
                #print(cams_flux_bio)
                if year ==2019: 
                    plt.figure()
                    cams_flux_bio.isel(dict({'time':11})).plot(x='longitude', y = 'latitude')
                    plt.savefig('/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/CAMS_bio_2019.png')
              
            

        total_flux_yr = cams_flux_ant['sum'] + cams_flux_bio
        if first_flux:
                first_flux = False
                total_flux = total_flux_yr
        else:
            total_flux = xr.concat([total_flux, total_flux_yr], dim = 'time')

        '''
        elif sector=='AIR_v1.1': 
            print('Air')
            cams_file = 'CAMS-AU-'+sector+'_carbon-monoxide_'+str(year)+'_regr1x1.nc'            
            flux_air_part = xr.open_mfdataset(
            cams_path+cams_file,
            combine="by_coords",
            chunks="auto",
            )

            print(flux_air_part['level'])
            if first_air:
                first_air = False
                cams_flux_air = flux_air_part
            else:
                cams_flux_air = xr.concat([cams_flux_air, flux_air_part], dim="time")
            print('finished air')
        '''
    #print('load_cams_flux : total_flux')
    #print(total_flux)

     # can be deleted when footprint has -179.5 coordinate
    #cams_flux = cams_flux[:, :, 1:]
    return total_flux


def get_fp_array(fp_dataset: xr.Dataset) -> xr.DataArray:
    """Extracts and formats footprint data

    Args:
        fp_dataset (xr.Dataset): Footprint data

    Returns:
        xr.DataArray: Formated dataarray of footprint
    """    
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        fd_array = fp_dataset.spec001_mr[0, :, :, 0, :, :]
    dt = np.timedelta64(90, "m")
    fd_array = fd_array.assign_coords(time=(fd_array.time + dt))
    # flip time axis
    fd_array = fd_array.sortby("time")
    return fd_array

def combine_flux_and_footprint(
    flux: xr.DataArray, footprint: xr.DataArray, chunks: dict = None
) -> xr.Dataset:
    """Computes enhancement for each release in footprint data

    Args:
        flux (xr.DataArray): Carbon Tracker Fluxes
        footprint (xr.DataArray): Footprint data
        chunks (dict, optional): Chunks for dask to allow faster with smaller memory demand. Defaults to None.

    Returns:
        xr.Dataset: Enhancements for each layer
    """
    if chunks is not None:
        footprint = footprint.chunk(chunks=chunks)
        flux = flux.chunk(chunks=chunks)
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        print(flux.to_dataset(name = 'total_flux').total_flux.values.max())
        print(footprint.to_dataset().spec001_mr.values.max())
        #print(flux)
        fp_co = xr.merge([footprint.to_dataset(), flux.to_dataset(name = 'total_flux')])
      #  print(fp_co)
    # 1/layer height*flux [kg/m²*s]*MCO[kg/mol]*fp[m^3*s/kg] * Mair[kg/mol]
    print('fp_co:')
    #print(fp_co.spec001_mr.values.max())
    #print(fp_co.total_flux.values.max())
    #print(fp_co.time.values)
    tot_fp_co = xr.Dataset()
    for i, time in enumerate(footprint.time.values): 
        #print(i)
        #print(time)
        fp_co = (
            1 / 100 * flux/0.028 * footprint[:,i,:,:] * 0.029*10**9)
        #  1 / 100 * fp_co.total_flux*0.028 * fp_co.spec001_mr * 0.029) 
         # dim : time, latitude, longitude, pointspec: 36
         # #.astype(int) -time.astype('datetime64[Y]').astype(int)+ 1)
        t = pd.to_datetime(time)
        fp_co = fp_co.assign_coords({'time':('time', [t])})
        fp_co.name = 'CO'
        tot_fp_co = xr.merge([tot_fp_co, fp_co])
    # sum over time, latitude and longitude, remaining dim = layer
    #print('fp_co:')
    #print(fp_co.values)
    tot_fp_co = tot_fp_co.sum(dim=["time", "latitude", "longitude"])
    tot_fp_co = tot_fp_co.compute()
    #print(tot_fp_co)
    #tot_fp_co.name = "CO"
    #tot_fp_co = tot_fp_co.to_dataset()
    #print('fp_co before return')
    #print(fp_co)
    return tot_fp_co

def calc_enhancement(
    fp_data: xr.Dataset,
    cams_path: str,
    startdate: Union[np.datetime64, datetime],
    enddate: Union[np.datetime64, datetime],
    boundary: list[float, float, float, float] = None,
    chunks: dict = None,
) -> np.ndarray:
    """Calculates enhancement using CT2019B fluxes and FLEXPART output.

    Args:
        fp_data (xr.Dataset): FLEXPART output dataset
        ct_file (str): String describing file names of CT2019B flux data before time stamp, e.g: '/path/to/data/CT2019B.flux1x1.
        startdate (Union[np.datetime64, datetime]):  Start of time frame to load data of
        enddate (Union[np.datetime64, datetime]):  End of time frame to load data of
        boundary (list[float, float, float, float], optional): boundary to calculate enhancement for [longitude left, longitude right, latitude lower, latitude upper]. Defaults to None.
        chunks (dict, optional): Chunks for dask to allow faster with smaller memory demand. Defaults to None.

    Returns:
        np.ndarray: Array of enhancement for each release of FLEXPART run
    """    
    # test the right order of dates
    assert enddate > startdate, "The startdate has to be before the enddate"

    # read CT fluxes (dayfiles) and merge into one file
    cams_flux = load_cams_data(cams_path, startdate, enddate).compute()

    # from .interp can be deleted as soon as FP dim fits CTFlux dim, layers repeated???? therfore only first 36
    footprint = get_fp_array(fp_data).compute()
    # selct times of Footprint in fluxes
    #print(footprint.time.min())
    #print(footprint.time.min().values.astype('datetime64[M]').astype(int) % 12 + 1)# % 12 + 1)
    #print(pd.to_datetime(footprint.time.min()))
    #print(datetime(year=year, month= i, day = 1))
    fp_time_min = datetime(year = footprint.time.min().values.astype('datetime64[Y]').astype(int) + 1970, 
                           month = footprint.time.min().values.astype('datetime64[M]').astype(int) % 12 + 1 , day = 1)
    fp_time_max = datetime(year = footprint.time.max().values.astype('datetime64[Y]').astype(int) + 1970, 
                           month = footprint.time.max().values.astype('datetime64[M]').astype(int) % 12 + 1 , day = 1)
    #print('fp_time_max'+str(footprint.time.min()))
    #print(fp_time_min)
    cams_flux = cams_flux.set_index(time="MonthDate")
    cams_flux = cams_flux.sel(time=slice(fp_time_min,fp_time_max))
    #print(cams_flux)
    if boundary is not None:
        cams_flux = select_extent(cams_flux, *boundary)

    # OLDVERSION pressure_weights = 1/(p_surf-0.1)*(fp_pressure_layers[0:-1]-fp_pressure_layers[1:])
    # FP_pressure_layers = np.array(range(numLayer,0,-1))*1000/numLayer
    fp_co = combine_flux_and_footprint(cams_flux, footprint, chunks=chunks)

    molefractions = fp_co.CO.values
    print('enhancement molefractions:')
    print(molefractions)
    return molefractions 

#######################
# Optimal Lambda
#######################

def calc_point(reg: bayesinverse.Regression, alpha:float) -> tuple[float, float]:
    """Calculates points on L-curve given a regulatization parameter and the model

    Args:
        reg (bayesinverse.Regression): Regression model
        alpha (float): Regularization weight factor

    Returns:
        tuple[float, float]: Logarithm of losses of forward model and regulatization
    """    
    result = reg.compute_l_curve([alpha])
    return np.log10(result["loss_forward_model"][0]), np.log10(result["loss_regularization"][0])

def euclidean_dist(P1: tuple[float, float], P2: tuple[float, float]) -> float:
    """Calculates euclidean distanceof two points given in tuples

    Args:
        P1 (tuple[float, float]): Point 1
        P2 (tuple[float, float]): Point 2

    Returns:
        float: distance
    """    
    P1 = np.array(P1)
    P2 = np.array(P2)
    return np.linalg.norm(P1-P2)**2

def calc_curvature(Pj: tuple[float, float], Pk: tuple[float, float], Pl: tuple[float, float]) -> float:
    """Calculates menger curvature of circle by three points

    Args:
        Pj (tuple[float, float]): Point with smallest regularization weight
        Pk (tuple[float, float]): Point with regularization weight in between 
        Pl (tuple[float, float]): Point with largest regulatization weight

    Returns:
        float: curvature
    """    
    return 2*(Pj[0] * Pk[1] + Pk[0] * Pl[1] + Pl[0] * Pj[1] - Pj[0] * Pl[1] - Pk[0] * Pj[1] - Pl[0] * Pk[1])/np.sqrt(euclidean_dist(Pj, Pk) * euclidean_dist(Pk, Pl) * euclidean_dist(Pl, Pj))

def get_l2(l1: float, l4: float) -> float:
    """Calculates position of regularization weight 2 from full boundary

    Args:
        l1 (float): Regularization weight lower boundary
        l4 (float): Regularization weight upper boundary

    Returns:
        float: New regularization weight in between
    """      
    phi = (1 + np.sqrt(5))/2
    
    x1 = np.log10(l1)
    x4 = np.log10(l4)
    x2 = (x4 + phi * x1) / (1 + phi)
    return 10**x2

def get_l3(l1: float, l2: float, l4: float) -> float:
    """Calculates position of regularization weight 3 other values

    Args:
        l1 (float): Regularization weight lower boundary
        l2 (float): Regularization weight in between
        l4 (float): Regularization weight upper boundary

    Returns:
        float: New regularization weight
    """    
    return 10**(np.log10(l1) + (np.log10(l4) - np.log10(l2)))

def optimal_lambda(reg: bayesinverse.Regression, interval: tuple[float, float], threshold: float):    
    """Find optiomal value for weigthing factor in regression. Based on https://doi.org/10.1088/2633-1357/abad0d

    Args:
        reg (bayesinverse.Regression): regression Model
        interval (tuple[float, float]): Values of weightung factors. Values within to search 
        threshold (float): Search stops if normalized search interval (upper boundary - lower boundary)/ upper boundary is smaller then threshold
    """    
    
    l1 = interval[0]
    l4 = interval[1]
    l2 =  get_l2(l1, l4)
    l3 = get_l3(l1, l2, l4)
    l_list = [l1, l2, l3, l4]

    p_list = []
    for l in l_list:
        p_list.append(calc_point(reg, l))
    
    while (l_list[3] - l_list[0]) / l_list[3] >= threshold:
        c2 = calc_curvature(*p_list[:3])
        c3 = calc_curvature(*p_list[1:])
        # Find convex part of curve
        while c3 <= 0:
            l_list[3] = l_list[2]
            l_list[2] = l_list[1]
            p_list[3] = p_list[2]
            p_list[2] = p_list[1]
            l_list[1] = get_l2(l_list[0], l_list[3])
            p_list[1] = calc_point(reg, l_list[1])
            c3 = calc_curvature(*p_list[1:])
            # print(l_list[0], l_list[3])
        # Approach higher curvature
        if c2 > c3:
            # Store current guess
            l = l_list[1]
            # Set new boundaries
            l_list[3] = l_list[2]
            l_list[2] = l_list[1]
            p_list[3] = p_list[2]
            p_list[2] = p_list[1]
            l_list[1] = get_l2(l_list[0], l_list[3])
            p_list[1] = calc_point(reg, l_list[1])
        else:
            # Store current guess
            l = l_list[2]
            # Set new boundaries
            l_list[0] = l_list[1]
            p_list[0] = p_list[1]
            l_list[1] = l_list[2]
            p_list[1] = p_list[2]
            l_list[2] = get_l3(l_list[0], l_list[1], l_list[3])
            p_list[2] = calc_point(reg, l_list[2])
        # print(l)
    return l