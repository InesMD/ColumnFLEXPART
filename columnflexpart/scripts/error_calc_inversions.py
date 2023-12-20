import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
datapath_CO_gridded = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/'
datapath_CO_ECO = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Ecosystems_AK_split_with_4/'
datapath_CO2_gridded = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/'
datapath_CO2_ECO = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Ecoregions_split_4/'

#CO_prior_grid = xr.open_mfdataset(datapath_CO_gridded+"1e-05_CO_prior_error_week*.nc", combine='by_coords')
#CO_post_grid = xr.open_mfdataset(datapath_CO_gridded+"1e-05_CO_posterior_error_week*.nc", combine='by_coords')
#CO2_prior_grid = xr.open_mfdataset(datapath_CO2_gridded+"1e-05_CO2_prior_error_week*.nc", combine='by_coords')
#CO2_post_grid = xr.open_mfdataset(datapath_CO2_gridded+"1e-05_CO2_posterior_error_week*.nc", combine='by_coords')

def ratio_err_CO_over_CO2(errCO, errCO2, fluxCO, fluxCO2): 
    return np.sqrt((errCO/fluxCO2)**2+(fluxCO*errCO2/fluxCO2**2)**2)

bioclass_mask_gridded = "/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1"
bioclass_maksed_final = "/home/b/b382105/ColumnFLEXPART/resources/Ecosystems_AK_based_split_with_21_and_20_larger_and_4_split.nc"

#CO_prior_eco = xr.open_mfdataset(datapath_CO_ECO+"0.03_CO_prior_error_week_*.nc", combine='by_coords')
#CO_post_eco = xr.open_mfdataset(datapath_CO_ECO+"0.03_CO_posterior_error_week*.nc", combine='by_coords')
#CO2_prior_eco = xr.open_mfdataset(datapath_CO2_ECO+"0.03_CO2_prior_error_week*.nc", combine='by_coords')
#CO2_post_eco = xr.open_mfdataset(datapath_CO2_ECO+"0.03_CO2_posterior_error_week*.nc", combine='by_coords')
def map_on_grid(bioclass_mask, xarr: xr.DataArray) -> xr.DataArray:#[self.time_coord]
    mapped_xarr = xr.DataArray(
        data = np.zeros(
            (
                len(xarr['week']),
                len(bioclass_mask.longitude),
                len(bioclass_mask.latitude)
            )
        ),
        coords = {
            'week': xarr['week'], 
            "longitude": bioclass_mask.longitude, 
            "latitude": bioclass_mask.latitude
        }
    )
    for bioclass in xarr.bioclass:
        mapped_xarr = mapped_xarr + (bioclass_mask == bioclass) * xarr.where(xarr.bioclass == bioclass, drop=True)
        mapped_xarr = mapped_xarr.squeeze(drop=True) # seems to drop every value for CO
    return mapped_xarr

#masks
datapathmask_eco = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Ecoregions_split_4/flat_error_area_scaled/total_emissions/'
name_maks_eco = '3.00e-02_CO2_3.00e-02_CO_mask_ak_threshold_0.001and_co_5e-1_week_'
name_mask_strict_eco = '3.00e-02_CO2_3.00e-02_CO_mask_ak_threshold_0.1and_co_5e-1_week_'
#gridded:
datapathmask_gridded = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/total_emissions_and_ratios_check/'
name_mask_gridded = '1.00e-05_CO2_1.00e-05_CO_mask_ak_threshold_0.001and_co_5e-1_week_'

import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
def plotting(mask, ratio,week, path_and_name_bioclass_mask):
    plt.figure(figsize = (11,9))
    plt.rcParams.update({'font.size':25})
    ax = plt.axes(projection=ccrs.PlateCarree())  
    my_cmap = LinearSegmentedColormap.from_list('', ['gainsboro','white'])
    bioclasses = xr.open_dataset(path_and_name_bioclass_mask)
    bioclasses = bioclasses.where((bioclasses.Long >=110)&(bioclasses.Long <=155)&(bioclasses.Lat >=-45)&(bioclasses.Lat <=-10))
    bioclasses = bioclasses.rename({'bioclass' : '__xarray_dataarray_variable__', 'Long': 'longitude', 'Lat': 'latitude'})
    true_mask = mask.__xarray_dataarray_variable__.where(bioclasses.__xarray_dataarray_variable__>0)
    true_mask.plot(x = 'longitude', y ='latitude', cmap = my_cmap, add_colorbar = False)
    
    ratio.plot(x='longitude', y='latitude', vmax = 1000, vmin = 0,cmap = 'pink_r',  cbar_kwargs = {'label' : f'weekly $\Delta$CO/$\Delta$CO$_2$ [ppb/ppm]', 'shrink': 0.77})
    plt.scatter(x = 150.8793,y =-34.4061,color="black")
    #plt.title('Fire emission ratios 2019 week '+str(week))
    
    ax.coastlines()
    plt.ylim((-45, -10))
    plt.xlim((110, 155))
    plt.show()
    plt.savefig(datapath_CO2_gridded+'ratio_error_1000_times_errors'+str(week)+'.png', bbox_inches = 'tight', dpi = 250)
    plt.close()

def gridded():
    # GRIDDED
    eco_mask = xr.open_dataset(bioclass_mask_gridded).rename({'Lat':'latitude', 'Long': 'longitude'})
    print(eco_mask.bioclass)
    for week in [1,48,49,50,51,52]:
        CO2_spatial_result = xr.open_dataarray(datapath_CO2_gridded+"results_bioclass_week"+str(week)+'.nc')*10**6*12
        CO_spatial_result = xr.open_dataarray(datapath_CO_gridded+'results_bioclass_week'+str(week)+'.nc')*10**6*12
        #print(CO2_spatial_result.max())
    # print(CO_spatial_result_ECO)
    #   print(CO_post_eco)
        CO_post_grid = xr.open_dataarray(datapath_CO_gridded+"1e-05_CO_posterior_error_week"+str(week)+".nc")

        #print(CO_post_grid)
        #CO2_prior_grid = xr.open_mfdataset(datapath_CO2_gridded+"1e-05_CO2_prior_error_week*.nc", combine='by_coords')
        CO2_post_grid = xr.open_dataarray(datapath_CO2_gridded+"1e-05_CO2_posterior_error_week"+str(week)+".nc")
        print('CO2')
        print(CO2_post_grid.max())
        ratio_err = ratio_err_CO_over_CO2(CO_post_grid, CO2_post_grid, CO_spatial_result, CO2_spatial_result )
        #print(ratio_err)
        #ratio_err = ratio_err.where(week == week)#.drop('bioclass')
    #  print(ratio_err)
        spatial_ratio_err = map_on_grid(eco_mask.bioclass, ratio_err)
        print(spatial_ratio_err)
        #ratio_err.to_netcdf(datapath_CO2_ECO+'ratio_err_week'+str(week)+'.nc')
        spatial_ratio_err = spatial_ratio_err.squeeze()#.rename({'bioclass0':'__xarray_dataarray_variable__'})
    # print(spatial_ratio_err)
        mask = xr.open_dataset(datapathmask_gridded+name_mask_gridded+str(week)+'.nc')#.rename({'__xarray_dataarray_variable__': 'bioclass0'})
        #print(mask)
        spatial_ratio_err = spatial_ratio_err*mask.__xarray_dataarray_variable__
        print(spatial_ratio_err)
        spatial_ratio_err = xr.where(spatial_ratio_err!=0,spatial_ratio_err.values, np.nan)
        plotting(mask, spatial_ratio_err*10**3,week, bioclass_mask_gridded)


def eco(): 
    eco_mask = xr.open_dataset(bioclass_maksed_final).rename({'Lat':'latitude', 'Long': 'longitude'})
    #print(eco_mask)
    for week in [1,48,49,50,51,52]:
        CO2_spatial_result_ECO = xr.open_dataset(datapath_CO2_ECO+"flat_error_area_scaled/results_bioclass_week"+str(week)+'.nc')*10**6*12
        CO_spatial_result_ECO = xr.open_dataset(datapath_CO_ECO+'results_bioclass_week'+str(week)+'.nc')*10**6*12
    # print(CO_spatial_result_ECO)
    #   print(CO_post_eco)
        CO_post_eco = xr.open_dataarray(datapath_CO_ECO+"0.03_CO_posterior_error_week"+str(week)+".nc")
        CO2_post_eco = xr.open_dataarray(datapath_CO2_ECO+"0.03_CO2_posterior_error_week"+str(week)+".nc")
        ratio_err = ratio_err_CO_over_CO2(CO_post_eco, CO2_post_eco, CO_spatial_result_ECO, CO2_spatial_result_ECO )
        print(CO_post_eco.max())
        ratio_err = ratio_err.where(week == week)#.drop('bioclass')
    #  print(ratio_err)
        spatial_ratio_err = map_on_grid(eco_mask.bioclass, ratio_err.__xarray_dataarray_variable__)
        #ratio_err.to_netcdf(datapath_CO2_ECO+'ratio_err_week'+str(week)+'.nc')
        spatial_ratio_err = spatial_ratio_err.squeeze()#.rename({'bioclass0':'__xarray_dataarray_variable__'})
    # print(spatial_ratio_err)
        mask = xr.open_dataset(datapathmask_eco+name_maks_eco+str(week)+'.nc')#.rename({'Lat': 'latitude', 'Long': 'longitude', 'bioclass': '__xarray_dataarray_variable__'})#.rename({'__xarray_dataarray_variable__': 'bioclass0'})
        print(mask)
        spatial_ratio_err = spatial_ratio_err*mask.__xarray_dataarray_variable__
        spatial_ratio_err = xr.where(spatial_ratio_err!=0,spatial_ratio_err, np.nan)
        plotting(mask, spatial_ratio_err*10**3, week,"/home/b/b382105/ColumnFLEXPART/resources/Ecosystems_AK_based_split_with_21_and_20_larger_and_4_split.nc")

gridded()