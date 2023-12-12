
from columnflexpart.classes.coupled_inversion import CoupledInversion 
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib import colors
import matplotlib as mpl

def split_predictions(Inversion):
    predictionsCO2_fire = Inversion.predictions_flat[Inversion.predictions_flat['bioclass']<Inversion.predictions_flat['bioclass'].shape[0]/4]
    predictionsCO2_bio = Inversion.predictions_flat[(Inversion.predictions_flat['bioclass']<Inversion.predictions_flat['bioclass'].shape[0]/2)
                                                &(Inversion.predictions_flat['bioclass']>=Inversion.predictions_flat['bioclass'].shape[0]/4)]
    predictionsCO_fire = Inversion.predictions_flat[(Inversion.predictions_flat['bioclass']>=Inversion.predictions_flat['bioclass'].shape[0]/2)
                                               &(Inversion.predictions_flat['bioclass']<Inversion.predictions_flat['bioclass'].shape[0]*3/4)]
    predictionsCO_bio = Inversion.predictions_flat[(Inversion.predictions_flat['bioclass']>=Inversion.predictions_flat['bioclass'].shape[0]*3/4)]

    predictionsCO2_bio = predictionsCO2_bio.assign_coords(bioclass = np.arange(0,len(predictionsCO2_fire['bioclass'])))
    predictionsCO_fire = predictionsCO_fire.assign_coords(bioclass = np.arange(0,len(predictionsCO2_fire['bioclass'])))
    predictionsCO_bio = predictionsCO_bio.assign_coords(bioclass = np.arange(0,len(predictionsCO2_fire['bioclass'])))

    return [predictionsCO2_fire, predictionsCO2_bio, predictionsCO_fire, predictionsCO_bio]


def split_output(xarray_to_split, variable_name):

    predictionsCO2_fire = xarray_to_split[xarray_to_split[variable_name]<int(xarray_to_split[variable_name].shape[0]/4)]
    predictionsCO2_bio = xarray_to_split[(xarray_to_split[variable_name]<int(xarray_to_split[variable_name].shape[0]/2))
                                                &(xarray_to_split[variable_name]>=int(xarray_to_split[variable_name].shape[0]/4))]
    predictionsCO_fire = xarray_to_split[(xarray_to_split[variable_name]>=int(xarray_to_split[variable_name].shape[0]/2))
                                               &(xarray_to_split[variable_name]<int(xarray_to_split[variable_name].shape[0]*3/4))]
    predictionsCO_bio = xarray_to_split[(xarray_to_split[variable_name]>=int(xarray_to_split[variable_name].shape[0]*3/4))]

    predictionsCO2_bio = predictionsCO2_bio.assign_coords({variable_name : np.arange(0,len(predictionsCO2_fire[variable_name]))}).rename({variable_name: 'bioclass'})
    predictionsCO_fire = predictionsCO_fire.assign_coords({variable_name :  np.arange(0,len(predictionsCO2_fire[variable_name]))}).rename({variable_name: 'bioclass'})
    predictionsCO_bio = predictionsCO_bio.assign_coords({variable_name : np.arange(0,len(predictionsCO2_fire[variable_name]))}).rename({variable_name: 'bioclass'})
    predictionsCO2_fire = predictionsCO2_fire.rename({variable_name: 'bioclass'})

    return [predictionsCO2_fire, predictionsCO2_bio, predictionsCO_fire, predictionsCO_bio]


def split_eco_flux(Inversion):
        flux_eco = Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week]
        fluxCO2_fire = flux_eco[:Inversion.number_of_reg].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,Inversion.number_of_reg)) 
        fluxCO2_bio = flux_eco[Inversion.number_of_reg:2*Inversion.number_of_reg].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,Inversion.number_of_reg))
        fluxCO_fire = flux_eco[2*Inversion.number_of_reg: 3*Inversion.number_of_reg].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,Inversion.number_of_reg))
        fluxCO_bio = flux_eco[3*Inversion.number_of_reg:].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,Inversion.number_of_reg))

        return fluxCO2_fire*10**-6, fluxCO2_bio*10**-6, fluxCO_fire*10**-9, fluxCO_bio*10**-9

def split_gridded_flux(Inversion):
        flux_eco = Inversion.flux_grid[:,Inversion.flux_grid.week == Inversion.week]
        number_of_reg = 699
        fluxCO2_fire = flux_eco[:number_of_reg].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,number_of_reg)) 
        fluxCO2_bio = flux_eco[number_of_reg:2*number_of_reg].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,number_of_reg))
        fluxCO_fire = flux_eco[2*number_of_reg: 3*number_of_reg].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,number_of_reg))
        fluxCO_bio = flux_eco[3*number_of_reg:].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,number_of_reg))

        return fluxCO2_fire*10**-6, fluxCO2_bio*10**-6, fluxCO_fire*10**-9, fluxCO_bio*10**-9

def get_averaging_kernel(self, reduce: bool=False, only_cols: bool=False):
        ak = self.reg.get_averaging_kernel()
        if reduce:
            weeks = self.flux.week.values
            filtered_weeks = self.get_filtered_weeks(self.flux).values
            mask = np.ones((len(self.flux.week), self.n_eco))
            mask[~np.isin(weeks, filtered_weeks)] = 0
            mask = mask.flatten().astype(bool)
            ak = ak[mask]
            if not only_cols:
                ak = ak[:, mask]
        return ak

def calculate_averaging_kernel(Inversion): 
    ak = Inversion.get_averaging_kernel()

    akCO2_fire = ak[:Inversion.number_of_reg, :Inversion.number_of_reg]
    akCO2_bio = ak[Inversion.number_of_reg:Inversion.number_of_reg*2, Inversion.number_of_reg:Inversion.number_of_reg*2]
    akCO_fire = ak[2*Inversion.number_of_reg:Inversion.number_of_reg*3, 2*Inversion.number_of_reg:Inversion.number_of_reg*3]
    akCO_bio = ak[3*Inversion.number_of_reg:, 3*Inversion.number_of_reg:]
    
    return akCO2_fire, akCO2_bio, akCO_fire, akCO_bio
    
def plot_averaging_kernel(Inversion,ak_final, class_num, savepath, savename, alphaCO, alpha, name_part): 
    ak_xr = xr.DataArray(data = ak_final.diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,Inversion.number_of_reg))))#, week = [1,48,49,50,51,52])))
    ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, class_num)
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())  
    orig_map=plt.cm.get_cmap('gist_heat')
    reversed_map = orig_map.reversed()
    ak_spatial.plot( x='longitude', y='latitude',ax = ax,vmax = 1, vmin = 0,cmap = reversed_map, cbar_kwargs = {'shrink':0.835})#norm = LogNorm(vmin = 1e-3),
    plt.scatter(x = 150.8793,y =-34.4061,color="black")
    ax.coastlines()
    #plt.title('Averaging kernel for 2019/12')
    plt.savefig(savepath+savename, dpi = 250, bbox_inches = 'tight')
    ak_spatial.to_netcdf(savepath+str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_ak_spatial_'+name_part+'.nc')



def calculate_and_plot_averaging_kernel(Inversion, savepath, alpha, alphaCO, week): 
    akCO2_fire, akCO2_bio, akCO_fire, akCO_bio = calculate_averaging_kernel(Inversion)
    name_list = ['_CO2_fire_', '_CO2_bio_', '_CO_fire_', '_CO_bio_']
    for idx,ak in enumerate([akCO2_fire, akCO2_bio, akCO_fire, akCO_bio]):
        savename = str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_ak_spatial_'+name_list[idx]+'_'+str(week)+'.png'
        plot_averaging_kernel(Inversion,ak,Inversion.number_of_reg, savepath, savename, alphaCO, alpha, name_list[idx])



def plot_spatial_result(spatial_result, savepath, savename, cmap, vmax =None, vmin =None, cbar_kwargs = {'shrink':  0.835}, norm = None):
    '''
    for weekly spatial plotting of fluxes, saves plot 
    spatial_result : xarray to plot with latitude and longitude as coordinates
    molecule_name : Either "CO2" or "CO"
    savepath : path to save output image, must exist
    savename : Name of output image
    '''
    plt.rcParams.update({'font.size': 15})
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())  
    spatial_result.plot(x = 'longitude', y = 'latitude', ax = ax, cmap =cmap,vmin = vmin, vmax = vmax, norm = norm,
                            cbar_kwargs = cbar_kwargs)
    
    plt.scatter(x = 150.8793,y =-34.4061,color="black")
    ax.coastlines()
    #plt.title('Week '+str(week))
    plt.savefig(savepath+savename, bbox_inches = 'tight', dpi = 250)
    plt.close()

def plot_prior_spatially(Inversion, flux, name, idx, savepath, eco = True):
    plt.rcParams.update({'font.size':15})    
    factor = 10**6*12 
    #lux_mean = Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week]*factor
    if eco: 
        spatial_flux = Inversion.map_on_grid(flux*factor)
    else: 
        spatial_flux = Inversion.map_on_gridded_grid(flux*factor)
        
    spatial_flux.to_netcdf(savepath+'Prior_'+name+'_spatial_week_'+str(Inversion.week)+'.nc')
    savename = 'Prior_'+name+'_spatial_week'+str(Inversion.week)+'.png'
    #plot_spatial_result(spatial_flux, savepath,savename, 'seismic', vmax = 0.2, vmin = -0.2, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835}  )

    if idx == 0: 
        plot_spatial_result(spatial_flux, savepath,savename, 'seismic', vmax = 65, vmin = -65, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835}  )
    elif idx == 1:
        plot_spatial_result(spatial_flux, savepath,savename, 'seismic', vmax = 65, vmin = -65, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835}  )
    elif idx == 2:
        plot_spatial_result(spatial_flux, savepath,savename, 'seismic', vmax = 4, vmin = -4, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835}  )
    elif idx == 3:
        plot_spatial_result(spatial_flux, savepath,savename, 'seismic', vmax = 4, vmin = -4, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835}  )

def plot_emission_ratio(savepath, Inversion, predictionsCO2, predictionsCO, alpha, alphaCO, fluxCO2, fluxCO):
    # scaling factors
    predictions = predictionsCO2/predictionsCO
    spatial_result = Inversion.map_on_grid(predictions)# gleichwer name Feuer und nicht feuer
    savename = str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_ratio_CO2_over_CO_Scaling_factor_"+str(Inversion.week)+".png"

    max_value = max(abs(spatial_result.values.max()-1), abs(1-spatial_result.values.min()))
    divnorm=colors.TwoSlopeNorm(vmin = 1-max_value, vcenter=1, vmax = 1+max_value)
    plt.rcParams.update({'font.size':15})
    plot_spatial_result(spatial_result, savepath, savename, 'coolwarm', norm = divnorm)
    # flux: # für eco setup 
    spatial_flux = Inversion.map_on_grid(predictionsCO2/predictionsCO * fluxCO2/fluxCO*44/28)
    savename = str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_ratio_CO2_over_CO_flux_"+str(Inversion.week)+".png"
    max_value = max(abs(spatial_flux.values.max()-14.4), abs(14.4-spatial_flux.values.min()))
    divnorm=colors.TwoSlopeNorm(vmin = 14.4-max_value, vcenter=14.4, vmax = 14.4+max_value)
    plot_spatial_result(spatial_flux, savepath, savename, 'coolwarm',  cbar_kwargs = {'label' : r'$\Delta$CO$_2$/$\Delta$CO [gCO$_2$/gCO]', 'shrink': 0.835} ,norm = divnorm)
    divnorm=colors.TwoSlopeNorm(vmin = 14.4-1.9*2, vcenter=14.4, vmax = 14.4+1.9*2)
    savename = str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_ratio_CO2_over_CO_flux_cut_cbar_"+str(Inversion.week)+".png"
    plot_spatial_result(spatial_flux, savepath, savename, 'coolwarm',  cbar_kwargs = {'label' : r'$\Delta$CO$_2$/$\Delta$CO [gCO$_2$/gCO]', 'shrink': 0.835} ,norm = divnorm)
    divnorm=colors.TwoSlopeNorm(vmin = 14.4-17.6, vcenter=14.4, vmax = 14.4+17.6)
    savename = str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_ratio_CO2_over_CO_flux_cut_to_32_cbar_"+str(Inversion.week)+".png"
    plot_spatial_result(spatial_flux, savepath, savename, 'coolwarm',  cbar_kwargs = {'label' : r'$\Delta$CO$_2$/$\Delta$CO [gCO$_2$/gCO]', 'shrink': 0.835} ,norm = divnorm)
    #spatial_resultCO = Inversion.map_on_grid(predictionsCO)
    
    return

def plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions,flux, name, idx, alpha,alphaCO,vmin =None, vmax = None, diff =False):
    factor = 12*10**6
    plt.rcParams.update({'font.size':15})   
    
    spatial_result = Inversion.map_on_grid(predictions)
    if diff== True: 
        diff_flux = (predictions- xr.ones_like(predictions))* flux
        diff_spatial_flux = Inversion.map_on_grid(diff_flux)*factor 
        savename = str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_"+name+'Diff_to_prior_week_'+str(Inversion.week)+'.png'

        if idx == 2: 
            plot_spatial_result(diff_spatial_flux, savepath, savename, 'seismic',vmax = 10, vmin = -10, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        elif idx == 0:
                plot_spatial_result(diff_spatial_flux, savepath, savename, 'seismic', vmax = 300, vmin = -300,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        elif idx == 1 or idx == 3:
            plot_spatial_result(diff_spatial_flux, savepath, savename, 'seismic',vmax = 0.6, vmin = -0.6, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        #elif idx == 3:
        #    plot_spatial_result(diff_spatial_flux,"CO", savepath, savename, 'seismic', vmax = 10, vmin = -10,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
    
    else: 
        # plot scaling factors
        savename = str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_Scaling_factor_"+name+str(Inversion.week)+".png"
        if idx == 0 or idx == 1: 
            plot_spatial_result(spatial_result, savepath, savename, 'viridis')
        else:
            plot_spatial_result(spatial_result, savepath, savename, 'viridis')
        
        #plot fluxes
        savename = str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_Spatial_flux_"+name+str(Inversion.week)+".png"
        spatial_flux = Inversion.map_on_grid(predictions * flux)*factor

        if idx == 0: 
            plot_spatial_result(spatial_flux, savepath, savename, 'seismic',vmax = 300, vmin = -300, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        elif idx == 1:
            plot_spatial_result(spatial_flux, savepath, savename, 'seismic', vmax = 10, vmin = -10,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        elif idx == 2:
            if vmax == None and vmin == None: 
                plot_spatial_result(spatial_flux, savepath, savename, 'seismic',vmax = 10, vmin = -10, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
            else: 
              plot_spatial_result(spatial_flux, savepath, savename, 'seismic',vmax = vmax, vmin = vmin, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})  
        elif idx == 3:
            plot_spatial_result(spatial_flux, savepath, savename, 'seismic', vmax = 0.1, vmin = -0.1,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        #predictions = predictions.where((predictions.bioclass.values >4 and predictions.bioclass.values < 26), drop= True)
        #flux = flux.where((predictions.bioclass >4 and predictions.bioclass < 26), drop = True)
        #spatial_flux = Inversion.map_on_grid(predictions * flux)*factor
        spatial_flux.to_netcdf(path =savepath+'spatial_results'+name+'_week_'+str(Inversion.week)+'.nc')

        #predictions = predictions.where((predictions.bioclass >4 and predictions.bioclass < 26 and predictions.bioclass != 19 and predictions.bioclass != 23), drop= True)
        #flux = flux.where((predictions.bioclass >4 and predictions.bioclass < 26), drop = True)
        #spatial_flux = Inversion.map_on_grid(predictions * flux)*factor
        #spatial_flux.to_netcdf(path =savepath+'spatial_results_southeast_without_19_and_23_'+name+'_week_'+str(Inversion.week)+'.nc')


def plot_spatial_averaging_kernel(ak_spatial, savepath,alpha, molecule_name, weekly= False, week = 0 ): 
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())  
    orig_map=plt.cm.get_cmap('gist_heat') 
    reversed_map = orig_map.reversed()
    ak_spatial.plot( x='longitude', y='latitude',vmin = 0, vmax = 1,ax = ax,cmap = reversed_map,cbar_kwargs = {'shrink':0.835})#norm = LogNorm(vmin = 1e-3),
    plt.scatter(x = 150.8793,y =-34.4061,color="black")
    ax.coastlines()
    plt.title(molecule_name)
    if weekly == True: 
        plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_ak_spatial_week'+str(week)+'.png', bbox_inches = 'tight', dpi = 450)
    else: 
        plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_ak_spatial.png', bbox_inches = 'tight', dpi = 450)
    plt.close()

'''
def plot_averaging_kernel(Inversion, alpha, class_num, week_num,savepath, plot_spatially, weekly = False):
    
    ak_finalCO2, ak_finalCO, ak_sumCO2, ak_sumCO = Inversion.calculate_averaging_kernel(class_num, week_num)
    plt.rcParams.update({'font.size':15})
    if plot_spatially ==  True and weekly == False: 
        ak_xrCO2 = xr.DataArray(data = ak_finalCO2.diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))#, week = [1,48,49,50,51,52])))
        ak_spatialCO2 = Inversion.map_on_grid_without_time_coord(ak_xrCO2, class_num)
        plot_spatial_averaging_kernel(ak_spatialCO2, savepath, alpha, 'CO2')

        ak_xrCO = xr.DataArray(data = ak_finalCO.diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))#, week = [1,48,49,50,51,52])))
        ak_spatialCO = Inversion.map_on_grid_without_time_coord(ak_xrCO, class_num)
        plot_spatial_averaging_kernel(ak_spatialCO, savepath, alpha, 'CO')

    #sk_shape # Seite 47
    elif plot_spatially == True and weekly == True: 
        week_list = [1,48,49,50,51,52]
        for week in range(0,week_num): 
            ak_xrCO2 = xr.DataArray(data = ak_sumCO2[week*class_num:(week+1)*class_num].diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))
            ak_spatialCO2 = Inversion.map_on_grid_without_time_coord(ak_xrCO2, class_num)
            plot_spatial_averaging_kernel(ak_spatialCO2, savepath, alpha, 'CO2', weekly, week_list[week] )

            ak_xrCO = xr.DataArray(data = ak_sumCO[week*class_num:(week+1)*class_num].diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))
            ak_spatialCO = Inversion.map_on_grid_without_time_coord(ak_xrCO, class_num)
            plot_spatial_averaging_kernel(ak_spatialCO2, savepath, alpha, 'CO', weekly, week_list[week])

    else: 
        plt.figure()
        plt.imshow(ak_finalCO2, vmin = 0, vmax = 1) #vmax = 1,
        plt.colorbar()
        plt.savefig(savepath+str("{:e}".format(alpha))+'_CO2_ak_final_2d.png')
        plt.close()

        plt.figure()
        plt.imshow(ak_finalCO, vmin = 0, vmax = 1) #vmax = 1,
        plt.colorbar()
        plt.savefig(savepath+str("{:e}".format(alpha))+'_CO_ak_final_2d.png')
        plt.close()
 
'''
def plot_input_mask(savepath,datapath_and_name, selection= None): 
    ds = xr.open_dataset(datapath_and_name)
    if selection != None: 
        ds0 = ds.where(ds.bioclass==selection[0])
        for i in selection: 
            dsi = ds.where(ds.bioclass==i, drop = True)
            ds0 = xr.merge([ds0,dsi])
        ds = ds0

    #ds_ecosystem = xr.open_dataset('/home/b/b382105/ColumnFLEXPART/resources/bioclass_mask1.nc')
    plt.rcParams.update({'font.size':25})    
    plt.figure(figsize=(14, 10))    
    ax = plt.axes(projection=ccrs.PlateCarree())   
    ds['bioclass'].plot(x = 'Long', y = 'Lat',cmap = 'nipy_spectral',add_colorbar = True, cbar_kwargs = {'shrink':  0.88, 'label' : r'region number'})
    ax.set_xlim((110.0, 155.0))
    ax.set_ylim((-45.0, -10.0))
    ax.coastlines()
    plt.savefig(savepath+'bioclasses.png',dpi = 250, bbox_inches = 'tight')

    return len(set(ds.bioclass.values.flatten()))


def create_mask_based_on_regions(savepath,datapath_and_name, selection= None): 
    ds = xr.open_dataset(datapath_and_name)
    if selection != None: 
        ds0 = ds.where((ds.bioclass.values > 4) &( ds.bioclass.values < 26), 0)
        ds0 = ds0.where((ds0.bioclass.values ==0), 1)

        ds1 = ds.where(((ds.bioclass.values > 4) &( ds.bioclass.values < 19))|((ds.bioclass.values>19)&(ds.bioclass.values<23))|((ds.bioclass.values<26)&(ds.bioclass.values>23)), 0)
        ds1 = ds1.where((ds1.bioclass.values ==0), 1)
        #for i in selection: 
        #    dsi = ds.where(ds.bioclass==i, drop = True)
        #    ds0 = xr.merge([ds0,dsi])
        #ds = ds0

    #ds_ecosystem = xr.open_dataset('/home/b/b382105/ColumnFLEXPART/resources/bioclass_mask1.nc')
    plt.rcParams.update({'font.size':25})    
    plt.figure(figsize=(14, 10))    
    ax = plt.axes(projection=ccrs.PlateCarree())   
    ds0['bioclass'].plot(x = 'Long', y = 'Lat',cmap = 'nipy_spectral',add_colorbar = True, cbar_kwargs = {'shrink':  0.88, 'label' : r'region number'})
    ax.set_xlim((110.0, 155.0))
    ax.set_ylim((-45.0, -10.0))
    ax.coastlines()
    ds0.to_netcdf(savepath+'input_mask_masked.nc')
    ds1.to_netcdf(savepath+'input_mask_masked_without_19_and_23.nc')
    plt.savefig(savepath+'bioclasses_mask.png',dpi = 250, bbox_inches = 'tight')
    plt.figure(figsize=(14, 10))    
    ax = plt.axes(projection=ccrs.PlateCarree())   
    ds1['bioclass'].plot(x = 'Long', y = 'Lat',cmap = 'nipy_spectral',add_colorbar = True, cbar_kwargs = {'shrink':  0.88, 'label' : r'region number'})
    ax.set_xlim((110.0, 155.0))
    ax.set_ylim((-45.0, -10.0))
    ax.coastlines()
    plt.savefig(savepath+'bioclasses_mask_without_19_and_23.png',dpi = 250, bbox_inches = 'tight')

    return len(set(ds.bioclass.values.flatten()))      


def multiply_footprints_with_fluxes_for_concentrations(Inversion, flux, footprints, factor, background):
    footprints_flat = footprints.stack(new=[Inversion.time_coord, *Inversion.spatial_valriables])
    concentration_results = footprints_flat.values*factor*flux.values
    conc_sum = concentration_results.sum(axis = 1)
    conc_tot = conc_sum + background

    return conc_tot

def multiply_K_with_scaling_factors_for_concentrations(Inversion, flux, footprints, factor, background):
    footprints_flat = footprints.stack(new=[Inversion.time_coord, *Inversion.spatial_valriables])
    concentration_results = footprints_flat.values*factor#*flux.values
    conc_sum = concentration_results.sum(axis = 1)
    conc_tot = conc_sum + background

    return conc_tot
    

def calc_total_conc_by_multiplication_with_K(Inversion, predictions_flat):
    '''
    multiplication of posterior scaling factors with entire K matrix to obtain total CO and CO2 concentrations 
    '''
    K = Inversion.K.where((Inversion.K.week == Inversion.week), drop = True)
    
    conc = K.dot(predictions_flat.rename(dict(bioclass = 'final_regions')))
    prior_conc = K.dot(xr.ones_like(predictions_flat.rename(dict(bioclass = 'final_regions'))))

    predictions_CO2, predictions_CO = Inversion.select_relevant_times(printing = False)

    #number of CO measurements: 
    nmeas = len(predictions_CO['background_inter'][:])

    concCO2 = conc[:nmeas,0]+predictions_CO2['background_inter']
    concCO = conc[nmeas:,0]+predictions_CO['background_inter']

    priorCO2 = prior_conc[:nmeas,0]+predictions_CO2['background_inter']
    priorCO = prior_conc[nmeas:,0]+predictions_CO['background_inter']

    return concCO2, concCO, priorCO2, priorCO, predictions_CO, predictions_CO2

def calc_single_conc_by_multiplication_with_K(Inversion, predictions_flat):
    '''
    multiplication of posterior scaling factors with splitted K matrix to obtain total CO and CO2 concentrations 
    '''
    K = Inversion.K.where((Inversion.K.week == Inversion.week), drop = True)
    predictions_CO2, predictions_CO = Inversion.select_relevant_times(printing = False)
    #number of CO measurements: 
    nmeas = len(predictions_CO['background_inter'][:])

    # split K matrix 
    K_CO2fire = K.where((K.final_regions< len(K.final_regions.values)*1/4)&
                        (K.measurement< K.measurement.shape[0]/2),drop = True)
    K_CO2bio = K.where((K.final_regions< len(K.final_regions.values)*1/2)&
                       (K.final_regions>= len(K.final_regions.values)*1/4)&
                    (K.measurement< K.measurement.shape[0]/2),drop = True)
    K_COfire = K.where((K.final_regions< len(K.final_regions.values)*3/4)&
                       (K.final_regions>= len(K.final_regions.values)*1/2)&
                    (K.measurement>= K.measurement.shape[0]/2),drop = True)
    K_CObio = K.where((K.final_regions>= len(K.final_regions.values)*3/4)&
                    (K.measurement>= K.measurement.shape[0]/2),drop = True)

    # split scaling factors
    pred_CO2fire =predictions_flat.where(predictions_flat.bioclass < Inversion.number_of_reg, 
                                                   drop = True).rename(dict(bioclass = 'final_regions'))
    pred_CO2bio =predictions_flat.where((predictions_flat.bioclass <2*Inversion.number_of_reg)&
                                                  (predictions_flat.bioclass >= Inversion.number_of_reg), 
                                                drop = True).rename(dict(bioclass = 'final_regions'))
    pred_COfire =predictions_flat.where((predictions_flat.bioclass < 3*Inversion.number_of_reg)&
                                                  (predictions_flat.bioclass >= 2*Inversion.number_of_reg), 
                                                drop = True).rename(dict(bioclass = 'final_regions'))
    pred_CObio =predictions_flat.where((predictions_flat.bioclass >= Inversion.number_of_reg*3), 
                                                drop = True).rename(dict(bioclass = 'final_regions'))

    # multiplication
    K_list = [K_CO2fire, K_CO2bio, K_COfire, K_CObio]
    conc = []
    prior_conc = []
    for idx, pred in enumerate([pred_CO2fire, pred_CO2bio, pred_COfire, pred_CObio]):
        if idx == 0 or idx == 1: 
            conc.append(K_list[idx].dot(pred)[:,0]+predictions_CO2['background_inter'])
            prior_conc.append(K_list[idx].dot(xr.ones_like(pred))[:,0]+predictions_CO2['background_inter'])
        else:
            conc.append(K_list[idx].dot(pred)[:,0]+predictions_CO['background_inter'])
            prior_conc.append(K_list[idx].dot(xr.ones_like(pred))[:,0]+predictions_CO['background_inter'])

    return conc, prior_conc, predictions_CO, predictions_CO2





def calc_concentrations(Inversion, pred_fire, pred_bio, flux_fire, flux_bio, molecule_name,alpha, savepath):
    'not useful at the moment, replaced by calc_total_conc_by_multiplication_with_K'


    'for either CO or CO2 not both at the same time'
    datapath_predictions = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/'
    predictions_CO2, predictions_CO = Inversion.select_relevant_times(printing = False)
    if molecule_name == 'CO':
        ds = predictions_CO
        factor = 10**9 
        #predictions = Inversion.predictions.where(Inversion.predictions.bioclass>= Inversion.predictions.bioclass.values.max()/2-1, drop = True)
        footprints_bio = Inversion.K.where((Inversion.K.final_regions>= Inversion.K.final_regions.values.max()*3/4)&
                                                (Inversion.K.measurement>= Inversion.K.measurement.shape[0]/2)&
                                                (Inversion.K.week == Inversion.week), drop = True).rename(dict(final_regions = 'bioclass'))

        footprints_fire = Inversion.K.where((Inversion.K.final_regions>= Inversion.K.final_regions.values.max()/2)&
                                            (Inversion.K.final_regions< Inversion.K.final_regions.values.max()*3/4)&
                                        (Inversion.K.measurement>= Inversion.K.measurement.shape[0]/2)&
                                        (Inversion.K.week == Inversion.week), drop = True).rename(dict(final_regions = 'bioclass'))
        
    elif molecule_name =='CO2': 
        ds = predictions_CO2
        factor = 10**6  
        footprints_fire = Inversion.K.where((Inversion.K.final_regions< Inversion.K.final_regions.values.max()/4)&
                                (Inversion.K.measurement< Inversion.K.measurement.shape[0]/2)&
                                (Inversion.K.week == Inversion.week), drop = True).rename(dict(final_regions = 'bioclass'))
        
        footprints_bio = Inversion.K.where((Inversion.K.final_regions>= Inversion.K.final_regions.values.max()/4)&
                                    (Inversion.K.final_regions< Inversion.K.final_regions.values.max()/2)&
                                (Inversion.K.measurement< Inversion.K.measurement.shape[0]/2)&
                                (Inversion.K.week == Inversion.week), drop = True).rename(dict(final_regions = 'bioclass'))

    conc_sum_bio = multiply_footprints_with_fluxes_for_concentrations(Inversion, flux_bio.squeeze(dim='week').drop('week')*pred_bio, 
                                                                      footprints_bio, factor, ds['background_inter'])
    
    conc_sum_fire = multiply_footprints_with_fluxes_for_concentrations(Inversion, flux_fire.squeeze(dim='week').drop('week')*pred_fire, 
                                                                    footprints_fire, factor, ds['background_inter'])
    conc_sum_bio_prior = multiply_footprints_with_fluxes_for_concentrations(Inversion, flux_bio.squeeze(dim='week').drop('week'), 
                                                                    footprints_bio, factor, ds['background_inter'])
    conc_sum_fire_prior = multiply_footprints_with_fluxes_for_concentrations(Inversion, flux_fire.squeeze(dim='week').drop('week'), 
                                                                    footprints_fire, factor, ds['background_inter'])

    #conc_sum_bio = multiply_footprints_with_fluxes_for_concentrations(Inversion, flux_bio.squeeze(dim='week').drop('week')*pred_bio, 
    #                                                                  Inversion.footprints_eco.where(Inversion.footprints_eco.week == Inversion.week, drop = True), factor, ds['background_inter'])
    
    #conc_sum_fire = multiply_footprints_with_fluxes_for_concentrations(Inversion, flux_fire.squeeze(dim='week').drop('week')*pred_fire, 
    #                                                                 Inversion.footprints_eco.where(Inversion.footprints_eco.week == Inversion.week, drop = True), factor, ds['background_inter'])
    #conc_sum_bio_prior = multiply_footprints_with_fluxes_for_concentrations(Inversion, flux_bio.squeeze(dim='week').drop('week'), 
    #                                                                 Inversion.footprints_eco.where(Inversion.footprints_eco.week == Inversion.week, drop = True), factor, ds['background_inter'])
    #conc_sum_fire_prior = multiply_footprints_with_fluxes_for_concentrations(Inversion, flux_fire.squeeze(dim='week').drop('week'), 
    #                                                                 Inversion.footprints_eco.where(Inversion.footprints_eco.week == Inversion.week, drop = True), factor, ds['background_inter'])





    return conc_sum_fire , conc_sum_bio, conc_sum_fire_prior, conc_sum_bio_prior, ds

def plot_fire_bio_concentrations(conc_tot_fire,conc_tot_bio,prior_fire, prior_bio, ds, savepath, alpha, alphaCO,molecule_name): 
    if molecule_name == 'CO':
        y = 35
        ymin = 0
        ymax = 300
        unit = 'ppb'
    elif molecule_name == 'CO2': 
        y = 407
        ymin = 407
        ymax = 420
        unit = 'ppm'
    else:
        raise Exception('Molecule name not defined, only Co and CO2 allowed')

    df = pd.DataFrame(data =conc_tot_fire.values, columns = ['conc_fire'])
    df.insert(loc = 1, column ='conc_bio', value = conc_tot_bio.values)
    df.insert(loc = 1, column ='prior_bio', value = prior_bio.values)
    df.insert(loc = 1, column ='prior_fire', value = prior_fire.values)
    df.insert(loc = 1, column = 'time', value = ds['time'])
    df.insert(loc = 1, column = 'measurement_uncertainty', value = ds['measurement_uncertainty'])
    df.insert(loc = 1, column = 'xco2_measurement', value = ds['xco2_measurement'])

    plt.rcParams.update({'font.size':18})   
    plt.rcParams.update({'errorbar.capsize': 5})
    fig, (ax, ax2) = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'height_ratios': [4, 1]},figsize = (12,8))

    ds.plot(x='time', y='background_inter',color = 'k', marker='.', markersize = 7,linestyle='None', ax=ax, label= 'Model background')
    ds.plot(x='time', y='xco2_measurement',marker='.',ax=ax, markersize = 7,linestyle='None',color = 'dimgrey',label= 'Measurement')#yerr = 'measurement_uncertainty', 
    df.plot(x = 'time', y = 'prior_fire',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'salmon',label = 'Prior fire')
    df.plot(x = 'time', y = 'conc_bio',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'skyblue',label = 'Posterior bio & fossil')
    df.plot(x = 'time', y = 'conc_fire',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'red',label = 'Posterior fire')
 
    ax.legend(markerscale = 2)
    ax.set_xticks([datetime.datetime(year =2019, month = 12, day = 1),datetime.datetime(year =2019, month = 12, day = 5), datetime.datetime(year =2019, month = 12, day = 10), datetime.datetime(year =2019, month = 12, day = 15),
                datetime.datetime(year =2019, month = 12, day = 20), datetime.datetime(year =2019, month = 12, day = 25), datetime.datetime(year =2019, month = 12, day = 30), 
                ], 
                rotation=45)#datetime(year = 2020, month = 1, day = 4)
    ax.set_xlim(left =  datetime.datetime(year = 2019, month = 11, day=30), right = datetime.datetime(year = 2019, month = 12, day=30, hour = 15))
    ax.grid(axis = 'both')
    ax.set_ylim((ymin,ymax))
    ax.set_ylabel('concentration ['+unit+']', labelpad=6)
    ax.errorbar(x= datetime.datetime(year =2019, month = 12, day = 31, hour = 4), y = y, yerr = ds['measurement_uncertainty'].mean(), marker = '.',markersize = 7,linestyle='None',color = 'dimgrey')
    plt.xlabel('date')

    myFmt = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(myFmt)

    ## Rotate date labels automatically
    fig.autofmt_xdate()
    ds_mean = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/preparation/one_hour_runs/TCCON_mean_measurements_sept_to_march.pkl')
    ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =12, day=1))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =1, day=9))]
    ax2.bar(ds_mean['datetime'], ds_mean['number_of_measurements'], width=0.1, color = 'dimgrey')
    ax2.set_ylabel('# measurements', labelpad=17)
    ax2.grid(axis='x')
  
    plt.subplots_adjust(hspace=0)
    plt.savefig(savepath+str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_'+molecule_name+'_concentrations_results.png', dpi = 300, bbox_inches = 'tight')


def plot_total_concentrations(conc_tot, prior_tot, ds, savepath, alpha, alphaCO,molecule_name): 
    if molecule_name == 'CO':
        y = 35
        unit = 'ppb'
    elif molecule_name == 'CO2': 
        y = 407
        unit = 'ppm'
    else:
        raise Exception('Molecule name not defined, only Co and CO2 allowed')
    

    ############create pandas Dataframe for plotting convenience #######
    df = pd.DataFrame(data =conc_tot.values, columns = ['conc'])
    df.insert(loc = 1, column ='prior', value = prior_tot.values)
    df.insert(loc = 1, column = 'time', value = ds['time'])
    df.insert(loc = 1, column = 'measurement_uncertainty', value = ds['measurement_uncertainty'])
    df.insert(loc = 1, column = 'xco2_measurement', value = ds['xco2_measurement'])

    plt.rcParams.update({'font.size':18})   
    plt.rcParams.update({'errorbar.capsize': 5})
    fig, (ax, ax2) = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'height_ratios': [4, 1]},figsize = (12,8))

    ds.plot(x='time', y='background_inter',color = 'k', marker='.', markersize = 7,linestyle='None', ax=ax, label= 'Model background')
    ds.plot(x='time', y='xco2_measurement',marker='.',ax=ax, markersize = 7,linestyle='None',color = 'dimgrey',label= 'Measurement')#yerr = 'measurement_uncertainty', 
    df.plot(x = 'time', y = 'prior',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'salmon',label = 'Total model prior')
    df.plot(x = 'time', y = 'conc',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'red',label = 'Total model posterior')
    
    ax.legend(markerscale = 2)
    ax.set_xticks([datetime.datetime(year =2019, month = 12, day = 1),datetime.datetime(year =2019, month = 12, day = 5), datetime.datetime(year =2019, month = 12, day = 10), datetime.datetime(year =2019, month = 12, day = 15),
                datetime.datetime(year =2019, month = 12, day = 20), datetime.datetime(year =2019, month = 12, day = 25), datetime.datetime(year =2019, month = 12, day = 30), 
                ], 
                rotation=45)#datetime(year = 2020, month = 1, day = 4)
    ax.set_xlim(left =  datetime.datetime(year = 2019, month = 11, day=30), right = datetime.datetime(year = 2019, month = 12, day=31, hour = 15))
    ax.grid(axis = 'both')
    ax.set_ylabel('concentration ['+unit+']', labelpad=6)
    ax.errorbar(x= datetime.datetime(year =2019, month = 12, day = 31, hour = 4), y = y, yerr = ds['measurement_uncertainty'].mean(), marker = '.',markersize = 7,linestyle='None',color = 'dimgrey')
    plt.xlabel('date')

    myFmt = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(myFmt)

    ## Rotate date labels automatically
    fig.autofmt_xdate()
    ds_mean = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/preparation/one_hour_runs/TCCON_mean_measurements_sept_to_march.pkl')
    ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =12, day=1))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =1, day=9))]
    ax2.bar(ds_mean['datetime'], ds_mean['number_of_measurements'], width=0.1, color = 'dimgrey')
    ax2.set_ylabel('# measurements', labelpad=17)
    ax2.grid(axis='x')
  
    plt.subplots_adjust(hspace=0)
    plt.savefig(savepath+str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_'+molecule_name+'_total_concentrations_results_total_K_times_scaling.png', dpi = 300, bbox_inches = 'tight')




def plot_single_concentrations(Inversion, alpha, alphaCO,savepath): 

    #conc_tot_fire, conc_tot_bio, prior_fire, prior_bio, ds = calc_concentrations(Inversion, pred_fire, pred_bio, flux_fire, flux_bio,molecule_name,alpha, savepath)
 
    conc, prior_conc, dsCO, dsCO2 = calc_single_conc_by_multiplication_with_K(Inversion, Inversion.predictions_flat)

    plot_fire_bio_concentrations(conc[0],conc[1], prior_conc[0], prior_conc[1], dsCO2, savepath, alpha, alphaCO, 'CO2')
    plot_fire_bio_concentrations(conc[2],conc[3], prior_conc[2], prior_conc[3], dsCO, savepath, alpha, alphaCO, 'CO')
   
    return conc[0]+conc[1]-dsCO2['background_inter'], conc[2]+conc[3]-dsCO['background_inter']

def plot_single_total_concentrations(Inversion, alpha,alphaCO, savepath):
    '''
    conc_tot_fire, conc_tot_bio, prior_fire, prior_bio, ds = calc_concentrations(Inversion, pred_fire, pred_bio, flux_fire, flux_bio,molecule_name,alpha, savepath)
    
    conc_tot = conc_tot_fire - ds['background_inter'] + conc_tot_bio
    prior_tot = prior_fire - ds['background_inter'] + prior_bio
   '''
    conc_totCO2, conc_totCO, priorCO2, priorCO, dsCO, dsCO2 = calc_total_conc_by_multiplication_with_K(Inversion, Inversion.predictions_flat)
    


    plot_total_concentrations(conc_totCO2, priorCO2, dsCO2, savepath, alpha, alphaCO, 'CO2')
    plot_total_concentrations(conc_totCO, priorCO, dsCO, savepath, alpha, alphaCO, 'CO')

    return conc_totCO2, conc_totCO


def plot_difference_of_posterior_concentrations_to_measurements(Inversion, savepath, alpha, alphaCO):

    plt.rcParams.update({'font.size' : 19 })

    conc_totCO2, conc_totCO, priorCO2, priorCO, dsCO, dsCO2 = calc_total_conc_by_multiplication_with_K(Inversion, Inversion.predictions_flat)
    
    diffCO2 = conc_totCO2.values - dsCO2['xco2_measurement']
    diffCO = conc_totCO.values - dsCO['xco2_measurement']

    df = pd.DataFrame(data = dsCO2['time'], columns = ['time'])

    df.insert(loc= 1, value = diffCO2.values, column = 'diffCO2')
    df.insert(loc= 1, column = 'CO2_meas', value = dsCO2['xco2_measurement'])
    df.insert(loc= 1, value = np.array(diffCO)[:], column = 'diffCO')

    mask = (df['time']>=Inversion.date_min)&(df['time']<=Inversion.date_min+datetime.timedelta(days=7)) # selbe maske, da CO und CO2 genau gleich sortiert 
    df = df[mask].reset_index()
    df = df.sort_values(['time'], ascending = True).reset_index()
    df.insert(loc = 1, column = 'date', value = df['time'][:].dt.strftime('%Y-%m-%d'))

    # plotting 
    fig, ax1 = plt.subplots(figsize = (17,7))
    Xaxis = np.arange(0,2*len(df['time'][:]),2)
    ax1.set_ylabel(r'$\Delta$CO$_2$ concentration [ppm]')
    ax1.set_xlabel('date')
    #ax1.tick_params(axis = 'y')
    max_value = max(abs(df['diffCO2'].max()), abs(df['diffCO2'].min()))
    ax1.set_ylim((-max_value*1.1, max_value*1.1))
    lns1 = ax1.bar(Xaxis-0.3, df['diffCO2'], width = 0.6, color = 'gray',label = r'$\Delta$CO$_2$')

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\Delta$CO concentration [ppb]')
    max_value = max(abs(df['diffCO'].max()), abs(df['diffCO'].min()))
    ax2.set_ylim((-max_value*1.1, max_value*1.1))
    lns2 = ax2.bar(Xaxis+0.3, df['diffCO'], width = 0.6, color = 'cornflowerblue', alpha = 0.6, label = r'$\Delta$CO')
    #ax2.tick_params(axis = 'y')# not great but ok 
    ax2.set_xticks(Xaxis)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    # ticks
    ticklabels = ['']*len(df['time'][:])
    # Every 4th ticklable shows the month and day
    ticklabels[0] = df['date'][0]
    reference = df['date'][0]
    for i in np.arange(1,len(df['time'][:])):
        if df['date'][i]>reference:
            ticklabels[i] = df['date'][i]
            reference = df['date'][i]

    ax2.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(ticklabels))
    plt.gcf().autofmt_xdate(rotation = 45)
    #ax2.set_xticks(Xaxis, ticklabels)
    #ax2.tick_params(axis = 'x', labelrotation = 30)

    plt.axhline(y=0, color='k', linestyle='-')

    fig.savefig(savepath+"{:.2e}".format(alpha)+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_diff_measurement_total_conc.png", dpi = 300, bbox_inches = 'tight')

    return 

def plot_single_conc_with_errors(df, savepath, alpha, alphaCO): 
    # plotting 
    fig, ax = plt.subplots(2,1,figsize=(18,8))
    Xaxis = np.arange(0,2*len(df['time'][:]),2)
    ax[0].set_ylabel(r'CO$_2$ [ppm]')
    ax[0].set_xlabel('date')
    ax[0].set_xticks(Xaxis, ['']*len(df['time'][:]))
    #ax1.tick_params(axis = 'y')
    max_value = max(abs(df['CO2_fire'].max()), abs(df['CO2_fire'].min()))
    ax[0].set_ylim((408, max_value+2))
    lns1 = ax[0].plot(Xaxis, df['CO2_fire'],color = 'firebrick',  label = r'fire posterior')
    ax[0].fill_between(Xaxis, df['CO2_fire']-df['CO2fire_std'], df['CO2_fire']+df['CO2fire_std'], color = 'firebrick', alpha = 0.2)

    #total_CO2 = df['CO2_fire']+df['CO2_bio']-df['CO2_background']
    #lns1 = ax[0].plot(Xaxis,total_CO2,color = 'gray',  label = r'total posterior')
    #ax[0].fill_between(Xaxis, total_CO2-(df['CO2fire_std']+df['CO2bio_std']), total_CO2+(df['CO2fire_std']+df['CO2bio_std']), color = 'lightgrey', alpha = 0.5)

    lns1 = ax[0].plot(Xaxis, df['CO2_bio'],color = 'green',label = r'bio posterior')
    ax[0].fill_between(Xaxis, df['CO2_bio']-df['CO2bio_std'], df['CO2_bio']+df['CO2bio_std'], color = 'darkseagreen')

    ax[0].plot(Xaxis-0.3, df['CO2_meas'],color = 'black',label = r'measurements')

    ax2 = ax[1]#.twinx()
    ax2.set_ylabel(r'CO [ppb]')
    max_value = max(abs(df['CO_fire'].max()), abs(df['CO_fire'].min()))
    ax2.set_ylim((50, max_value+50))
    lns1 = ax2.plot(Xaxis, df['CO_fire'],color = 'firebrick',label = r'fire posterior')
    ax2.fill_between(Xaxis, df['CO_fire']-df['COfire_std'], df['CO_fire']+df['COfire_std'], color = 'firebrick', alpha = 0.2)

    #total_CO = df['CO_fire']+df['CO_bio']-df['CO_background']
    #lns1 = ax2.plot(Xaxis,total_CO,color = 'gray',  label = r'total posterior')
    #ax2.fill_between(Xaxis, total_CO-(df['COfire_std']+df['CObio_std']), total_CO+(df['COfire_std']+df['CObio_std']), color = 'lightgrey', alpha = 0.5)

    lns1 = plt.plot(Xaxis, df['CO_bio'],color = 'green',label = r'bio posterior')
    ax2.fill_between(Xaxis, df['CO_bio']-df['CObio_std'], df['CO_bio']+df['CObio_std'], color = 'darkseagreen')

    ax2.plot(Xaxis-0.3, df['CO_meas'],color = 'black',label = r'measurements')

    ax[1].legend(loc = 'upper left')
    
    # ticks
    ticklabels = ['']*len(df['time'][:])
    # Every 4th ticklable shows the month and day
    ticklabels[0] = df['date'][0]
    reference = df['date'][0]
    for i in np.arange(1,len(df['time'][:])):
        if df['date'][i]>reference:
            ticklabels[i] = df['date'][i]
            reference = df['date'][i]
    ax[1].set_xticks(Xaxis, ticklabels)
    ax[1].xaxis.set_major_formatter(mpl.ticker.FixedFormatter(ticklabels))
    plt.gcf().autofmt_xdate(rotation = 45)

    plt.axhline(y=0, color='k', linestyle='-')

    fig.savefig(savepath+"{:.2e}".format(alpha)+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_single_concentrations_with_errors_measurement.png", dpi = 300, bbox_inches = 'tight')



def plot_total_conc_with_errors(df, savepath, alpha, alphaCO): 
    # plotting 
    fig, ax = plt.subplots(2,1,figsize=(18,10))
    Xaxis = np.arange(0,2*len(df['time'][:]),2)
    ax[0].set_ylabel(r'CO$_2$ [ppm]')
    ax[0].set_xlabel('date')
    ax[0].set_xticks(Xaxis, ['']*len(df['time'][:]))
    #ax1.tick_params(axis = 'y')
    max_value = max(abs(df['CO2_fire'].max()), abs(df['CO2_fire'].min()))
    ax[0].set_ylim((406, max_value+2))
 
    #total CO2
    total_CO2 = df['CO2_fire']+df['CO2_bio']-df['CO2_background']
    lns1 = ax[0].plot(Xaxis,total_CO2,color = 'red',  label = r'total posterior')
    ax[0].fill_between(Xaxis, total_CO2-(df['CO2fire_std']+df['CO2bio_std']), total_CO2+(df['CO2fire_std']+df['CO2bio_std']), color = 'red', alpha = 0.5)
    #prior CO2
    lns1 = ax[0].plot(Xaxis,df['CO2_prior'],color = 'salmon',  label = r'total prior')
    ax[0].fill_between(Xaxis, df['CO2_prior']-(df['CO2_prior_std']), df['CO2_prior']+(df['CO2_prior_std']), color = 'salmon', alpha = 0.3)

    ax[0].plot(Xaxis-0.3, df['CO2_meas'],color = 'dimgrey',label = r'measurements')

    ## CO plot
    ax2 = ax[1]#.twinx()
    ax2.set_ylabel(r'CO [ppb]')
    max_value = max(abs(df['CO_fire'].max()), abs(df['CO_fire'].min()))
    ax2.set_ylim((0, max_value+100))
    # total CO
    total_CO = df['CO_fire']+df['CO_bio']-df['CO_background']
    lns1 = ax2.plot(Xaxis,total_CO,color = 'red',  label = r'total posterior')
    ax2.fill_between(Xaxis, total_CO-(df['COfire_std']+df['CObio_std']), total_CO+(df['COfire_std']+df['CObio_std']), color = 'red', alpha = 0.5)
    # prior CO 
    #total_prior_CO = df['CO_fire_prior']+df['CO_bio_prior']-df['CO_background']
    lns1 = ax2.plot(Xaxis,df['CO_prior'],color = 'salmon',  label = r'total prior')
    ax2.fill_between(Xaxis, df['CO_prior']-(df['CO_prior_std']), df['CO_prior']+(df['CO_prior_std']), color = 'salmon', alpha = 0.3)


    ax2.plot(Xaxis-0.3, df['CO_meas'],color = 'dimgrey',label = r'measurements')

    ax[1].legend(loc = 'upper left')
    
    # ticks
    ticklabels = ['']*len(df['time'][:])
    # Every 4th ticklable shows the month and day
    ticklabels[0] = df['date'][0]
    reference = df['date'][0]
    for i in np.arange(1,len(df['time'][:])):
        if df['date'][i]>reference:
            ticklabels[i] = df['date'][i]
            reference = df['date'][i]
    ax[1].set_xticks(Xaxis, ticklabels)
    ax[1].xaxis.set_major_formatter(mpl.ticker.FixedFormatter(ticklabels))
    plt.gcf().autofmt_xdate(rotation = 45)

    plt.axhline(y=0, color='k', linestyle='-')

    fig.savefig(savepath+"{:.2e}".format(alpha)+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_total_concentrations_with_errors_measurement.png", dpi = 300, bbox_inches = 'tight')




def plot_single_concentrations_measurements_and_errors(Inversion, savepath, alpha, alphaCO, prior_std):

    plt.rcParams.update({'font.size' : 19 })

    #conc_totCO2, conc_totCO, priorCO2, priorCO, dsCO, dsCO2 = calc_total_conc_by_multiplication_with_K(Inversion, Inversion.)

    conc_err, prior_conc, dsCO, dsCO2 = calc_single_conc_by_multiplication_with_K(Inversion, Inversion.prediction_errs_flat.rename({'new': 'bioclass'}))
    conc, prior_conc, dsCO, dsCO2 = calc_single_conc_by_multiplication_with_K(Inversion, Inversion.predictions_flat)
    
    CO2fire_std = conc_err[0]-dsCO2['background_inter']
    CO2bio_std = conc_err[1]-dsCO2['background_inter']
    COfire_std = conc_err[2]-dsCO['background_inter']
    CObio_std = conc_err[3]-dsCO['background_inter']

    df = pd.DataFrame(data = dsCO2['time'], columns = ['time'])

    df.insert(loc= 1, value = conc[0].values, column = 'CO2_fire')
    df.insert(loc= 1, value = conc[1].values, column = 'CO2_bio')
    df.insert(loc= 1, value = conc[2].values, column = 'CO_fire')
    df.insert(loc= 1, value = conc[3].values, column = 'CO_bio')
    
    df.insert(loc= 1, column = 'CO2_meas', value = dsCO2['xco2_measurement'])
    df.insert(loc= 1, column = 'CO_meas', value = dsCO['xco2_measurement'])
    df.insert(loc=1, column = 'CO2_background', value = dsCO2['background_inter'])
    df.insert(loc=1, column='CO_background', value = dsCO['background_inter'])

    df.insert(loc= 1, value = CO2bio_std.values, column = 'CO2bio_std')
    df.insert(loc= 1, value = CO2fire_std.values, column = 'CO2fire_std')
    df.insert(loc= 1, value = COfire_std.values, column = 'COfire_std')
    df.insert(loc= 1, value = CObio_std.values, column = 'CObio_std')

    concCO2, concCO, priorCO2, priorCO, predictions_CO, predictions_CO2 = calc_total_conc_by_multiplication_with_K(Inversion,prior_std.rename({'new':'bioclass'}))
    # Ne prior std an K multiplizieren 
    df.insert(loc= 1, value = concCO2.values-predictions_CO2['background_inter'][:], column = 'CO2_prior_std')
    df.insert(loc= 1, value = concCO.values-predictions_CO['background_inter'][:], column = 'CO_prior_std')
    df.insert(loc= 1, value = priorCO2.values, column = 'CO2_prior')
    df.insert(loc= 1, value = priorCO.values, column = 'CO_prior')
    df = df[(df['time']<=(Inversion.date_min + datetime.timedelta(days = 6)))]
    df.to_pickle(savepath+"{:.2e}".format(alpha)+"_CO2_"+"{:.2e}".format(alphaCO)+"_concentrations_and_errors_only_one_week.pkl")

    #df.insert(loc= 1, value = prior_conc[0].values, column = 'CO2_fire_prior')
    #df.insert(loc= 1, value = prior_conc[1].values, column = 'CO2_bio_prior')
    #df.insert(loc= 1, value = prior_conc[2].values, column = 'CO_fire_prior')
    #df.insert(loc= 1, value = prior_conc[3].values, column = 'CO_bio_prior')
    
    #df.insert(loc= 1, value = prior_std_list[0].values*flux_list[0].values, column = 'CO2_fire_prior_std')
    #df.insert(loc= 1, value = prior_std_list[1].values*flux_list[1].values, column = 'CO2_bio_prior_std')
    #df.insert(loc= 1, value = prior_std_list[2].values*flux_list[2].values, column = 'CO_fire_prior_std')
    #df.insert(loc= 1, value = prior_std_list[3].values*flux_list[3].values, column = 'CO_bio_prior_std')
    '''
    mask = (df['time']>=Inversion.date_min)&(df['time']<=Inversion.date_min+datetime.timedelta(days=7)) # selbe maske, da CO und CO2 genau gleich sortiert 
    df = df[mask].reset_index()
    df = df.sort_values(['time'], ascending = True).reset_index()
    df.insert(loc = 1, column = 'date', value = df['time'][:].strftime('%Y-%m-%d'))#dt.

    plot_single_conc_with_errors(df, savepath, alpha, alphaCO)
    plot_total_conc_with_errors(df, savepath, alpha, alphaCO)
    '''
    return 





'''
def get_loss_terms(Regression, x):## muss noch angepasst werden 
    loss_regularization = (
        (Regression.x_prior - x)
        @ Regression.x_covariance_inv_sqrt
        @ Regression.x_covariance_inv_sqrt.T
        @ (Regression.x_prior - x)
    )
    loss_least_squares = np.sum((Regression.y - Regression.K @ x) ** 2)
    return loss_regularization, loss_least_squares


def compute_l_curve(Inversion, alpha_list=[0.1, 1.0, 10.0], cond=None):
        """
        Compute the so-called l-curve.

        Parameters
        ----------
        alpha_list : list of float, optional
            List of the regularization parameters, by default None
        cond : float, optional
            Cutoff for 'small' singular values; used to determine effective rank of a.
            Singular values smaller than cond * largest_singular_value are considered
            zero.

        Returns
        -------
        inversion_params : dict
            The resulting output from the inversions. The dictionary contains lists with
            entries for each `alpha` of `alpha_list`:
             - "x_est" : The estimated state vector.
             - "res" : Residues of the loss function.
             - "rank" : Effective rank of the (adapted) forward matrix.
             - "s" : Singular values of the (adapted) forward matrix.
             - "loss_regularization" : The values of the regularization term of the inversion equation.
             - "loss_forward_model" : The values of the measurement term of the inversion equation.

        Examples
        --------
        To plot the l-curve:

        >>> inv_params = regression.compute_l_curve()
        >>> matplotlib.pyplot.plot(
        ...     inv_params["loss_regularization"],
        ...     inv_params["loss_forward_model"],
        ... )

        To get the gain, averaging kernel, and posterior covariance matrix (only works
        for Bayesian inversion):

        >>> posterior_covariance = regression.get_posterior_covariance()
        >>> gain = regression.get_gain()
        >>> averaging_kernel = regression.get_averaging_kernel()

        """
        inversion_params = dict(
            alpha=alpha_list,
            x_est=[],
            res=[],
            rank=[],
            s=[],
            loss_regularization=[],
            loss_forward_model=[],
        )
        for alpha in alpha_list:
            # Update inversion parameters
            y_reg, K_reg = Inversion.model.get_reg_params(alpha)
            x_est, res, rank, s =Inversion.fit(cond=cond)
            inversion_params["x_est"].append(x_est)
            inversion_params["res"].append(res)
            inversion_params["rank"].append(rank)
            inversion_params["s"].append(s)
            loss_regularization, loss_forward_model = get_loss_terms(Inversion.reg,
                x=x_est
            )
            inversion_params["loss_regularization"].append(loss_regularization)
            inversion_params["loss_forward_model"].append(loss_forward_model)
        return inversion_params





'''




def calculate_ratio_error(std_CO2, std_CO, flux_CO2, flux_CO):
    
    err  = np.sqrt((std_CO2/flux_CO)**2+(flux_CO2/((flux_CO)**2)*std_CO)**2)
    return err






def plot_l_curve(Inversion, molecule_name, savepath, alpha, err = None):
    #[1e-8,3e-8, 1e-7, 4.9e-7, 1e-6,2e-6,5e-6, 1e-5,2e-5, 3e-5,5e-5,1e-4, 2e-4, 3e-4, 5e-4, 1e-3,2e-3,4.32e-3, 1e-2,1.8e-2,3e-2,5e-2,7e-2, 1e-1,1.8e-1,3e-1,5e-1, 1]
    # [1e-8,2.5e-8,5e-8,1e-7,2.5e-7,5.92e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1]

    #1e-60,1e-55,1e-52,1e-50,1e-45,1e-40,1e-38, 1e-36, 1e-35,1e-34,1e-33,1e-32,1e-31, 5e-31, 2e-30,1e-29,5e-29, 1e-28,5e-28,1e-27,1e-26,5e-26, 1e-25,5e-25,1e-24, 1e-23,1e-22,1e-21, 1e-20,1e-19,1e-18, 1e-17,
    inv_result = Inversion.reg.compute_l_curve(alpha_list =[1e-8,4e-8,1e-7,4e-7,1e-6,4e-6,1e-5,4e-5, 1e-4,4e-4,1e-3,4e-3,1e-2,4e-2, 1e-1,4e-1, 1,4, 10, 40, 100, 400,1000], cond = 1e-14)
    
    #self.l_curve_result = reg.compute_l_curve(alpha, cond)
    #self.alpha = alpha

    #inv_result = Inversion.compute_l_curve(alpha = [3e-19,1e-18,3e-18,1e-17,3e-17,1e-16,3e-16,1e-15,3e-15,1e-14, 5e-14,1e-13, 5e-13, 1e-12, 1e-11,3e-11,1e-10,5e-10,1e-9,4e-9,1e-8,2.5e-8,5e-8,1e-7,2e-7,
    #                                                5e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1], xerr = err)
    #inv_result = Inversion.compute_l_curve(alpha = [1e-17,5e-17,1e-16,5e-16,1e-15,5e-15,1e-14, 5e-14,1e-13, 5e-13, 1e-12, 5e-12,1e-11,5e-11,1e-10,5e-10,1e-9,4e-9,1e-8,2.5e-8,5e-8,1e-7,2e-7,5e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1], xerr = err)
    plt.figure(figsize=(10,8))
    #plt.scatter(inv_result["loss_regularization"],inv_result["loss_forward_model"])
    #plt.plot(inv_result["loss_regularization"],inv_result["loss_forward_model"])
    plt.scatter(inv_result["loss_forward_model"],inv_result["loss_regularization"], color = 'black')
    plt.scatter(inv_result["loss_forward_model"][10], inv_result["loss_regularization"][10], color = 'firebrick', label = '$\lambda = 4 *10^{-4}$')
    plt.plot(inv_result["loss_forward_model"],inv_result["loss_regularization"], color = 'black')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Regularization loss')
    plt.xlabel('Forward model loss')
    #plt.xlim((0,1e7))
    #plt.ylim((0, 4e-13))
    #fig, ax = Inversion.plot_l_curve(mark_ind = 17, mark_kwargs= dict(color = 'firebrick',label = r'$\lambda =1 \cdot 10^{-3}$'))
    #plt.grid(axis = 'y', color = 'grey', linestyle = '--' )
    plt.legend()
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_err_per_reg_l_curve_xerr_extended2_middle.png')
    plt.close()




def plot_weekly_concentrations(Inversion, molecule_name,alpha, savepath):
    if molecule_name == 'CO':
        unit = 'ppb'
    elif molecule_name == 'CO2': 
        unit = 'ppm'
    else: 
        Exception('Molecule name not defined! Use CO or CO2 only')
    conc_tot, ds = calc_concentrations(Inversion, molecule_name,alpha, savepath)

    plt.rcParams.update({'font.size':14})   
    plt.rcParams.update({'errorbar.capsize': 5})
    #fig, (ax, ax2) = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'height_ratios': [4, 1]},figsize = (14,10))
    fig, ax = plt.subplots(1,1, figsize = (14,10))#plt.figure(figsize = (14,10))
    df = pd.DataFrame(data =conc_tot.values, columns = ['conc'])
    df.insert(loc = 1, column = 'time', value = ds['time'])
    df.insert(loc = 1, column = 'measurement_uncertainty', value = ds['measurement_uncertainty'])
    df.insert(loc = 1, column = 'xco2_measurement', value = ds['xco2_measurement'])

    df48 = df[df['time']<datetime.datetime(year=2019, month = 12, day=2)]
    df49 = df[(df['time']>datetime.datetime(year=2019, month = 12, day=1, hour = 23))&(df['time']<datetime.datetime(year=2019, month = 12, day=9))]
    df50 = df[(df['time']>datetime.datetime(year=2019, month = 12, day=8, hour = 23))&(df['time']<datetime.datetime(year=2019, month = 12, day=16))]
    df51 = df[(df['time']>datetime.datetime(year=2019, month = 12, day=15, hour = 23))&(df['time']<datetime.datetime(year=2019, month = 12, day=23))]
    df52 = df[(df['time']>datetime.datetime(year=2019, month = 12, day=22, hour = 23))&(df['time']<datetime.datetime(year=2019, month = 12, day=30))]
    df1 = df[(df['time']>datetime.datetime(year=2019, month = 12, day=29, hour = 23))&(df['time']<datetime.datetime(year=2020, month = 1, day=6))]

    ds48 = ds[ds['time']<datetime.datetime(year=2019, month = 12, day=2)]
    ds49 = ds[(ds['time']>datetime.datetime(year=2019, month = 12, day=1, hour = 23))&(ds['time']<datetime.datetime(year=2019, month = 12, day=9))]
    ds50 = ds[(ds['time']>datetime.datetime(year=2019, month = 12, day=8, hour = 23))&(ds['time']<datetime.datetime(year=2019, month = 12, day=16))]
    ds51 = ds[(ds['time']>datetime.datetime(year=2019, month = 12, day=15, hour = 23))&(ds['time']<datetime.datetime(year=2019, month = 12, day=23))]
    ds52 = ds[(ds['time']>datetime.datetime(year=2019, month = 12, day=22, hour = 23))&(ds['time']<datetime.datetime(year=2019, month = 12, day=30))]
    ds1 = ds[(ds['time']>datetime.datetime(year=2019, month = 12, day=29, hour = 23))&(ds['time']<datetime.datetime(year=2020, month = 1, day=6))]

    background = []
    measurement = []
    prior = []
    conc = []
    for ds in [ds48, ds49, ds50, ds51, ds52, ds1]: 
        background.append(ds['background_inter'].mean())
        measurement.append(ds['xco2_measurement'].mean())
        prior.append(ds['xco2_inter'].mean())
    for df in [df48, df49, df50, df51, df52, df1]: 
        conc.append(df['conc'].mean())



    weeks = [48, 49, 50, 51, 52, 53]
    plt.plot(weeks, background, marker = 'o', label = 'Background')
    plt.plot(weeks, measurement, marker = 'o', label = 'Measurement')
    plt.plot(weeks, prior,  marker = 'o',label = 'Prior')
    plt.plot(weeks, conc,  marker = 'o',label= 'Posterior')

    #ds.plot(x='time', y='background_inter',color = 'k', marker='.', markersize = 7,linestyle='None', ax=ax, label= 'Model background')
    #ds.plot(x='time', y='xco2_measurement',marker='.',ax=ax, markersize = 7,linestyle='None',color = 'dimgrey',label= 'Measurement')#yerr = 'measurement_uncertainty', 
    #plt.fill_between(df['time'],df['xco2_measurement']-df['measurement_uncertainty'],
    #                 df['xco2_measurement']+df['measurement_uncertainty'],color = 'lightgrey' )
    #df.plot(x = 'time', y = 'conc',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'red',label = 'Model posterior')
    #ds.plot(x='time', y='xco2_inter', marker='.', ax = ax,markersize = 7,linestyle='None',color = 'salmon', label='Model prior')
    plt.legend()
    #plt.xticks([datetime(year =2019, month = 12, day = 1),datetime(year =2019, month = 12, day = 5), datetime(year =2019, month = 12, day = 10), datetime(year =2019, month = 12, day = 15),
    #            datetime(year =2019, month = 12, day = 20), datetime(year =2019, month = 12, day = 25), datetime(year =2019, month = 12, day = 30), 
    #            datetime(year = 2020, month = 1, day = 4)], 
    #            rotation=45)
    plt.xticks([48, 49, 50, 51, 52, 53])
    ax.set_xticklabels(['48', '49', '50', '51', '52', '1'])
    plt.xlim((47.5, 53.5))
    #plt.xlim(left =  datetime(year = 2019, month = 11, day=30), right = datetime(year = 2020, month = 1, day=8))
    plt.grid(axis = 'both')
    plt.ylabel('concentration ['+unit+']')
    #plt.errorbar(x= datetime(year =2020, month = 1, day = 7), y = 407.5, yerr = ds['measurement_uncertainty'].mean(), marker = '.',markersize = 7,linestyle='None',color = 'dimgrey')
    plt.xlabel('weeks')
    plt.title('Weekly mean concentration')
    myFmt = DateFormatter("%Y-%m-%d")
   # plt.xaxis.set_major_formatter(myFmt)
    ## Rotate date labels automatically
    #plt.autofmt_xdate()
    #ax.legend()
    #ax.grid(axis = 'both')
    #ax.set_ylabel('concentration [ppm]')
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_weekly_mean_concentrations_results.png')

    return


def sensitivity_studies_correlation(): 

    area_bioreg = [8.95512935e+13, 4.27021383e+11, 4.01383356e+12, 1.45736998e+12,
    2.15745316e+11, 9.81395187e+09, 3.95111921e+10, 3.18183296e+10,
    5.52121748e+10, 5.55915961e+10, 5.41460509e+10, 4.29887311e+10,
    3.13764949e+10, 1.05335427e+10, 9.31980461e+10, 5.11758810e+10,
    4.00158388e+10, 2.99495636e+10, 3.91250878e+10, 4.90667320e+10,
    1.54614599e+11, 9.65077342e+10, 6.07622615e+10, 6.98377101e+10,
    1.77802351e+11, 1.95969571e+11, 2.26322082e+11]

    # influence of reg param to result: 
    alpha = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    CO_fire = []
    CO2_fire = []
    for a in alpha: 
        df = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/Working_Setup/CO2_200_CO_200/CO2_2_CO_6/Corr_'+str(a)+'/'+str("{:e}".format(0.1))+'_predictions.pkl')
        CO_fire.append(np.array(df[int(108/2):int(3*108/4)]['prior_flux_eco']*df[int(108/2):int(3*108/4)]['predictions']*area_bioreg).sum()*12*10**(-12)*7*24*60*60*10**(-9))
        CO2_fire.append(np.array(df[:int(108/4)]['prior_flux_eco']*df[:int(108/4)]['predictions']*area_bioreg).sum()*12*10**(-12)*7*24*60*60*10**(-6))

    plt.figure(figsize = (8,6))
    plt.grid()
    plt.rcParams.update({'font.size': 19})
    plt.plot(alpha[:], CO2_fire[:], label = r'CO$_2$ fire', marker = 'o',color = 'dimgrey')#'steelblue')#'darkblue')
    plt.plot(alpha[:], CO_fire[:], label = 'CO fire', marker = 'o',color = 'forestgreen')#'teal')

    #plt.xscale("log")
    #plt.yscale("log")
    #plt.ylim((0,400))
    plt.xlabel('Correlation')
    plt.ylabel('total emissions [TgC/week]')
    plt.legend()

    plt.savefig('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/Working_Setup/Sensitivity_tests/sensitivity_corr_regular.png', 
            facecolor = 'w', dpi = 200, bbox_inches = 'tight')
    plt.show()
    return

