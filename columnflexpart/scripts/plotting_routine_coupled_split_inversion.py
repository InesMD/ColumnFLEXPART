
from columnflexpart.classes.coupled_inversion import CoupledInversion 
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
from matplotlib.dates import DateFormatter

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

    return predictionsCO2_fire, predictionsCO2_bio, predictionsCO_fire, predictionsCO_bio

def split_eco_flux(Inversion):
        flux_eco = Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week]
        fluxCO2_fire = flux_eco[:Inversion.number_of_reg].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,Inversion.number_of_reg)) 
        fluxCO2_bio = flux_eco[Inversion.number_of_reg:2*Inversion.number_of_reg].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,Inversion.number_of_reg))
        fluxCO_fire = flux_eco[2*Inversion.number_of_reg: 3*Inversion.number_of_reg].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,Inversion.number_of_reg))
        fluxCO_bio = flux_eco[3*Inversion.number_of_reg:].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,Inversion.number_of_reg))

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
    
def plot_averaging_kernel(Inversion,ak_final, class_num, savepath, savename): 
    ak_xr = xr.DataArray(data = ak_final.diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,Inversion.number_of_reg))))#, week = [1,48,49,50,51,52])))
    ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, class_num)
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())  
    orig_map=plt.cm.get_cmap('gist_heat')
    reversed_map = orig_map.reversed()
    ak_spatial.plot( x='longitude', y='latitude',ax = ax,vmax = 1, vmin = 0,cmap = reversed_map, cbar_kwargs = {'shrink':0.835})#norm = LogNorm(vmin = 1e-3),
    plt.scatter(x = 150.8793,y =-34.4061,color="black")
    ax.coastlines()
    plt.title('Averaging kernel for 2019/12')
    plt.savefig(savepath+savename, dpi = 250, bbox_inches = 'tight')



def calculate_and_plot_averaging_kernel(Inversion, savepath, alpha): 
    akCO2_fire, akCO2_bio, akCO_fire, akCO_bio = calculate_averaging_kernel(Inversion)
    name_list = ['_CO2_fire_', '_CO2_bio_', '_CO_fire_', '_CO_bio_']
    for idx,ak in enumerate([akCO2_fire, akCO2_bio, akCO_fire, akCO_bio]):
        savename = str("{:e}".format(alpha))+'_ak_spatial_'+name_list[idx]+'.png'
        plot_averaging_kernel(Inversion,ak,Inversion.number_of_reg, savepath, savename)

def plot_spatial_result(spatial_result, savepath, savename, cmap, vmax =None, vmin =None, cbar_kwargs = {'shrink':  0.835}):
    '''
    for weekly spatial plotting of fluxes, saves plot 
    spatial_result : xarray to plot with latitude and longitude as coordinates
    molecule_name : Either "CO2" or "CO"
    savepath : path to save output image, must exist
    savename : Name of output image
    '''
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())  
    spatial_result.plot(x = 'longitude', y = 'latitude', ax = ax, cmap =cmap,vmin = vmin, vmax = vmax, 
                            cbar_kwargs = cbar_kwargs)
    
    plt.scatter(x = 150.8793,y =-34.4061,color="black")
    ax.coastlines()
    #plt.title('Week '+str(week))
    plt.savefig(savepath+savename, bbox_inches = 'tight', dpi = 250)
    plt.close()

def plot_prior_spatially(Inversion, flux, name, idx, savepath):
    plt.rcParams.update({'font.size':15})    
    factor = 10**6*12 
    #lux_mean = Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week]*factor
    spatial_flux = Inversion.map_on_grid(flux*factor)
    savename = 'Prior_'+name+'_spatial_week'+str(Inversion.week)+'.png'
    #plot_spatial_result(spatial_flux, savepath,savename, 'seismic', vmax = 0.2, vmin = -0.2, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835}  )

    if idx == 0: 
        plot_spatial_result(spatial_flux, savepath,savename, 'seismic', vmax = 10, vmin = -10, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835}  )
    elif idx == 1:
        plot_spatial_result(spatial_flux, savepath,savename, 'seismic', vmax = 10, vmin = -10, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835}  )
    elif idx == 2:
        plot_spatial_result(spatial_flux, savepath,savename, 'seismic', vmax = 0.1, vmin = -0.1, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835}  )
    elif idx == 3:
        plot_spatial_result(spatial_flux, savepath,savename, 'seismic', vmax = 0.4, vmin = -0.4, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835}  )




def plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions,flux, name, idx, alpha, diff =False):
    factor = 12*10**6
    plt.rcParams.update({'font.size':15})   
    
    spatial_result = Inversion.map_on_grid(predictions)
    if diff== True: 
        diff_flux = (predictions- xr.ones_like(predictions))* flux
        diff_spatial_flux = Inversion.map_on_grid(diff_flux)*factor 
        savename = str("{:e}".format(alpha))+'_'+name+'Diff_to_prior_week_'+str(Inversion.week)+'.png'

        if idx == 2: 
            plot_spatial_result(diff_spatial_flux, savepath, savename, 'seismic',vmax = 20, vmin = -20, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        elif idx == 0:
                plot_spatial_result(diff_spatial_flux, savepath, savename, 'seismic', vmax = 250, vmin = -250,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        elif idx == 1 or idx == 3:
            plot_spatial_result(diff_spatial_flux, savepath, savename, 'seismic',vmax = 0.6, vmin = -0.6, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        #elif idx == 3:
        #    plot_spatial_result(diff_spatial_flux,"CO", savepath, savename, 'seismic', vmax = 10, vmin = -10,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
    
    else: 
        # plot scaling factors
        savename = str("{:e}".format(alpha))+"_Scaling_factor_"+name+str(Inversion.week)+".png"
        if idx == 0 or idx == 1: 
            plot_spatial_result(spatial_result, savepath, savename, 'viridis')
        else:
            plot_spatial_result(spatial_result, savepath, savename, 'viridis')
        
        #plot fluxes
        savename = str("{:e}".format(alpha))+"_Spatial_flux_"+name+str(Inversion.week)+".png"
        spatial_flux = Inversion.map_on_grid(predictions * flux)*factor

        if idx == 0: 
            plot_spatial_result(spatial_flux, savepath, savename, 'seismic',vmax = 250, vmin = -250, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        elif idx == 1:
            plot_spatial_result(spatial_flux, savepath, savename, 'seismic', vmax = 10, vmin = -10,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        elif idx == 2:
            plot_spatial_result(spatial_flux, savepath, savename, 'seismic',vmax = 10, vmin = -10, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
        elif idx == 3:
            plot_spatial_result(spatial_flux, savepath, savename, 'seismic', vmax = 0.1, vmin = -0.1,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})

    #total_spatial_result.to_netcdf(path =savepath+'spatial_results_week_'+str(week_min)+'_'+str(week_max)+'.pkl')

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
    plt.figure(figsize=(14, 10))    
    plt.rcParams.update({'font.size':25})    
    ax = plt.axes(projection=ccrs.PlateCarree())   
    ds['bioclass'].plot(x = 'Long', y = 'Lat',add_colorbar = True)
    ax.set_xlim((110.0, 155.0))
    ax.set_ylim((-45.0, -10.0))
    ax.coastlines()
    plt.savefig(savepath+'bioclasses.png')

    return len(set(ds.bioclass.values.flatten()))


def multiply_footprints_with_fluxes_for_concentrations(Inversion, flux, footprints, factor, background):
    print(footprints)
    footprints_flat = footprints.stack(new=[Inversion.time_coord, *Inversion.spatial_valriables])
    print(footprints_flat)
    print(flux)
    concentration_results = footprints_flat.values*factor*flux.values
    print('concentration results')
    print(concentration_results)
    print(concentration_results.shape)
    conc_sum = concentration_results.sum(axis = 1)
    print(conc_sum)
    conc_tot = conc_sum + background

    return conc_tot
    

def calc_concentrations(Inversion, pred_fire, pred_bio, flux_fire, flux_bio, molecule_name,alpha, savepath):
    'for either CO or CO2 not both at the same time'
    datapath_predictions = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/'
    predictions_CO2, predictions_CO = Inversion.select_relevant_times()
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

    return conc_sum_fire , conc_sum_bio, conc_sum_fire_prior, conc_sum_bio_prior, ds

def plot_fire_bio_concentrations(df, ds, savepath, alpha, molecule_name): 
    if molecule_name == 'CO':
        y = 35
    elif molecule_name == 'CO2': 
        y = 407
    else:
        raise Exception('Molecule name not defined, only Co and CO2 allowed')

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
    ax.set_xlim(left =  datetime.datetime(year = 2019, month = 11, day=30), right = datetime.datetime(year = 2019, month = 12, day=31, hour = 15))
    ax.grid(axis = 'both')
    ax.set_ylabel('concentration [ppm]', labelpad=6)
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
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_concentrations_results.png', dpi = 300, bbox_inches = 'tight')


def plot_total_concentrations(df, ds, savepath, alpha, molecule_name): 
    if molecule_name == 'CO':
        y = 35
    elif molecule_name == 'CO2': 
        y = 407
    else:
        raise Exception('Molecule name not defined, only Co and CO2 allowed')


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
    ax.set_ylabel('concentration [ppm]', labelpad=6)
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
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_total_concentrations_results.png', dpi = 300, bbox_inches = 'tight')




def plot_single_concentrations(Inversion, pred_fire, pred_bio,flux_fire, flux_bio, molecule_name, alpha, savepath): 

    conc_tot_fire, conc_tot_bio, prior_fire, prior_bio, ds = calc_concentrations(Inversion, pred_fire, pred_bio, flux_fire, flux_bio,molecule_name,alpha, savepath)
 
    df = pd.DataFrame(data =conc_tot_fire.values, columns = ['conc_fire'])
    df.insert(loc = 1, column ='conc_bio', value = conc_tot_bio.values)
    df.insert(loc = 1, column ='prior_bio', value = prior_bio.values)
    df.insert(loc = 1, column ='prior_fire', value = prior_fire.values)
    df.insert(loc = 1, column = 'time', value = ds['time'])
    df.insert(loc = 1, column = 'measurement_uncertainty', value = ds['measurement_uncertainty'])
    df.insert(loc = 1, column = 'xco2_measurement', value = ds['xco2_measurement'])

    plot_fire_bio_concentrations(df, ds, savepath, alpha, molecule_name)
   
    return

def plot_single_total_concentrations(Inversion, pred_fire, pred_bio,flux_fire, flux_bio, molecule_name, alpha, savepath):

    conc_tot_fire, conc_tot_bio, prior_fire, prior_bio, ds = calc_concentrations(Inversion, pred_fire, pred_bio, flux_fire, flux_bio,molecule_name,alpha, savepath)
 
    conc_tot = conc_tot_fire - ds['background_inter'] + conc_tot_bio
    #print('fire prior')
    #print(prior_fire)
    #print('prior_bio')
    #print(prior_bio)
    #print('background')
    #print(ds['background'])
    #print('prior_tot')

    prior_tot = prior_fire - ds['background_inter'] + prior_bio
    print(prior_tot)
    df = pd.DataFrame(data =conc_tot.values, columns = ['conc'])
    df.insert(loc = 1, column ='prior', value = prior_tot.values)
    df.insert(loc = 1, column = 'time', value = ds['time'])
    df.insert(loc = 1, column = 'measurement_uncertainty', value = ds['measurement_uncertainty'])
    df.insert(loc = 1, column = 'xco2_measurement', value = ds['xco2_measurement'])

    plot_total_concentrations(df, ds, savepath, alpha, molecule_name)
   
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











def plot_l_curve(Inversion, molecule_name, savepath, alpha, err = None):
    #[1e-8,3e-8, 1e-7, 4.9e-7, 1e-6,2e-6,5e-6, 1e-5,2e-5, 3e-5,5e-5,1e-4, 2e-4, 3e-4, 5e-4, 1e-3,2e-3,4.32e-3, 1e-2,1.8e-2,3e-2,5e-2,7e-2, 1e-1,1.8e-1,3e-1,5e-1, 1]
    # [1e-8,2.5e-8,5e-8,1e-7,2.5e-7,5.92e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1]

    #1e-60,1e-55,1e-52,1e-50,1e-45,1e-40,1e-38, 1e-36, 1e-35,1e-34,1e-33,1e-32,1e-31, 5e-31, 2e-30,1e-29,5e-29, 1e-28,5e-28,1e-27,1e-26,5e-26, 1e-25,5e-25,1e-24, 1e-23,1e-22,1e-21, 1e-20,1e-19,1e-18, 1e-17,
    print('compute l curve')
    #print(Inversion.reg.compute_l_curve())
    inv_result = Inversion.reg.compute_l_curve(alpha_list =[1e-8,4e-8,1e-7,4e-7,1e-6,4e-6,1e-5,4e-5, 1e-4,4e-4,1e-3,4e-3,1e-2,4e-2, 1e-1,4e-1, 1,4, 10], cond = 1e-14)
    
    #self.l_curve_result = reg.compute_l_curve(alpha, cond)
    #self.alpha = alpha
    
    print(len(inv_result['loss_regularization']))
    #inv_result = Inversion.compute_l_curve(alpha = [3e-19,1e-18,3e-18,1e-17,3e-17,1e-16,3e-16,1e-15,3e-15,1e-14, 5e-14,1e-13, 5e-13, 1e-12, 1e-11,3e-11,1e-10,5e-10,1e-9,4e-9,1e-8,2.5e-8,5e-8,1e-7,2e-7,
    #                                                5e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1], xerr = err)
    #inv_result = Inversion.compute_l_curve(alpha = [1e-17,5e-17,1e-16,5e-16,1e-15,5e-15,1e-14, 5e-14,1e-13, 5e-13, 1e-12, 5e-12,1e-11,5e-11,1e-10,5e-10,1e-9,4e-9,1e-8,2.5e-8,5e-8,1e-7,2e-7,5e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1], xerr = err)
    print('Plotting')
    plt.figure(figsize=(10,8))
    #print(inv_result)
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
    print('saving')
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_err_per_reg_l_curve_xerr_extended2_middle.png')




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
    #print(background)
    #print(prior)


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

