
from columnflexpart.classes.coupled_inversion import CoupledInversion 
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
from matplotlib.dates import DateFormatter


def plot_spatial_result(spatial_result, molecule_name, savepath, savename, cmap, vmax =None, vmin =None, cbar_kwargs = {'shrink':  0.835}):
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

def plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, week_min, week_max,alpha, diff =False):
    plt.rcParams.update({'font.size':15})   
    predictionsCO2_fire = Inversion.predictions_flat[Inversion.predictions_flat['bioclass']<Inversion.predictions_flat['bioclass'].shape[0]/4]
    predictionsCO2_bio = Inversion.predictions_flat[(Inversion.predictions_flat['bioclass']<Inversion.predictions_flat['bioclass'].shape[0]/2)
                                                &(Inversion.predictions_flat['bioclass']>=Inversion.predictions_flat['bioclass'].shape[0]/4)]
    predictionsCO_fire = Inversion.predictions_flat[(Inversion.predictions_flat['bioclass']>=Inversion.predictions_flat['bioclass'].shape[0]/2)
                                               &(Inversion.predictions_flat['bioclass']<Inversion.predictions_flat['bioclass'].shape[0]*3/4)]
    predictionsCO_bio = Inversion.predictions_flat[(Inversion.predictions_flat['bioclass']>=Inversion.predictions_flat['bioclass'].shape[0]*3/4)]

    predictionsCO2_bio = predictionsCO2_bio.assign_coords(bioclass = np.arange(0,len(predictionsCO2_fire['bioclass'])))
    predictionsCO_fire = predictionsCO_fire.assign_coords(bioclass = np.arange(0,len(predictionsCO2_fire['bioclass'])))
    predictionsCO_bio = predictionsCO_bio.assign_coords(bioclass = np.arange(0,len(predictionsCO2_fire['bioclass'])))
    
    #print(len(predictionsCO2_fire['bioclass']))
    #print(Inversion.flux_eco)
    flux_eco = Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week]
    fluxCO2_fire = flux_eco[:len(predictionsCO2_fire['bioclass'])].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,len(predictionsCO2_fire['bioclass']))) 
    fluxCO2_bio = flux_eco[len(predictionsCO2_fire['bioclass']):2*len(predictionsCO2_fire['bioclass'])].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,len(predictionsCO2_fire['bioclass'])))
    fluxCO_fire = flux_eco[2*len(predictionsCO2_fire['bioclass']): 3*len(predictionsCO2_fire['bioclass'])].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,len(predictionsCO2_fire['bioclass'])))
    fluxCO_bio = flux_eco[3*len(predictionsCO2_fire['bioclass']):].rename(dict(final_regions = 'bioclass')).assign_coords(bioclass = np.arange(0,len(predictionsCO2_fire['bioclass'])))

    flux_list = [fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio]
    name_list = ['CO2_fire_', 'CO2_ant_bio_', 'CO_fire_', 'CO_ant_bio_']
    #for week in range(week_min,week_max+1):  
    for idx,predictions in enumerate([predictionsCO2_fire,predictionsCO2_bio,predictionsCO_fire,predictionsCO_bio]):
        spatial_result = Inversion.map_on_grid(predictions)
        if diff== True: 
            diff_flux = (predictions- xr.ones_like(predictions))* flux_list[idx] 
            diff_spatial_flux = Inversion.map_on_grid(diff_flux)
            savename = str("{:e}".format(alpha))+'_'+name_list[idx]+'Diff_to_prior_week_'+str(Inversion.week)+'.png'

            if idx == 2: 
                plot_spatial_result(diff_spatial_flux,"CO2", savepath, savename, 'seismic',vmax = 20, vmin = -20, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
            elif idx == 0:
                 plot_spatial_result(diff_spatial_flux,"CO2", savepath, savename, 'seismic', vmax = 100, vmin = -100,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
            elif idx == 1 or idx == 3:
                plot_spatial_result(diff_spatial_flux,"CO", savepath, savename, 'seismic',vmax = 0.6, vmin = -0.6, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
            #elif idx == 3:
            #    plot_spatial_result(diff_spatial_flux,"CO", savepath, savename, 'seismic', vmax = 10, vmin = -10,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})

     
        else: 
            # plot scaling factors
            savename = str("{:e}".format(alpha))+"_Scaling_factor_"+name_list[idx]+str(Inversion.week)+".png"
            if idx == 0 or idx == 1: 
                plot_spatial_result(spatial_result,"CO", savepath, savename, 'viridis')
            else:
                plot_spatial_result(spatial_result,"CO2", savepath, savename, 'viridis')
            
            #plot fluxes
            savename = str("{:e}".format(alpha))+"_Spatial_flux_"+name_list[idx]+str(Inversion.week)+".png"
            spatial_flux = Inversion.map_on_grid(predictions * flux_list[idx])

            if idx == 0: 
                plot_spatial_result(spatial_flux,"CO2", savepath, savename, 'seismic',vmax = 100, vmin = -100, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
            elif idx == 1:
                 plot_spatial_result(spatial_flux,"CO2", savepath, savename, 'seismic', vmax = 10, vmin = -10,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
            elif idx == 2:
                plot_spatial_result(spatial_flux,"CO", savepath, savename, 'seismic',vmax = 10, vmin = -10, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
            elif idx == 3:
                plot_spatial_result(spatial_flux,"CO", savepath, savename, 'seismic', vmax = 10, vmin = -10,cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})

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



def calc_concentrations(Inversion, molecule_name,alpha, savepath):
    'for either CO or CO2 not both at the same time'
    datapath_predictions = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/'
    predictions_CO2, predictions_CO = Inversion.select_relevant_times()
    if molecule_name == 'CO':
        ds = predictions_CO
        factor = 10**9 
        predictions = Inversion.predictions.where(Inversion.predictions.bioclass>= Inversion.predictions.bioclass.values.max()/2-1, drop = True)
        footprints = Inversion.footprints.where((Inversion.footprints.bioclass>= Inversion.footprints.bioclass.values.max()/2-1)&
                                                (Inversion.footprints.measurement>= Inversion.footprints.measurement.shape[0]/2), drop = True)
    elif molecule_name =='CO2': 
        ds = predictions_CO2
        factor = 10**6  
        predictions = Inversion.predictions.where(Inversion.predictions.bioclass< Inversion.predictions.bioclass.values.max()/2-1, drop = True)
        footprints= Inversion.footprints.where((Inversion.footprints.bioclass< Inversion.footprints.bioclass.values.max()/2-1)&
                                               (Inversion.footprints.measurement< Inversion.footprints.measurement.shape[0]/2), drop = True)
        
    footprints_flat = footprints.stack(new=[Inversion.time_coord, *Inversion.spatial_valriables])
    predictions_flat = predictions.stack(new=[Inversion.time_coord, *Inversion.spatial_valriables])
    concentration_results = footprints_flat.values*predictions_flat.values*factor
    conc_sum = concentration_results.sum(axis = 1)
    conc_tot = conc_sum + ds['background_inter']
    return conc_tot , ds

def plot_single_concentrations(Inversion, molecule_name,alpha, savepath): 
    if molecule_name == 'CO':
        y = 35
    elif molecule_name == 'CO2': 
        y = 407
    else: 
        Exception('Molecule name not defined! Use CO or CO2 only')
    conc_tot, ds = calc_concentrations(Inversion, molecule_name,alpha, savepath)

    plt.rcParams.update({'font.size':18})   
    plt.rcParams.update({'errorbar.capsize': 5})
    fig, (ax, ax2) = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'height_ratios': [4, 1]},figsize = (12,8))
    df = pd.DataFrame(data =conc_tot.values, columns = ['conc'])
    df.insert(loc = 1, column = 'time', value = ds['time'])
    df.insert(loc = 1, column = 'measurement_uncertainty', value = ds['measurement_uncertainty'])
    df.insert(loc = 1, column = 'xco2_measurement', value = ds['xco2_measurement'])
    #print(df.time.values)
   
    ds.plot(x='time', y='background_inter',color = 'k', marker='.', markersize = 7,linestyle='None', ax=ax, label= 'Model background')
    ds.plot(x='time', y='xco2_measurement',marker='.',ax=ax, markersize = 7,linestyle='None',color = 'dimgrey',label= 'Measurement')#yerr = 'measurement_uncertainty', 
    #plt.fill_between(df['time'],df['xco2_measurement']-df['measurement_uncertainty'],
    #                 df['xco2_measurement']+df['measurement_uncertainty'],color = 'lightgrey' )
    df.plot(x = 'time', y = 'conc',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'red',label = 'Model posterior')
    ds.plot(x='time', y='xco2_inter', marker='.', ax = ax,markersize = 7,linestyle='None',color = 'salmon', label='Model prior')
    ax.legend()
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
    #ax.legend()
    #ax.grid(axis = 'both')
    #ax.set_ylabel('concentration [ppm]')
    ds_mean = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/preparation/one_hour_runs/TCCON_mean_measurements_sept_to_march.pkl')
    ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =12, day=1))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =1, day=9))]
    ax2.bar(ds_mean['datetime'], ds_mean['number_of_measurements'], width=0.1, color = 'dimgrey')
    ax2.set_ylabel('# measurements', labelpad=17)
    ax2.grid(axis='x')
    #ax.set_title('CO', fontsize = 30)
    plt.subplots_adjust(hspace=0)
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_concentrations_results.png', dpi = 300, bbox_inches = 'tight')

    return


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

