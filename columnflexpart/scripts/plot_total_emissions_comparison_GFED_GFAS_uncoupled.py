from columnflexpart.scripts.functions import getAreaOfGrid
import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
def cut_Transcom_region(ds):
    df =ds.to_dataframe(name = 'CO2').reset_index()
    #print(df)
    gdf = gpd.GeoDataFrame(
        df.CO2, geometry=gpd.points_from_xy(df.longitude,df.latitude))
    Transcom = pd.read_pickle("/home/b/b382105/Transcom_Regions.pkl")
    igdf = gdf.within(
        Transcom[(Transcom.transcom == 'AU')].geometry.iloc[0]
    )
    gdf = gdf.loc[igdf]
    return gdf

def get_CO2_masked_total_emissions(datapath_inversion, datapathmasks,name_mask_before_week,  spatial_results_name_before_week, savename):
        # for CO2 masked save

    dfArea = getAreaOfGrid()
    sums = []
    masked_area = []

    ds = xr.open_dataarray(datapath_inversion+'48/'+spatial_results_name_before_week+'48.nc')#str("{:e}".format(alpha))+'_CO2_spatial_results_week_48.nc')
    mask = xr.open_dataset(datapathmasks+name_mask_before_week)#+'48.nc')#'3.00e-02_CO2_3.00e-02_CO_mask_ak_threshold_0.001and_co_5e-1_week_48.nc')#'mask_co_co2_for_GFED_GFAS_week_48.nc')
    mask = mask.rename({'Lat': 'latitude', 'Long': 'longitude', 'bioclass': '__xarray_dataarray_variable__'})
    Area_lat_long = np.zeros((len(dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]),len(mask.longitude.values) ))
    for i in range(len(mask.longitude.values)): 
        Area_lat_long[:,i] = dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]
    Area_array = xr.Dataset(data_vars = dict(Area = (['latitude', 'longitude'], Area_lat_long)), 
                                            coords= dict(latitude = (["latitude"],dfArea['Lat'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:] ), longitude = (['longitude'], mask.longitude.values)))

    mask2 = xr.merge([mask,Area_array])
    mask2 = mask2.where(mask2.__xarray_dataarray_variable__== 1, drop = True)
    masked_Area = mask2.Area.sum()
    masked_area.append(masked_Area)

    ds = ds*mask.__xarray_dataarray_variable__
    gdf = cut_Transcom_region(ds)

    #print(gdf)
    #gC/m^2/s für eine Woche 
    sums.append( gdf['CO2'].mean()*10**(-6)*masked_Area*10**(-12)*(2*24*60*60))

    for week in range(49,53): 
        ds = xr.open_dataarray(datapath_inversion+str(week)+'/'+spatial_results_name_before_week+str(week)+'.nc')
  
        Area_lat_long = np.zeros((len(dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]),len(mask.longitude.values) ))
        for i in range(len(mask.longitude.values)): 
            Area_lat_long[:,i] = dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]
        Area_array = xr.Dataset(data_vars = dict(Area = (['latitude', 'longitude'], Area_lat_long)), 
                                                coords= dict(latitude = (["latitude"],dfArea['Lat'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:] ), longitude = (['longitude'], mask.longitude.values)))
        mask2 = xr.merge([mask,Area_array])
        mask2 = mask2.where(mask2.__xarray_dataarray_variable__== 1, drop = True)
        masked_Area = mask2.Area.sum()
        masked_area.append(masked_Area)
        mask = xr.open_dataset(datapathmasks+name_mask_before_week)#+str(week)+'.nc')
        mask = mask.rename({'Lat': 'latitude', 'Long': 'longitude', 'bioclass': '__xarray_dataarray_variable__'})
        
        ds = ds*mask.__xarray_dataarray_variable__
        gdf = cut_Transcom_region(ds)
        #gC/m^2/s für eine Woche 
        sums.append( gdf['CO2'].mean()*10**(-6)*masked_Area*10**(-12)*(7*24*60*60))

    ds = xr.open_dataarray(datapath_inversion+str(1)+'/'+spatial_results_name_before_week+'1.nc')
    mask = xr.open_dataset(datapathmasks+name_mask_before_week)#+'1.nc')
    mask = mask.rename({'Lat': 'latitude', 'Long': 'longitude', 'bioclass': '__xarray_dataarray_variable__'})
  
    Area_lat_long = np.zeros((len(dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]),len(mask.longitude.values) ))
    for i in range(len(mask.longitude.values)): 
        Area_lat_long[:,i] = dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]
    Area_array = xr.Dataset(data_vars = dict(Area = (['latitude', 'longitude'], Area_lat_long)), 
                                            coords= dict(latitude = (["latitude"],dfArea['Lat'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:] ), longitude = (['longitude'], mask.longitude.values)))
    mask2 = xr.merge([mask,Area_array])
    mask2 = mask2.where(mask2.__xarray_dataarray_variable__== 1, drop = True)
    masked_Area = mask2.Area.sum()
    masked_area.append(masked_Area)

    #gC/m^2/s für eine Woche 
    gdf = cut_Transcom_region(ds)
    sums.append( gdf['CO2'].mean()*10**(-6)*masked_Area*10**(-12)*(1*24*60*60))
    data = xr.DataArray(data = sums, coords = {'week': [48,49,50,51,52,1]})
    print(data)
    print(masked_area)
    data.to_netcdf(datapath_inversion+savename)#str("{:e}".format(alpha))+'_total_emission_masked_with_area_weekly.nc')
    return masked_area

def get_CO_masked_total_emissions(datapath_inversion, datapathmasks, name_mask_before_week, spatial_result_name_before_week, savename):
        # for CO masked save
    #datapath_inversion = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Ecosystems_AK_split_with_4/'
    #datapathmasks = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Ecoregions_split_4/flat_error_area_scaled/total_emissions/'
    dfArea = getAreaOfGrid()
    sums = []
    masked_area = []
    #alpha =3e-2
    ds = xr.open_dataarray(datapath_inversion+'48/'+spatial_result_name_before_week+'48.nc')
    #print(ds)
    mask = xr.open_dataset(datapathmasks+name_mask_before_week)#+'48.nc')
    mask = mask.rename({'Lat': 'latitude', 'Long': 'longitude', 'bioclass': '__xarray_dataarray_variable__'})
    Area_lat_long = np.zeros((len(dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]),len(mask.longitude.values) ))
    for i in range(len(mask.longitude.values)): 
        Area_lat_long[:,i] = dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]
    Area_array = xr.Dataset(data_vars = dict(Area = (['latitude', 'longitude'], Area_lat_long)), 
                                            coords= dict(latitude = (["latitude"],dfArea['Lat'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:] ), longitude = (['longitude'], mask.longitude.values)))
    mask2 = xr.merge([mask,Area_array])
    mask2 = mask2.where(mask2.__xarray_dataarray_variable__== 1, drop = True)
    masked_Area = mask2.Area.sum()
    masked_area.append(masked_Area)


    ds = ds*mask.__xarray_dataarray_variable__
    gdf = cut_Transcom_region(ds)

    #print(gdf)
    #gC/m^2/s für eine Woche 
    sums.append( gdf['CO2'].mean()*10**(-6)*masked_Area*10**(-12)*(2*24*60*60))

    for week in range(49,53): 
        ds = xr.open_dataarray(datapath_inversion+str(week)+'/'+spatial_result_name_before_week+str(week)+'.nc')
        mask = xr.open_dataset(datapathmasks+name_mask_before_week)#+str(week)+'.nc')
        mask = mask.rename({'Lat': 'latitude', 'Long': 'longitude', 'bioclass': '__xarray_dataarray_variable__'})
        Area_lat_long = np.zeros((len(dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]),len(mask.longitude.values) ))
        for i in range(len(mask.longitude.values)): 
            Area_lat_long[:,i] = dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]
        Area_array = xr.Dataset(data_vars = dict(Area = (['latitude', 'longitude'], Area_lat_long)), 
                                                coords= dict(latitude = (["latitude"],dfArea['Lat'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:] ), longitude = (['longitude'], mask.longitude.values)))
        mask2 = xr.merge([mask,Area_array])
        mask2 = mask2.where(mask2.__xarray_dataarray_variable__== 1, drop = True)
        masked_Area = mask2.Area.sum()
        masked_area.append(masked_Area)

        ds = ds*mask.__xarray_dataarray_variable__
        gdf = cut_Transcom_region(ds)
        #gC/m^2/s für eine Woche 
        sums.append( gdf['CO2'].mean()*10**(-6)*masked_Area*10**(-12)*(7*24*60*60))
    ds = xr.open_dataarray(datapath_inversion+'1/'+spatial_result_name_before_week+'1.nc')
    mask = xr.open_dataset(datapathmasks+name_mask_before_week)#'3.00e-02_CO2_3.00e-02_CO_mask_ak_threshold_0.001and_co_5e-1_week_1.nc')
    mask = mask.rename({'Lat': 'latitude', 'Long': 'longitude', 'bioclass': '__xarray_dataarray_variable__'})
    Area_lat_long = np.zeros((len(dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]),len(mask.longitude.values) ))
    for i in range(len(mask.longitude.values)): 
        Area_lat_long[:,i] = dfArea['Area'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:]
    Area_array = xr.Dataset(data_vars = dict(Area = (['latitude', 'longitude'], Area_lat_long)), 
                                            coords= dict(latitude = (["latitude"],dfArea['Lat'][(dfArea['Lat']>=mask.latitude.min().values)&(dfArea['Lat']<=mask.latitude.max().values)][:] ), longitude = (['longitude'], mask.longitude.values)))
    mask2 = xr.merge([mask,Area_array])
    mask2 = mask2.where(mask2.__xarray_dataarray_variable__== 1, drop = True)
    masked_Area = mask2.Area.sum()
    masked_area.append(masked_Area)

    #gC/m^2/s für eine Woche 
    gdf = cut_Transcom_region(ds)
    sums.append( gdf['CO2'].mean()*10**(-6)*masked_Area*10**(-12)*(1*24*60*60))
    data = xr.DataArray(data = sums, coords = {'week': [48,49,50,51,52,1]})
    print(data)
    print(masked_area)
    data.to_netcdf(datapath_inversion+savename)#str("{:e}".format(alpha))+'_total_emission_masked_with_area_weekly.nc')

from datetime import datetime
def multiply_masks_with_GFED(datapathmasks,masked_area, name_mask, year, month, molecule,datapath_and_name_GFED, mask_monthly = False,just_land_mask = False): 
    if molecule == 'CO': 
        factor = 12/28
    elif molecule =='CO2': 
        factor = 12/44
    GFED = pd.read_pickle(datapath_and_name_GFED)
    GFED.rename(columns ={'Long': 'longitude', 'Lat':'latitude'}, inplace = True)# -10 , 110
    GFED = GFED[(GFED['Year']==year)&(GFED['longitude']>=146)&(GFED['latitude']<=-26)&(GFED['latitude']>=-45)&(GFED['longitude']<=155)&(GFED['Month']==month)]#stimmt mit Literaturwert überein 
    #GFED['total_emission'] = GFED['total_emission']/GFED['Grid_area']
    xrGFED = GFED.set_index(['longitude', 'latitude']).to_xarray()
    weekly_mean_values = []
    if just_land_mask == False: 
        for idx,week in enumerate([48,49,50,51,52,1]): # 26, 146
            print('GFED')
            print(xrGFED)
            mask = xr.open_dataset(datapathmasks+name_mask)#+str(week)+'.nc')
            mask = mask.rename({'Lat': 'latitude', 'Long': 'longitude', 'bioclass': '__xarray_dataarray_variable__'})
            #mask = mask.rename({'Long': 'longitude', 'Lat':'latitude'})
            #print(xrGFED.Grid_area.sum()[0])
            #print(xrGFED.total_emission*masked_area[idx]*mask.__xarray_dataarray_variable__/xrGFED.Grid_area)
            #xrGFED = xrGFED.total_emission/xrGFED.Grid_area
            maskedGFED = xrGFED.total_emission#*mask.__xarray_dataarray_variable__.values#(7692024*10**6)
            print(maskedGFED)
            if week == 48: 
                weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(1/31))
            elif week == 1: 
                weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(2/31))
            else: 
                weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(7/31))
    else: 
        for idx,week in enumerate([48,49,50,51,52,1]):
            mask = xr.open_dataset(datapathmasks+'Mask_AU_land_bioclasses_based.nc')
            #mask = mask.rename({'Long': 'longitude', 'Lat':'latitude'})
            maskedGFED = xrGFED.total_emission*masked_area[idx]/xrGFED.Grid_area#*mask.bioclass
            if week == 48: 
                weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(1/31))
            elif week == 1: 
                weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(2/31))
            else: 
                weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(7/31))
    return weekly_mean_values



def multiply_masks_with_GFAS(datapathmasks,masked_area,name_mask, year, month, molecule,datapath_and_name_GFAS, just_land_mask = False): 
    if molecule == 'CO': 
        factor = 12/28
        random_factor = 12
        variable = 'cofire'
    elif molecule =='CO2': 
        random_factor = 1
        factor = 12/44
        variable = 'co2fire'
    GFAS = pd.read_pickle(datapath_and_name_GFAS) # 110, -10
    GFAS = GFAS[(GFAS['longitude']>=110)&(GFAS['latitude']<=-10)&(GFAS['latitude']>=-45)&(GFAS['longitude']<=155)]
    GFAS = GFAS[(pd.to_datetime(GFAS['date'])>=datetime(year= 2019, month = 12, day=1))&(pd.to_datetime(GFAS['date'])<=datetime(year= 2020, month = 1, day=1))]
    days_in_week = dict({'48': [1,2], '49': [2,3,4,5,6,7,8,9], '50': [9,10,11,12,13,14,15,16], '51': [16,17,18,19,20,21,22,23], '52':[23,24,25,26,27,28,29,30], '1':[30,31,32]})
    weekly_mean_values = []
    print((GFAS[variable]* np.cos(np.deg2rad(GFAS["latitude"][:]))* 111319* 111000*10**(-2)*10**(-9)*30.4*60*60*24*12).sum())
    for idx,week in enumerate([48,49,50,51,52]):
        GFAS_week = GFAS[(pd.to_datetime(GFAS['date'])>=datetime(year= 2019, month = 12, day=days_in_week[str(week)][0]))]
        GFAS_week= GFAS_week[(pd.to_datetime(GFAS_week['date'])<=datetime(year= 2019, month = 12, day=days_in_week[str(week)][-1]))]
        GFAS_week = GFAS_week.set_index(['date','latitude','longitude']).to_xarray()
        if just_land_mask == False: 
            mask = xr.open_dataset(datapathmasks+name_mask)#+str(week)+'.nc')
            mask = mask.rename({'Lat': 'latitude', 'Long': 'longitude', 'bioclass': '__xarray_dataarray_variable__'})
            maskedGFAS = GFAS_week[variable]* np.cos(np.deg2rad(GFAS_week["latitude"][:]))* 111319* 111000*10**(-2)*10**(-9)*mask.__xarray_dataarray_variable__
        #else: 
        #    mask = xr.open_dataset(datapathmasks+'Mask_AU_land_bioclasses_based.nc')
        #    maskedGFAS = GFAS_week[variable]*mask.bioclass

        #GFAS_week = GFAS_week.set_index(['longitude','latitude']).to_xarray()
        #
        print(maskedGFAS.max())
        if week == 48: 
            weekly_mean_values.append(maskedGFAS.sum()*(1*24*60*60)*random_factor)
        else: 
            weekly_mean_values.append(maskedGFAS.sum()*(7*24*60*60)*random_factor)

    for week in [1]: 
        GFAS_week = GFAS[(pd.to_datetime(GFAS['date'])>=datetime(year= 2019, month = 12, day=days_in_week[str(week)][0]))]
        GFAS_week = GFAS_week[(pd.to_datetime(GFAS_week['date'])<=datetime(year= 2020, month = 1, day=1))]
        GFAS_week = GFAS_week.set_index(['date','latitude','longitude']).to_xarray()
        if just_land_mask == False: 
            mask = xr.open_dataset(datapathmasks+name_mask)#+str(week)+'.nc')
            mask = mask.rename({'Lat': 'latitude', 'Long': 'longitude', 'bioclass': '__xarray_dataarray_variable__'})
            maskedGFAS = GFAS_week[variable]*mask.__xarray_dataarray_variable__*np.cos(np.deg2rad(GFAS_week["latitude"][:]))* 111319* 111000*10**(-2)*10**(-9)
        else: 
            mask = xr.open_dataset(datapathmasks+'Mask_AU_land_bioclasses_based.nc')
            maskedGFAS = GFAS_week[variable]#*mask.bioclass
        #maskedGFAS = GFAS_week.co2fire*mask.__xarray_dataarray_variable__
        
        weekly_mean_values.append(maskedGFAS.sum()*(2*24*60*60)*random_factor)
    return weekly_mean_values

def plot_CO2_GFAS_GFED_posterior_prior(savepath, savename,weekly_mean_GFAS, weekly_mean_GFED, posterior, prior, masked_area, prior_error, post_error):
    weeks = [48,49,50,51,52,53] 
    plt.rcParams.update({'font.size':18})   
    fig, ax = plt.subplot_mosaic([['lh', 'r'],['lb', 'r']],  gridspec_kw={'width_ratios': [5,1],'height_ratios':[5,1], 'hspace' : 0}, figsize = (10,6), constrained_layout = True)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0,
                        hspace=0.4)

    ax['lh'].plot(weeks, prior,color = 'orange')# orange
    ax['lh'].errorbar(weeks, prior, yerr = prior_error, marker = 'o', capthick=2, capsize=4,color = 'orange')
    ax['lh'].plot(weeks, posterior,color = 'coral')#,label='Posterior')# darkorange
    ax['lh'].errorbar(weeks, posterior, yerr = post_error, marker = 'o', capthick=2, capsize=4,color = 'coral')
    ax['lh'].plot(weeks, weekly_mean_GFAS,alpha = 0.9, marker = 'o',color = 'firebrick')#, label ='GFAS')# orangered
    ax['lh'].plot(weeks, weekly_mean_GFED, marker = 'o',color = 'maroon')# firebrick
    ax['lh'].set_ylim((-31,53))
    #ax.plot(weeks, GFAS_weekly_data,label = 'GFAS total', color = 'grey')

    ax['lb'].bar(weeks, np.array(masked_area)/(7692024*10**6)*100, color = 'slategray', width = 0.6)
    ax['lb'].set_xticks([48, 49, 50, 51, 52, 53])
    ax['lb'].set_xticklabels(['48', '49', '50', '51', '52', '1'])
    ax['lb'].set_ylabel('area [%]')
    ax['lb'].set_xlabel('week')
    ax['lh'].set_ylabel(f'CO$_2$ emission [TgC/week]')

    ax['r'].bar(0.6, np.array(prior).sum(),width = 0.6, label = 'Prior',color = 'orange')
    eb = ax['r'].errorbar(0.6, np.array(prior).sum(), yerr = np.array(prior_error).sum()/10,linestyle = 'dashed',marker = 'o', capthick=3, capsize=4,color = 'k')
    eb[-1][0].set_linestyle('--')
    ax['r'].bar(1.2,posterior.sum(), width = 0.6,label='Posterior', color = 'coral')
    eb2 = ax['r'].errorbar(1.2, np.array(posterior).sum(), yerr = np.array(post_error).sum()/10,marker = 'o', capthick=3, capsize=4, color = 'k')
    eb2[-1][0].set_linestyle('--')
    ax['r'].bar(1.8,np.array(weekly_mean_GFAS).sum() , width = 0.6,label='GFAS', alpha = 0.9,color = 'firebrick')#firebrick
    ax['r'].bar(2.4,np.array(weekly_mean_GFED).sum() , width = 0.6,label='GFED', color = 'maroon')
    ax['r'].set_xticks([0,1,2,3])
    ax['r'].set_xticklabels('')
    ax['r'].yaxis.set_label_position("right")
    ax['r'].yaxis.tick_right()
    ax['r'].set_ylabel(f'total CO$_2$ emission [TgC/month]')
    max = np.array([np.array(prior).sum(),posterior.sum(),np.array(weekly_mean_GFAS).sum(),np.array(weekly_mean_GFED).sum()]).max()
    ax['r'].set_ylim((0,max+0.05*max))#85))

    #ax2.legend()
    #ax.legend()
    fig.legend(bbox_to_anchor=(0.13,0.7,0.2,0.2))
    plt.show()
    fig.savefig(savepath+savename, facecolor = 'w', dpi = 300)

def plot_CO_GFAS_GFED_posterior_prior(savepath, savename, weekly_mean_GFAS, weekly_mean_GFED, posterior, prior, masked_area, prior_error, post_error):
    weeks = [48,49,50,51,52,53] 
    plt.rcParams.update({'font.size':18})   
    fig,ax = plt.subplot_mosaic([['lh', 'r'],['lb', 'r']],  gridspec_kw={'width_ratios': [5,1],'height_ratios':[5,1], 'hspace' : 0}, figsize = (10,6), constrained_layout = True)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0,
                        hspace=0.4)

    ax['lh'].plot(weeks, prior,color = 'orange')# orange
    ax['lh'].errorbar(weeks, prior, yerr = prior_error, marker = 'o', capthick=2, capsize=4,color = 'orange')
    ax['lh'].plot(weeks, posterior,color = 'coral')#,label='Posterior')# darkorange
    ax['lh'].errorbar(weeks, posterior, yerr = post_error, marker = 'o', capthick=2, capsize=4, color = 'coral')
    ax['lh'].plot(weeks, weekly_mean_GFAS,alpha = 0.9, marker = 'o',  color = 'firebrick')#, label ='GFAS')# orangered
    ax['lh'].plot(weeks, weekly_mean_GFED, marker = 'o', color = 'maroon')# firebrick
    ax['lh'].set_ylim((-4.1,6))
    #ax.plot(weeks, GFAS_weekly_data,label = 'GFAS total', color = 'grey')

    ax['lh'].set_xticks([48, 49, 50, 51, 52, 53])
    ax['lh'].set_xticklabels(['48', '49', '50', '51', '52', '1'])
    ax['lh'].set_xlabel('week')
    ax['lh'].set_ylabel(f'CO emission [TgC/week]')

    ax['lb'].bar(weeks, np.array(masked_area)/(7692024*10**6)*100, color = 'slategray', width = 0.6)
    ax['lb'].set_xticks([48, 49, 50, 51, 52, 53])
    ax['lb'].set_xticklabels(['48', '49', '50', '51', '52', '1'])
    ax['lb'].set_ylabel('area [%]')

    ax['r'].bar(0.6, np.array(prior).sum(),width = 0.6, label = 'Prior',color = 'orange')
    eb = ax['r'].errorbar(0.6, np.array(prior).sum(), yerr = np.array(prior_error).sum()/10,linestyle = 'dashed',marker = 'o', capthick=3, capsize=4,color = 'k')
    eb[-1][0].set_linestyle('--')
    ax['r'].bar(1.2,posterior.sum(), width = 0.6,label='Posterior', color = 'coral')
    ax['r'].errorbar(1.2, np.array(posterior).sum(), yerr = np.array(post_error).sum(),marker = 'o', capthick=3, capsize=4, color = 'k')
    ax['r'].bar(1.8,np.array(weekly_mean_GFAS).sum() , width = 0.6,label='GFAS', alpha = 0.9,color = 'firebrick')#firebrick
    ax['r'].bar(2.4,np.array(weekly_mean_GFED).sum() , width = 0.6,label='GFED', color = 'maroon')
    ax['r'].set_xticks([0,1,2,3])
    ax['r'].set_xticklabels('')
    ax['r'].yaxis.set_label_position("right")
    ax['r'].yaxis.tick_right()
    ax['r'].set_ylabel('total CO emission [TgC/month]')
    max = np.array([np.array(prior).sum(),posterior.sum(),np.array(weekly_mean_GFAS).sum(),np.array(weekly_mean_GFED).sum()]).max()
    ax['r'].set_ylim((0,max+0.1*max))#85))
    #ax2.set_ylim((0,4))#6.7))

    #ax2.legend()
    #ax.legend()
    fig.legend(bbox_to_anchor=(0.13,0.7,0.2,0.2))
    plt.show()
    fig.savefig(savepath+savename, facecolor = 'w', dpi = 300)

def get_prior_weekly(masked_area, datapath_inversion,filename): 
    prior_weekly = []
    prior = xr.open_dataset(datapath_inversion+'48/'+filename+str(48)+'.nc')#'prior_spatially_week_'+str(48))
    prior_weekly.append(prior.__xarray_dataarray_variable__.mean()*10**(-18)*masked_area[0]*(1*24*60*60))
    for idx,week in enumerate([49,50,51,52]): 
        prior = xr.open_dataset(datapath_inversion+str(week)+'/'+filename+str(week)+'.nc')#'prior_spatially_week_'+str(week))
        prior_weekly.append(prior.__xarray_dataarray_variable__.mean()*10**(-18)*masked_area[idx+1]*(7*24*60*60))
    prior = xr.open_dataset(datapath_inversion+'1/'+filename+str(1)+'.nc')#'prior_spatially_week_'+str(1))
    prior_weekly.append(prior.__xarray_dataarray_variable__.mean()*10**(-18)*masked_area[5]*(2*24*60*60))

    return prior_weekly


GFED_CO = '/work/bb1170/RUN/b382105/Dataframes/GFEDs/GDF_AU_CO_2019_2020.pkl'#GDF_CO_AU20192020_regr1x1.pkl'
GFED_CO2 =  '/work/bb1170/RUN/b382105/Dataframes/GFEDs/GDF_CO2_AU20162021.pkl'#GDF_CO2_AU20192020_regr1x1.pkl'
GFAS_CO = '/work/bb1170/RUN/b382105/Dataframes/GFAS/GDF_CO_Test2_GFAS_AU_2019_2020.pkl'
GFAS_CO2 = '/work/bb1170/RUN/b382105/Dataframes/GFAS/GDF_Test2_GFAS_AU_2009_2020.pkl'
savepath_CO = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/total_emissions/'
savepath_CO2 = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/total_emissions/'
datapathmask = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/48/'
datapath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/All_weeks/'
name_mask = 'input_mask_masked_without_19_and_23.nc'
name_spatial_results_before_week_CO = 'spatial_results_CO_fire_week_'
name_spatial_results_before_week_CO2 = 'spatial_results_CO2_fire_week_'
weeks = [48,49,50,51,52,53] 
# gridded masked area: 
masked_area = [3.04356787e+10, 0,2.20228191e+11, 7.48989837e+11, 1.30073375e+12, 6.56704376e+11]

masked_area = get_CO2_masked_total_emissions(datapath, datapathmask, name_mask, name_spatial_results_before_week_CO2,str("{:e}".format(1e-2))+'_CO2_'+str("{:e}".format(2.12))+'_CO_total_emission_CO2_masked_with_area_weekly.nc' )
get_CO_masked_total_emissions(datapath, datapathmask, name_mask, name_spatial_results_before_week_CO,str("{:e}".format(1e-2))+'_CO2_'+str("{:e}".format(2.12))+'_CO_total_emission_CO_masked_with_area_weekly.nc')
CO_posterior = xr.open_dataarray(datapath+str("{:e}".format(1e-2))+'_CO2_'+str("{:e}".format(2.12))+'_CO_total_emission_CO_masked_with_area_weekly.nc')
CO2_posterior = xr.open_dataarray(datapath+str("{:e}".format(1e-2))+'_CO2_'+str("{:e}".format(2.12))+'_CO_total_emission_CO2_masked_with_area_weekly.nc')

CO_prior = get_prior_weekly(masked_area, datapath, 'Prior_CO_fire_grid_spatial_week_')
CO2_prior = get_prior_weekly(masked_area,datapath,'Prior_CO2_fire_grid_spatial_week_')


GFAS_weekly_CO2 = multiply_masks_with_GFAS(datapathmask, masked_area,name_mask, 2019, 12, 'CO2',GFAS_CO2, just_land_mask = False)
GFED_weekly_CO2 = multiply_masks_with_GFED(datapathmask,masked_area, name_mask, 2019, 12, 'CO2',GFED_CO2, mask_monthly = False,just_land_mask = False)
GFED_weekly_CO = multiply_masks_with_GFED(datapathmask,masked_area, name_mask, 2019, 12, 'CO', GFED_CO, mask_monthly = False,just_land_mask = False)
GFAS_weekly_CO = multiply_masks_with_GFAS(datapathmask,masked_area,name_mask, 2019, 12, 'CO', GFAS_CO, just_land_mask = False)

# for every week masked for region numbers: 
CO_prior_errors_weekly = []
CO_posterior_errors_weekly = []
CO2_prior_errors_weekly = []
CO2_posterior_errors_weekly = []
for idx, week in enumerate([48,49,50,51,52,1]): 
    if week == 48: 
        days = 1
    elif week == 1:
        days = 2
    else: days = 7
    errors = pd.read_pickle(datapath+str(week)+'/1.00e-02_CO2_2.12e+00_CO_predictions.pkl')
    prior_errors = np.sqrt(errors['prior_err_cov'])*errors['prior_flux_eco']
    prior_errors = prior_errors.drop([19,23,73,77])

    CO2_prior_errors_weekly.append(prior_errors[5:24].mean()*12*10**(-12)*24*60*60*days*masked_area[idx]*10**(-6))
    CO_prior_errors_weekly.append(prior_errors[56:76].sum()*12*10**(-12)*24*60*60*days*masked_area[idx]*10**(-9))
    posterior_errors = errors['posterior_err_std']*errors['prior_flux_eco']
    CO2_posterior_errors_weekly.append(posterior_errors[5:24].mean()*12*10**(-12)*24*60*60*days*masked_area[idx]*10**(-6))
    CO_posterior_errors_weekly.append(posterior_errors[56:76].mean()*12*10**(-12)*24*60*60*days*masked_area[idx]*10**(-9))
print(CO2_prior_errors_weekly)
print(CO_posterior_errors_weekly)
print(CO_prior_errors_weekly)

plot_CO_GFAS_GFED_posterior_prior(savepath_CO, 'Comparison_CO_GFED_GFAS_PRIOR_POSTERIOR_areas_considered.png', GFAS_weekly_CO, GFED_weekly_CO, CO_posterior, CO_prior, 
                                  masked_area, CO_prior_errors_weekly, CO_posterior_errors_weekly)
plot_CO2_GFAS_GFED_posterior_prior(savepath_CO2, 'Comparison_CO2_GFED_GFAS_PRIOR_POSTERIOR_areas_considered.png', GFAS_weekly_CO2, GFED_weekly_CO2, CO2_posterior, CO2_prior,
                                    masked_area, CO2_prior_errors_weekly, CO2_posterior_errors_weekly)