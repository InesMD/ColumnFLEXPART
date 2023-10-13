import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#for Coupled Inversion CO 
import geopandas as gpd
from datetime import datetime

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



def calculate_total_weekly_emissions():
    sumsCOfire = []
    sumsCO2fire = []
    sumsCObio = []
    sumsCO2bio = []

    alphaCO2 = 1e-2
    alphaCO = 2.12

    week = 48

    datapath_inversion = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/48/'
    COfire = xr.open_dataarray(datapath_inversion+'spatial_results_CO_fire_week_'+str(week)+'.nc')
    CO2fire = xr.open_dataarray(datapath_inversion+'spatial_results_CO2_fire_week_'+str(week)+'.nc')
    CObio = xr.open_dataarray(datapath_inversion+'spatial_results_CO_ant_bio_week_'+str(week)+'.nc')
    CO2bio = xr.open_dataarray(datapath_inversion+'spatial_results_CO2_ant_bio_week_'+str(week)+'.nc')
    gdf_CO_fire = cut_Transcom_region(COfire)
    gdf_CO2_fire = cut_Transcom_region(CO2fire)
    gdf_CO_bio = cut_Transcom_region(CObio)
    gdf_CO2_bio = cut_Transcom_region(CO2bio)
    sumsCOfire.append(gdf_CO_fire['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(1*24*60*60))
    sumsCO2fire.append(gdf_CO2_fire['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(1*24*60*60))
    sumsCObio.append(gdf_CO_bio['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(1*24*60*60))
    sumsCO2bio.append(gdf_CO2_bio['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(1*24*60*60))



    for week in [49,50,51,52]: 

        datapath_inversion = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/'+str(week)+'/'
        #datapathmasks = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/test/'

        #ds = xr.open_dataarray(datapath_inversion+str("{:e}".format(alphaCO2))+'_CO2_'+str("{:e}".format(alphaCO))+'_CO_spatial_results_week_48.nc')
        COfire = xr.open_dataarray(datapath_inversion+'spatial_results_CO_fire_week_'+str(week)+'.nc')
        CO2fire = xr.open_dataarray(datapath_inversion+'spatial_results_CO2_fire_week_'+str(week)+'.nc')
        CObio = xr.open_dataarray(datapath_inversion+'spatial_results_CO_ant_bio_week_'+str(week)+'.nc')
        CO2bio = xr.open_dataarray(datapath_inversion+'spatial_results_CO2_ant_bio_week_'+str(week)+'.nc')
        #mask = xr.open_dataset(datapathmasks+'mask_co_co2_for_GFED_GFAS_week_48.nc')
        #ds = ds*mask.__xarray_dataarray_variable__
        gdf_CO_fire = cut_Transcom_region(COfire)
        gdf_CO2_fire = cut_Transcom_region(CO2fire)
        gdf_CO_bio = cut_Transcom_region(CObio)
        gdf_CO2_bio = cut_Transcom_region(CO2bio)
        #gC/m^2/s für eine Woche 
        sumsCOfire.append(gdf_CO_fire['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(7*24*60*60))
        sumsCO2fire.append(gdf_CO2_fire['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(7*24*60*60))
        sumsCObio.append(gdf_CO_bio['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(7*24*60*60))
        sumsCO2bio.append(gdf_CO2_bio['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(7*24*60*60))


    week = 1
    datapath_inversion = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/1/'
    COfire = xr.open_dataarray(datapath_inversion+'spatial_results_CO_fire_week_'+str(week)+'.nc')
    CO2fire = xr.open_dataarray(datapath_inversion+'spatial_results_CO2_fire_week_'+str(week)+'.nc')
    CObio = xr.open_dataarray(datapath_inversion+'spatial_results_CO_ant_bio_week_'+str(week)+'.nc')
    CO2bio = xr.open_dataarray(datapath_inversion+'spatial_results_CO2_ant_bio_week_'+str(week)+'.nc')
    gdf_CO_fire = cut_Transcom_region(COfire)
    gdf_CO2_fire = cut_Transcom_region(CO2fire)
    gdf_CO_bio = cut_Transcom_region(CObio)
    gdf_CO2_bio = cut_Transcom_region(CO2bio)
    sumsCOfire.append(gdf_CO_fire['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(2*24*60*60))
    sumsCO2fire.append(gdf_CO2_fire['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(2*24*60*60))
    sumsCObio.append(gdf_CO_bio['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(2*24*60*60))
    sumsCO2bio.append(gdf_CO2_bio['CO2'].mean()*10**(-6)*7692024*10**6*10**(-12)*(2*24*60*60))

    datapath_inversion = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/total_emissions/'
    print(sumsCOfire)
    data = xr.DataArray(data = sumsCOfire, coords = {'week': [48,49,50,51,52,1]})
    data.to_netcdf(datapath_inversion+str("{:.2e}".format(alphaCO2))+'_'+str("{:.2e}".format(alphaCO))+'_total_emission_CO_fire_weekly.nc')
    data = xr.DataArray(data = sumsCO2fire, coords = {'week': [48,49,50,51,52,1]})
    data.to_netcdf(datapath_inversion+str("{:.2e}".format(alphaCO2))+'_'+str("{:.2e}".format(alphaCO))+'_total_emission_CO2_fire_weekly.nc')
    data = xr.DataArray(data = sumsCObio, coords = {'week': [48,49,50,51,52,1]})
    data.to_netcdf(datapath_inversion+str("{:.2e}".format(alphaCO2))+'_'+str("{:.2e}".format(alphaCO))+'_total_emission_CO_bio_weekly.nc')
    data = xr.DataArray(data = sumsCO2bio, coords = {'week': [48,49,50,51,52,1]})
    data.to_netcdf(datapath_inversion+str("{:.2e}".format(alphaCO2))+'_'+str("{:.2e}".format(alphaCO))+'_total_emission_CO2_bio_weekly.nc')

def plot_weekly_emissions(alphaCO,alphaCO2):
# for coupled Inversion CO
    prior_fluxCO_fire = []
    prior_fluxCO2_fire = []
    prior_fluxCO_bio = []
    prior_fluxCO2_bio = []
    for week in [48,49,50,51,52,1]:
        pkl = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/'+str(week)+'/'+str("{:.2e}".format(alphaCO2))+'_CO2_'+str("{:.2e}".format(alphaCO))+'_CO_predictions.pkl')
        #print(pkl['prior_flux_eco'][:int(108/4)].values)
        prior_fluxCO2_fire.append(pkl['prior_flux_eco'][:int(108/4)].values.mean()*10**(-3)*12)
        prior_fluxCO2_bio.append(pkl['prior_flux_eco'][int(108/4):int(108/4)*2].values.mean()*10**(-3)*12)
        prior_fluxCO_fire.append(pkl['prior_flux_eco'][int(108/4)*2:int(108/4)*3].values.mean()*10**(-3)*12)
        prior_fluxCO_bio.append(pkl['prior_flux_eco'][int(108/4)*3:].values.mean()*10**(-3)*12)

    datapath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/total_emissions/'
    ds = xr.open_dataarray(datapath+str("{:.2e}".format(alphaCO2))+'_'+str("{:.2e}".format(alphaCO))+'_total_emission_CO_fire_weekly.nc')
    dsbio = xr.open_dataarray(datapath+str("{:.2e}".format(alphaCO2))+'_'+str("{:.2e}".format(alphaCO))+'_total_emission_CO_bio_weekly.nc')
    plt.rcParams.update({'font.size':19})
    plt.figure(figsize = (8,6))
    plt.plot([48,49,50,51,52,53], ds, label = 'CO fire')
    plt.plot([48,49,50,51,52,53], dsbio, label = 'CO bio')
    plt.plot([48,49,50,51,52,53], np.array(ds)+np.array(dsbio), label = 'CO total')
    plt.plot([48,49,50,51,52,53], np.array(prior_fluxCO_fire), label = 'prior CO fire')
    plt.plot([48,49,50,51,52,53], np.array(prior_fluxCO_bio), label = 'prior CO bio')
    plt.plot([48,49,50,51,52,53], np.array(prior_fluxCO_bio)+np.array(prior_fluxCO_fire), label = 'prior CO total')
    plt.legend()
    plt.xlabel('week')
    plt.ylabel(f'CO emission [TgC/week]')


    print((np.array(ds)+np.array(dsbio)).sum()*4*3/7)
    print((np.array(prior_fluxCO_bio)+np.array(prior_fluxCO_fire)).sum()*4*3/7)

def calculate_GFAS_weekly_mean(molecule,datapath_and_name_GFAS): 
    if molecule == 'CO': 
        factor = 12/28
        variable = 'cofire'
    elif molecule =='CO2': 
        factor = 12/44
        variable = 'co2fire'
    GFAS = pd.read_pickle(datapath_and_name_GFAS)
    GFAS = GFAS[(GFAS['longitude']>=110)&(GFAS['latitude']<=-10)&(GFAS['latitude']>=-45)&(GFAS['longitude']<=155)]
    GFAS = GFAS[(pd.to_datetime(GFAS['date'])>=datetime(year= 2019, month = 12, day=1))&(pd.to_datetime(GFAS['date'])<=datetime(year= 2020, month = 1, day=1))]
    days_in_week = dict({'48': [1,2], '49': [2,3,4,5,6,7,8,9], '50': [9,10,11,12,13,14,15,16], '51': [16,17,18,19,20,21,22,23], '52':[23,24,25,26,27,28,29,30], '1':[30,31,32]})
    weekly_mean_values = []
    for week in [48,49,50,51,52]:
        GFAS_week = GFAS[(pd.to_datetime(GFAS['date'])>=datetime(year= 2019, month = 12, day=days_in_week[str(week)][0]))]
        GFAS_week= GFAS_week[(pd.to_datetime(GFAS_week['date'])<=datetime(year= 2019, month = 12, day=days_in_week[str(week)][-1]))]
        GFAS_week = GFAS_week.set_index(['date','latitude','longitude']).to_xarray()

        if week == 48: 
            weekly_mean_values.append(GFAS_week[variable].mean().values*7692024*10**6*10**(-9)*factor*(1*24*60*60))
        else: 
            weekly_mean_values.append(GFAS_week[variable].mean().values*7692024*10**6*10**(-9)*factor*(7*24*60*60))

    for week in [1]: 
        GFAS_week = GFAS[(pd.to_datetime(GFAS['date'])>=datetime(year= 2019, month = 12, day=days_in_week[str(week)][0]))]
        GFAS_week = GFAS_week[(pd.to_datetime(GFAS_week['date'])<=datetime(year= 2020, month = 1, day=1))]
        GFAS_week = GFAS_week.set_index(['date','latitude','longitude']).to_xarray()
       
        weekly_mean_values.append(GFAS_week[variable].mean().values*7692024*10**6*10**(-9)*factor*(2*24*60*60))
    return weekly_mean_values


def calculate_GFED_weekly_mean(year, month, molecule,datapath_and_name_GFED): 
    if molecule == 'CO': 
        factor = 12/28
    elif molecule =='CO2': 
        factor = 12/44
    GFED = pd.read_pickle(datapath_and_name_GFED)
    GFED.rename(columns ={'Long': 'longitude', 'Lat':'latitude'}, inplace = True)
    GFED = GFED[(GFED['Year']==year)&(GFED['longitude']>=110)&(GFED['latitude']<=-10)&(GFED['latitude']>=-45)&(GFED['longitude']<=155)&(GFED['Month']==month)]#stimmt mit Literaturwert überein 
    xrGFED = GFED.set_index(['longitude', 'latitude']).to_xarray()
    weekly_mean_values = []
    for week in [48,49,50,51,52,1]:
        if week == 48: 
            weekly_mean_values.append(xrGFED.total_emission.sum(['latitude', 'longitude']).values*factor*(1/31))
        elif week == 1: 
            weekly_mean_values.append(xrGFED.total_emission.sum(['latitude', 'longitude']).values*factor*(2/31))
        else: 
            weekly_mean_values.append(xrGFED.total_emission.sum(['latitude', 'longitude']).values*factor*(7/31))

    return weekly_mean_values


def plot_CO2_GFAS_GFED_posterior_prior(savepath, weekly_mean_GFAS, weekly_mean_GFED, posterior, prior):
    weeks = [48,49,50,51,52,53] 
    plt.rcParams.update({'font.size':18})   
    fig, (ax, ax2) = plt.subplots(1,2,  gridspec_kw={'width_ratios': [5,1]},figsize = (10,6))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0,
                        hspace=0.4)

    ax.plot(weeks, prior,color = 'orange')# orange
    ax.plot(weeks, posterior,color = 'coral')#,label='Posterior')# darkorange
    ax.plot(weeks, weekly_mean_GFAS,alpha = 0.9, color = 'firebrick')#, label ='GFAS')# orangered
    ax.plot(weeks, weekly_mean_GFED, color = 'maroon')# firebrick
    #ax.plot(weeks, GFAS_weekly_data,label = 'GFAS total', color = 'grey')

    ax.set_xticks([48, 49, 50, 51, 52, 53])
    ax.set_xticklabels(['48', '49', '50', '51', '52', '1'])
    ax.set_xlabel('week')
    ax.set_ylabel(f'CO$_2$ emission [TgC/week]')

    ax2.bar(0.6, np.array(prior).sum(),width = 0.6, label = 'Prior',color = 'orange')
    ax2.bar(1.2,posterior, width = 0.6,label='Posterior', color = 'coral')
    ax2.bar(1.8,np.array(weekly_mean_GFAS).sum() , width = 0.6,label='GFAS', alpha = 0.9,color = 'firebrick')#firebrick
    ax2.bar(2.4,np.array(weekly_mean_GFED).sum() , width = 0.6,label='GFED', color = 'maroon')
    ax2.set_xticks([0,1,2,3])
    ax2.set_xticklabels('')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel(f'total CO$_2$ emission [TgC/month]')
    ax2.set_ylim((0,85))

    #ax2.legend()
    #ax.legend()
    fig.legend(bbox_to_anchor=(0.13,0.7,0.2,0.2))
    plt.show()
    fig.savefig(savepath+'CO2_fluxes_posterior_GFAS_GFED_masked_weekly_bio_and_fire.png', facecolor = 'w', dpi = 300)

def plot_CO_GFAS_GFED_posterior_prior(savepath, weekly_mean_GFAS, weekly_mean_GFED, posterior, prior):
    weeks = [48,49,50,51,52,53] 
    plt.rcParams.update({'font.size':18})   
    fig, (ax, ax2) = plt.subplots(1,2,  gridspec_kw={'width_ratios': [5,1]},figsize = (10,6))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0,
                        hspace=0.4)

    ax.plot(weeks, prior,color = 'orange')# orange
    ax.plot(weeks, posterior,color = 'coral')#,label='Posterior')# darkorange
    ax.plot(weeks, weekly_mean_GFAS,alpha = 0.9, color = 'firebrick')#, label ='GFAS')# orangered
    ax.plot(weeks, weekly_mean_GFED, color = 'maroon')# firebrick
    #ax.plot(weeks, GFAS_weekly_data,label = 'GFAS total', color = 'grey')

    ax.set_xticks([48, 49, 50, 51, 52, 53])
    ax.set_xticklabels(['48', '49', '50', '51', '52', '1'])
    ax.set_xlabel('week')
    ax.set_ylabel(f'CO emission [TgC/week]')

    ax2.bar(0.6, np.array(prior).sum(),width = 0.6, label = 'Prior',color = 'orange')
    ax2.bar(1.2,posterior, width = 0.6,label='Posterior', color = 'coral')
    ax2.bar(1.8,np.array(weekly_mean_GFAS).sum() , width = 0.6,label='GFAS', alpha = 0.9,color = 'firebrick')#firebrick
    ax2.bar(2.4,np.array(weekly_mean_GFED).sum() , width = 0.6,label='GFED', color = 'maroon')
    ax2.set_xticks([0,1,2,3])
    ax2.set_xticklabels('')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel('total CO emission [TgC/month]')
    ax2.set_ylim((0,6.7))

    #ax2.legend()
    #ax.legend()
    fig.legend(bbox_to_anchor=(0.13,0.7,0.2,0.2))
    plt.show()
    fig.savefig(savepath+'CO_fluxes_posterior_GFAS_GFED_masked_weekly_bio_and_fire.png', facecolor = 'w', dpi = 300)


############### Calculate everything ##################
savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/total_emissions/'
alphaCO = 2.12
alphaCO2 = 1e-2
# CO GFAS and GFED
weekly_mean_GFAS = calculate_GFAS_weekly_mean('CO','/work/bb1170/RUN/b382105/Dataframes/GFAS/GDF_CO_Test2_GFAS_AU_2019_2020.pkl')
weekly_mean_GFED = calculate_GFED_weekly_mean(2019, 12,'CO', '/work/bb1170/RUN/b382105/Dataframes/GFEDs/GDF_CO_AU20192020_regr1x1.pkl')
print('CO GFAS weekly mean:')
print(weekly_mean_GFAS)
print('CO GFAS month sum: '+str(np.array(weekly_mean_GFAS).sum()*4*3/7))
print('CO GFED weekly mean:')
print(weekly_mean_GFED)
print('CO GFED month sum: '+ str(np.array(weekly_mean_GFED).sum()*4*3/7))

# CO2 GFAS and GFED
weekly_mean_GFAS_CO2 = calculate_GFAS_weekly_mean('CO2','/work/bb1170/RUN/b382105/Dataframes/GFAS/GDF_Test2_GFAS_AU_2009_2020.pkl')
weekly_mean_GFED_CO2 = calculate_GFED_weekly_mean(2019, 12,'CO2', '/work/bb1170/RUN/b382105/Dataframes/GFEDs/GDF_CO2_AU20192020_regr1x1.pkl')
print('CO2 GFAS weekly mean:')
print(weekly_mean_GFAS_CO2)
print('CO2 GFED month sum: '+ str(np.array(weekly_mean_GFED_CO2).sum()*4*3/7))
print('CO2 GFED weekly mean:')
print(weekly_mean_GFED_CO2)
print('CO2 GFED month sum: '+ str(np.array(weekly_mean_GFED_CO2).sum()*4*3/7))

### get prior ##### achtung prdictions werden überschrieben!!!!!!!!!!!!
prior_fluxCO_fire = []
prior_fluxCO2_fire = []
prior_fluxCO_bio = []
prior_fluxCO2_bio = []
for week in [48,49,50,51,52,1]:
    pkl = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/'+str(week)+'/'+str("{:.2e}".format(alphaCO2))+'_CO2_'+str("{:.2e}".format(alphaCO))+'_CO_predictions.pkl')
    #print(pkl['prior_flux_eco'][:int(108/4)].values)
    prior_fluxCO2_fire.append(pkl['prior_flux_eco'][:int(108/4)].values.mean()*12)
    prior_fluxCO2_bio.append(pkl['prior_flux_eco'][int(108/4):int(108/4)*2].values.mean()*12)
    prior_fluxCO_fire.append(pkl['prior_flux_eco'][int(108/4)*2:int(108/4)*3].values.mean()*10**(-3)*12)
    prior_fluxCO_bio.append(pkl['prior_flux_eco'][int(108/4)*3:].values.mean()*10**(-3)*12)


###### Posterior CO ######## 
#calculate_total_weekly_emissions()
datapath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/total_emissions/'
ds = xr.open_dataarray(datapath+str("{:.2e}".format(alphaCO2))+'_'+str("{:.2e}".format(alphaCO))+'_total_emission_CO_fire_weekly.nc')
dsbio = xr.open_dataarray(datapath+str("{:.2e}".format(alphaCO2))+'_'+str("{:.2e}".format(alphaCO))+'_total_emission_CO_bio_weekly.nc')

# Plot CO 
plot_CO_GFAS_GFED_posterior_prior(savepath,weekly_mean_GFAS,weekly_mean_GFED, np.array(ds)+np.array(dsbio), np.array(prior_fluxCO_fire)+np.array(prior_fluxCO_bio))

###### Posterior CO2 ######## 
#calculate_total_weekly_emissions()
datapath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/total_emissions/'
ds = xr.open_dataarray(datapath+str("{:.2e}".format(alphaCO2))+'_'+str("{:.2e}".format(alphaCO))+'_total_emission_CO2_fire_weekly.nc')
dsbio = xr.open_dataarray(datapath+str("{:.2e}".format(alphaCO2))+'_'+str("{:.2e}".format(alphaCO))+'_total_emission_CO2_bio_weekly.nc')

# Plot CO2
plot_CO2_GFAS_GFED_posterior_prior(savepath,weekly_mean_GFAS_CO2,weekly_mean_GFED_CO2, np.array(ds)+np.array(dsbio), np.array(prior_fluxCO2_fire)+np.array(prior_fluxCO2_bio))