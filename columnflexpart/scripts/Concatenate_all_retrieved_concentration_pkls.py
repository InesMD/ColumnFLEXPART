import pandas as pd 
from columnflexpart.classes.coupled_inversion import CoupledInversion 
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import cartopy.crs as ccrs
import matplotlib as mpl
import xarray as xr
def plot_total_conc_with_errors(df, savepath, alpha, alphaCO): 
    # plotting 
    df = df.sort_values(['time'], ascending = True)
    df['date'] = pd.to_datetime(df['time'], format = '%Y-%M-%D').dt.date
    ds_mean = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/preparation/one_hour_runs/TCCON_mean_measurements_sept_to_march.pkl')
    #ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =9, day=1))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =2, day=28))]
    df= df[(df['time']>=datetime.datetime(year=2019, month =8, day=30))&(df['time']<= datetime.datetime(year=2020, month =3, day=1))].reset_index()

    ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =8, day=30))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =3, day=1))].dropna().sort_values(['datetime'], ascending = True).reset_index()
    print(ds_mean)
    df.insert(loc = 0, column = 'number_of_measurements', value = ds_mean['number_of_measurements'])
    print(df)

    plt.rcParams.update({'font.size':18})   
    fig, ax = plt.subplots(3,1,sharex = True, gridspec_kw={'height_ratios': [4,4,1.5]},figsize=(18,10))
    Xaxis = np.arange(0,len(df['time'][:]),1)
    ax[0].set_ylabel(r'CO$_2$ [ppm]')
    ax[0].set_xlabel('date')
    #ax[0].set_xticks(Xaxis, ['']*len(df['time'][:]))
    #ax1.tick_params(axis = 'y')
    max_value = max(abs(df['CO2_fire'].max()), abs(df['CO2_fire'].min()))
    ax[0].set_ylim((406, max_value+2))

 
    #total CO2
    total_CO2 = df['CO2_fire']+df['CO2_bio']-df['CO2_background']
    ax[0].plot(Xaxis-0.3, df['CO2_meas'],color = 'dimgrey',label = r'measurements')

    ax[0].plot(Xaxis, df['CO2_background'], color = 'k', label = 'background')
    #prior CO2
    lns1 = ax[0].plot(Xaxis,df['CO2_prior'],color = 'salmon',  label = r'total prior')
    ax[0].fill_between(Xaxis, df['CO2_prior']-(df['CO2_prior_std']), df['CO2_prior']+(df['CO2_prior_std']), color = 'salmon', alpha = 0.3)

    lns1 = ax[0].plot(Xaxis,total_CO2,color = 'red',  label = r'total posterior')
    ax[0].fill_between(Xaxis, total_CO2-(df['CO2fire_std']+df['CO2bio_std']), total_CO2+(df['CO2fire_std']+df['CO2bio_std']), color = 'red', alpha = 0.5)


    ## CO plot
    ax2 = ax[1]#.twinx()
    ax2.set_ylabel(r'CO [ppb]')
    max_value = max(abs(df['CO_fire'].max()), abs(df['CO_fire'].min()))
    ax2.set_ylim((0, max_value+100))
    # total CO
    total_CO = df['CO_fire']+df['CO_bio']-df['CO_background']
    ax2.plot(Xaxis-0.3, df['CO_meas'],color = 'dimgrey',label = r'measurements')
    ax2.plot(Xaxis, df['CO_background'], color = 'k', label = 'background')
    lns1 = ax2.plot(Xaxis,df['CO_prior'],color = 'salmon',  label = r'total prior')
    ax2.fill_between(Xaxis, df['CO_prior']-(df['CO_prior_std']), df['CO_prior']+(df['CO_prior_std']), color = 'salmon', alpha = 0.3)
    lns1 = ax2.plot(Xaxis,total_CO,color = 'red',  label = r'total posterior')
    ax2.fill_between(Xaxis, total_CO-(df['COfire_std']+df['CObio_std']), total_CO+(df['COfire_std']+df['CObio_std']), color = 'red', alpha = 0.5)
    # prior CO 
    #total_prior_CO = df['CO_fire_prior']+df['CO_bio_prior']-df['CO_background']
    ax[2].bar(Xaxis-0.3, df['number_of_measurements'], width=0.6, color = 'dimgrey')
    ax[2].set_ylabel('N', labelpad=17)
    ax[2].grid(axis = 'x')
    ax2.grid(axis = 'both')
    ax[0].grid(axis = 'both')
    ax[1].legend(loc = 'upper left')
    ax[2].set_xlim((0,len(df['time'])))
    
    # ticks
    ticklabels = ['']*len(df['time'][:])
    # Every 4th ticklable shows the month and day
    #ticklabels[0] = df['date'][0]
    reference = df['date'][0].month
    ticklabel_list = [df['date'][0]]
    index_of_ticklabel_list = [0]
    for i in np.arange(1,len(df['time'][:])):
        if df['date'][i].month>reference and reference<12:
            ticklabel_list.append(df['date'][i])
            index_of_ticklabel_list.append(i)
            ticklabels[i] = df['date'][i]
            reference = df['date'][i].month
        elif reference == 12 and df['date'][i].month == 1: 
            ticklabel_list.append(df['date'][i])
            index_of_ticklabel_list.append(i)
            ticklabels[i] = df['date'][i]
            reference = df['date'][i].month
    #ax[1].set_xticks(Xaxis, ticklabels)
    print(ticklabel_list)
    print(index_of_ticklabel_list)
    ax[2].set_xticks(index_of_ticklabel_list, ticklabel_list)
    ax[2].xaxis.set_major_formatter(mpl.ticker.FixedFormatter(ticklabel_list))

    #ds_mean = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/preparation/one_hour_runs/TCCON_mean_measurements_sept_to_march.pkl')
    #ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =9, day=1))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =2, day=28))]
    #ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =12, day=2))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =1, day=1))]


    #myFmt = DateFormatter("%Y-%m-%d")
    #ax[1].xaxis.set_major_formatter(myFmt)
    plt.gcf().autofmt_xdate(rotation = 45)
    
    plt.axhline(y=0, color='k', linestyle='-')
    
    #ax[0].set_xlim((datetime.datetime(year = 2019, month = 12, day = 1), datetime.datetime(year = 2019, month = 12, day = 31, hour = 23)))
    #fig.autofmt_xdate()
    plt.subplots_adjust(hspace=0)

    fig.savefig(savepath+"{:.2e}".format(alpha)+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_total_concentrations_with_errors_measurement_entire_time_series.png", dpi = 300, bbox_inches = 'tight')


def plotting(ds,savepath):

    ds.insert(loc=1, column = 'CO2_total_post', value= ds['CO2_fire']+ds['CO2_bio']-ds['CO2_background'])
    ds.insert(loc=1, column = 'CO_total_post', value= ds['CO_fire']+ds['CO_bio']-ds['CO_background'])

    plt.rcParams.update({'font.size':18})   
    plt.rcParams.update({'errorbar.capsize': 5})
    fig, (ax2, ax, ax3) = plt.subplots(nrows = 3, sharex = True, gridspec_kw={'height_ratios': [4,4,1.5]},figsize = (18,8))

    ds.plot(x='time', y='CO2_background',color = 'k', marker='.', markersize = 7,linestyle='None', ax=ax, label= 'Model background', legend = False)
    ds.plot(x='time', y='CO2_meas',marker='.',ax=ax, markersize = 7,linestyle='None',color = 'dimgrey',label= 'Measurement', legend = False)#yerr = 'measurement_uncertainty', 
    ds.plot(x = 'time', y = 'CO2_prior',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'salmon',label = 'Total model prior', legend = False)
    ds.plot(x = 'time', y = 'CO2_total_post',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'red',label = 'Total model posterior', legend = False)

    #ax.legend(markerscale = 2)
    #ax.set_xticks([datetime.datetime(year =2019, month = 12, day = 1),datetime.datetime(year =2019, month = 12, day = 5), datetime.datetime(year =2019, month = 12, day = 10), datetime.datetime(year =2019, month = 12, day = 15),
    #            datetime.datetime(year =2019, month = 12, day = 20), datetime.datetime(year =2019, month = 12, day = 25), datetime.datetime(year =2019, month = 12, day = 30), 
    #            ], 
    #            rotation=45)#datetime(year = 2020, month = 1, day = 4)
    #ax.set_xlim(left =  datetime.datetime(year = 2019, month = 11, day=30), right = datetime.datetime(year = 2019, month = 12, day=31, hour = 15))
    ax.grid(axis = 'both')
    ax.set_ylabel(r'CO$_2$ [ppm]', labelpad=6)
    #ax.errorbar(x= datetime.datetime(year =2019, month = 12, day = 31, hour = 4), y = y, yerr = ds['measurement_uncertainty'].mean(), marker = '.',markersize = 7,linestyle='None',color = 'dimgrey')
    plt.xlabel('date')

    ds.plot(x='time', y='CO_background',color = 'k', marker='.', markersize = 7,linestyle='None', ax=ax2, label= 'Model background')
    ds.plot(x='time', y='CO_meas',marker='.',ax=ax2, markersize = 7,linestyle='None',color = 'dimgrey',label= 'Measurement')#yerr = 'measurement_uncertainty', 
    ds.plot(x = 'time', y = 'CO_prior',marker = '.', ax = ax2, markersize = 7,linestyle='None',color = 'salmon',label = 'Total model prior')
    ds.plot(x = 'time', y = 'CO_total_post',marker = '.', ax = ax2, markersize = 7,color = 'red',linestyle='None',label = 'Total model posterior')
    ax2.grid(axis = 'both')
    ax2.set_ylabel('CO [ppb]', labelpad=6)
    

    myFmt = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(myFmt)
    #ax.set_xlim((datetime.datetime(year = 2019, month = 8, day = 30), datetime.datetime(year = 2020, month = 3, day = 3)))
    ax.set_xlim((datetime.datetime(year = 2019, month = 12, day = 1), datetime.datetime(year = 2019, month = 12, day = 31, hour = 23)))
    ## Rotate date labels automatically
    fig.autofmt_xdate()
    ds_mean = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/preparation/one_hour_runs/TCCON_mean_measurements_sept_to_march.pkl')
    #ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =9, day=1))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =2, day=28))]
    ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =12, day=1))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =1, day=1))]
    ax3.bar(ds_mean['datetime'], ds_mean['number_of_measurements'], width=0.1, color = 'dimgrey')
    ax3.set_ylabel('N', labelpad=17)
    ax3.grid(axis='x')
  
    plt.subplots_adjust(hspace=0)
    plt.savefig(savepath+'concentrations_plot_only_weekly_emissions_December.png', bbox_inches = 'tight', dpi = 300)#str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_'+molecule_name+'_total_concentrations_results_total_K_times_scaling.png', dpi = 300, bbox_inches = 'tight')

def plotting_diff_and_ak(ds,savepath):

    ds.insert(loc=1, column = 'CO2_meas_minus_total_post', value= ds['CO2_meas']-ds['CO2_fire']+ds['CO2_bio']-ds['CO2_background'])
    ds.insert(loc=1, column = 'CO_meas_minus_total_post', value= ds['CO_meas']-ds['CO_fire']+ds['CO_bio']-ds['CO_background'])

    plt.rcParams.update({'font.size':18})   
    plt.rcParams.update({'errorbar.capsize': 5})
    fig, (ax, ax2, ax3) = plt.subplots(nrows = 3, sharex = True, gridspec_kw={'height_ratios': [4,4,2]},figsize = (18,8))

    ds.plot(x='time', y='CO2_meas_minus_total_post',color = 'k', marker='.', markersize = 3,linestyle='None', ax=ax, legend = False)#, label= 'Model background')
    ax.grid(axis = 'both')
    ax.set_ylabel(r'$\Delta$CO$_2$ [ppm]', labelpad=6)
    #ax.errorbar(x= datetime.datetime(year =2019, month = 12, day = 31, hour = 4), y = y, yerr = ds['measurement_uncertainty'].mean(), marker = '.',markersize = 7,linestyle='None',color = 'dimgrey')
    plt.xlabel('date')

    ds.plot(x='time', y='CO_meas_minus_total_post',color = 'dimgrey', marker='.', markersize = 3,linestyle='None', ax=ax2, legend = False)
    ax2.grid(axis = 'both')
    ax2.set_ylabel(r'$\Delta$CO [ppb]', labelpad=6)
    
    ds.plot(x = 'time', y = 'akCO_max', ax = ax3,  label = 'AK CO')
    ds.plot(x = 'time', y = 'akCO2_max', ax = ax3, label = r'AK CO$_2$')

    myFmt = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(myFmt)

    ## Rotate date labels automatically
    fig.autofmt_xdate()
    #ds_mean = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/preparation/one_hour_runs/TCCON_mean_measurements_sept_to_march.pkl')
    #ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =9, day=1))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =2, day=28))]
    #ax3.bar(ds_mean['datetime'], ds_mean['number_of_measurements'], width=0.1, color = 'dimgrey')
    #ax3.set_ylabel('# measurements', labelpad=17)
    #ax3.grid(axis='x')
  
    plt.subplots_adjust(hspace=0)
    plt.savefig(savepath+'AK_max_and_diff_conc_weak_reg.png', bbox_inches = 'tight', dpi = 300)#str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_'+molecule_name+'_total_concentrations_results_total_K_times_scaling.png', dpi = 300, bbox_inches = 'tight')


def plot_total_conc_with_errors_linear_x_axis(df, savepath, alpha, alphaCO): 
    # plotting 
    df = df.sort_values(['time'], ascending = True).reset_index()
    df['date'] = pd.to_datetime(df['time'], format = '%Y-%M-%D').dt.month

    plt.rcParams.update({'font.size':18})   
    fig, ax = plt.subplots(3,1,sharex = True, gridspec_kw={'height_ratios': [4,4,1.5]},figsize=(16,10))
    #axis = np.arange(0,2*len(df['time'][:]),2)
    ax[0].set_ylabel(r'CO$_2$ [ppm]')
    ax[0].set_xlabel('date')
    #ax[0].set_xticks(Xaxis, ['']*len(df['time'][:]))
    #ax1.tick_params(axis = 'y')
    max_value = max(abs(df['CO2_fire'].max()), abs(df['CO2_fire'].min()))
    ax[0].set_ylim((406, max_value+2))

 
    #total CO2
    total_CO2 = df['CO2_fire']+df['CO2_bio']-df['CO2_background']
    ax[0].plot( df['time'], df['CO2_meas'],color = 'dimgrey',marker = '.',label = r'measurements')
    #prior CO2
    lns1 = ax[0].plot(df['time'],df['CO2_prior'],color = 'salmon', marker = '.', label = r'total prior')
    ax[0].fill_between(df['time'], df['CO2_prior']-(df['CO2_prior_std']), df['CO2_prior']+(df['CO2_prior_std']), color = 'salmon', alpha = 0.3)

    lns1 = ax[0].plot(df['time'],total_CO2,color = 'red',  marker = '.',label = r'total posterior')
    ax[0].fill_between(df['time'], total_CO2-(df['CO2fire_std']+df['CO2bio_std']), total_CO2+(df['CO2fire_std']+df['CO2bio_std']), color = 'red', alpha = 0.5)


    ## CO plot
    ax2 = ax[1]#.twinx()
    ax2.set_ylabel(r'CO [ppb]')
    max_value = max(abs(df['CO_fire'].max()), abs(df['CO_fire'].min()))
    ax2.set_ylim((0, max_value+100))
    # total CO
    total_CO = df['CO_fire']+df['CO_bio']-df['CO_background']
    ax2.plot(df['time'], df['CO_meas'],color = 'dimgrey',marker = '.',label = r'measurements')
    lns1 = ax2.plot(df['time'],df['CO_prior'],color = 'salmon',  marker = '.',label = r'total prior')
    ax2.fill_between(df['time'], df['CO_prior']-(df['CO_prior_std']), df['CO_prior']+(df['CO_prior_std']), color = 'salmon', alpha = 0.3)
    lns1 = ax2.plot(df['time'],total_CO,color = 'red', marker = '.', label = r'total posterior')
    ax2.fill_between(df['time'], total_CO-(df['COfire_std']+df['CObio_std']), total_CO+(df['COfire_std']+df['CObio_std']), color = 'red', alpha = 0.5)
    # prior CO 
    #total_prior_CO = df['CO_fire_prior']+df['CO_bio_prior']-df['CO_background']

    ax[1].legend(loc = 'upper left')

    ds_mean = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/preparation/one_hour_runs/TCCON_mean_measurements_sept_to_march.pkl')
    #ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =9, day=1))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =2, day=28))]
    ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =12, day=1))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =1, day=1))]
    ax[2].bar(ds_mean['datetime'], ds_mean['number_of_measurements'], width=0.1, color = 'dimgrey')
    ax[2].set_ylabel('N', labelpad=17)
    ax[2].grid(axis = 'x')
    ax2.grid(axis = 'both')
    ax[0].grid(axis = 'both')

    '''
    # ticks
    ticklabels = ['']*len(df['time'][:])
    # Every 4th ticklable shows the month and day
    ticklabels[0] = df['date'][0]
    reference = df['date'][0]
    for i in np.arange(1,len(df['time'][:])):
        if reference<12 and df['date'][i]>reference:
            ticklabels[i] = df['date'][i]
            reference = df['date'][i]
        elif reference == 12 and df['date'][i] == 1: 
            ticklabels[i] = df['date'][i]
            reference = df['date'][i]
    ax[1].set_xticks(Xaxis, ticklabels)
    ax[1].xaxis.set_major_formatter(mpl.ticker.FixedFormatter(ticklabels))
    plt.gcf().autofmt_xdate(rotation = 45)
    
    plt.axhline(y=0, color='k', linestyle='-')
    '''
    
    myFmt = DateFormatter("%Y-%m-%d")
    ax[0].xaxis.set_major_formatter(myFmt)
    ax[0].set_xlim((datetime.datetime(year = 2019, month = 12, day = 1), datetime.datetime(year = 2019, month = 12, day = 31, hour = 23)))
    fig.autofmt_xdate()
    plt.subplots_adjust(hspace=0)
    fig.savefig(savepath+"{:.2e}".format(alpha)+"_CO2_"+"{:.2e}".format(alphaCO)+"_CO_total_concentrations_with_errors_measurement_test.png", dpi = 300, bbox_inches = 'tight')


def plot_spatial_result(spatial_result, savepath, savename, cmap, vmax =None, vmin =None, cbar_kwargs = {'shrink':  0.835}, norm = None):
    '''
    for weekly spatial plotting of fluxes, saves plot 
    spatial_result : xarray to plot with latitude and longitude as coordinates
    molecule_name : Either "CO2" or "CO"
    savepath : path to save output image, must exist
    savename : Name of output image
    '''
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())  
    spatial_result.plot(x = 'longitude', y = 'latitude', ax = ax, cmap =cmap,vmin = vmin, vmax = vmax, norm = norm,
                            cbar_kwargs = cbar_kwargs)
    
    plt.scatter(x = 150.8793,y =-34.4061,color="black")
    ax.coastlines()
    #plt.title('Week '+str(week))
    plt.savefig(savepath+savename, bbox_inches = 'tight', dpi = 250)
    plt.close()



import numpy as np
basic_path = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/'
ds_tot = pd.DataFrame()
ak_CO2 = []
ak_CO = []
alphaCO2 = 1e-2
alphaCO = 2.12
for week in [35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,1, 2,3, 4, 5, 6,7,8,9]:# namen checken!!!!!!!!!!!!!!
    path = basic_path +'All_weeks/'+str(week)+'/'
    ds = pd.read_pickle(path+"{:.2e}".format(alphaCO2)+"_CO2_"+"{:.2e}".format(alphaCO)+'_concentrations_and_errors_only_one_week.pkl')
    ds.insert(loc = 1, column = 'akCO_mean', value = np.ones(ds['time'].shape[0])*xr.open_dataarray(path+"{:.2e}".format(alphaCO2)+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_ak_spatial__CO_fire_.nc').values.mean())
    ds.insert(loc = 1, column = 'akCO2_mean', value = np.ones(ds['time'].shape[0])*xr.open_dataarray(path+"{:.2e}".format(alphaCO2)+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_ak_spatial__CO2_fire_.nc').values.mean())
    ds.insert(loc = 1, column = 'akCO_max', value = np.ones(ds['time'].shape[0])*xr.open_dataarray(path+"{:.2e}".format(alphaCO2)+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_ak_spatial__CO_fire_.nc').values.max())
    ds.insert(loc = 1, column = 'akCO2_max', value = np.ones(ds['time'].shape[0])*xr.open_dataarray(path+"{:.2e}".format(alphaCO2)+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_ak_spatial__CO2_fire_.nc').values.max())
    print(ds)
    if week == 44: 
        ds = ds[ds['time']<= datetime.datetime(year = 2019, month = 11, day = 3)]
    ds_tot = pd.concat([ds_tot, ds])

#for week in [45]: 
#    path = basic_path +str(week)+'/'
#    ds = pd.read_pickle(path+'1.00e-04_CO2_1.00e-02_concentrations_and_errors.pkl')
#    ds.insert(loc = 1, column = 'akCO_mean', value = np.ones(ds['time'].shape[0])*xr.open_dataarray(path+'1.00e-02_CO2_2.12e+00_CO_ak_spatial__CO_fire_.nc').values.mean())
#    ds.insert(loc = 1, column = 'akCO2_mean', value = np.ones(ds['time'].shape[0])*xr.open_dataarray(path+'1.00e-02_CO2_2.12e+00_CO_ak_spatial__CO2_fire_.nc').values.mean())
#    ds_tot = pd.concat([ds_tot, ds])
#for week in [49,50,51,52]:
#    path = basic_path +str(week)+ '/'
#    ds = pd.read_pickle(path+'1.00e-02_CO2_2.12e+00_concentrations_and_errors.pkl')
#    ds.insert(loc = 1, column = 'akCO_mean', value = np.ones(ds['time'].shape[0])*xr.open_dataarray(path+'1.00e-02_CO2_2.12e+00_CO_ak_spatial__CO_fire_.nc').values.mean())
#    ds.insert(loc = 1, column = 'akCO2_mean', value = np.ones(ds['time'].shape[0])*xr.open_dataarray(path+'1.00e-02_CO2_2.12e+00_CO_ak_spatial__CO2_fire_.nc').values.mean())
#    ds_tot = pd.concat([ds_tot, ds])
ds_tot.reset_index(inplace=True)
print(ds_tot)
#plotting_diff_and_ak(ds_tot,basic_path+'All_weeks/Images/')

#plotting(ds_tot,basic_path+'All_weeks/Images/')
plot_total_conc_with_errors(ds_tot, basic_path+'All_weeks/Images/',alphaCO2, alphaCO)
#plot_total_conc_with_errors_linear_x_axis(ds_tot, basic_path+'All_weeks/Images/',alphaCO2, alphaCO)
path = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_100_CO_100/CO2_1_CO_1/Corr_0.7/'
tot_spatial_result =  xr.open_dataset(path+str(48)+'/spatial_results_CO2_fire_week_'+str(48)+'.nc')
for week in [49,50,51,52,1]:
    spatial_result = xr.open_dataset(path+str(week)+'/spatial_results_CO2_fire_week_'+str(week)+'.nc')
    tot_spatial_result.update({'__xarray_dataarray_variable__':(('longitude', 'latitude'),tot_spatial_result.__xarray_dataarray_variable__.values+ spatial_result.__xarray_dataarray_variable__.values )})
    #tot_spatial_result = tot_spatial_result + spatial_result
    print(tot_spatial_result)
plot_spatial_result(spatial_result.__xarray_dataarray_variable__, path+'total_emissions/', 'total_December_spatial_CO2_fire.png','seismic',vmax = 300, vmin = -300, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})#
    #, cmap, vmax =None, vmin =None, cbar_kwargs = {'shrink':  0.835}, norm = None)
#    print(tot_spatial_result)
    #plot_spatial_result(spatial_result, path+'total_emissions/', savename, cmap, vmax =None, vmin =None, cbar_kwargs = {'shrink':  0.835}, norm = None)
    #spatial_result = xr.open_dataset(path+str(week)+'/spatial_results_CO2_fire_week_'+str(week)+'.nc')
    #plot_spatial_result(spatial_result, path+'total_emissions/', savename, cmap, vmax =None, vmin =None, cbar_kwargs = {'shrink':  0.835}, norm = None)
    #spatial_result = xr.open_dataset(path+str(week)+'/spatial_results_CO_ant_bio_week_'+str(week)+'.nc')
    #plot_spatial_result(spatial_result, path+'total_emissions/', savename, cmap, vmax =None, vmin =None, cbar_kwargs = {'shrink':  0.835}, norm = None)
    #spatial_result = xr.open_dataset(path+str(week)+'/spatial_results_CO2_ant_bio_week_'+str(week)+'.nc')
    #plot_spatial_result(spatial_result, path+'total_emissions/', savename, cmap, vmax =None, vmin =None, cbar_kwargs = {'shrink':  0.835}, norm = None)
