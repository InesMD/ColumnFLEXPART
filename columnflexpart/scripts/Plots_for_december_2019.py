import pandas as pd 
from columnflexpart.classes.coupled_inversion import CoupledInversion 
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import xarray as xr

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
    plt.savefig(savepath+str("{:.2e}".format(alpha))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_'+molecule_name+'_total_concentrations_results_total_K_times_scaling_December.png', dpi = 300, bbox_inches = 'tight')


plot_total_concentrations(conc_tot, prior_tot, ds, savepath, alpha, alphaCO,'CO')
plot_total_concentrations(conc_tot, prior_tot, ds, savepath, alpha, alphaCO,'CO2')