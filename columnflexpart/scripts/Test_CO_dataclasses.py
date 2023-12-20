from columnflexpart.classes.flexdatasetCO import FlexDatasetCO
import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
datapath_predictions = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/'
ds = pd.read_pickle(datapath_predictions+'predictions3_CO.pkl')
print(ds)
print(ds['enhancement'])
print(ds['background'])
print(ds['xco2_measurement'])
print(ds['measurement_uncertainty'].mean())
'''
ds48 = ds[ds['time']<datetime(year=2019, month = 12, day=2)]
ds49 = ds[(ds['time']>datetime(year=2019, month = 12, day=1, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=9))]
ds50 = ds[(ds['time']>datetime(year=2019, month = 12, day=8, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=16))]
ds51 = ds[(ds['time']>datetime(year=2019, month = 12, day=15, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=23))]
ds52 = ds[(ds['time']>datetime(year=2019, month = 12, day=22, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=30))]
ds1 = ds[(ds['time']>datetime(year=2019, month = 12, day=29, hour = 23))&(ds['time']<datetime(year=2020, month = 1, day=6))]


background = []
measurement = []
prior = []
conc = []
for ds in [ds48, ds49, ds50, ds51, ds52, ds1]: 
    background.append(ds['background_inter'].mean())
    measurement.append(ds['xco2_measurement'].mean())
    prior.append(ds['xco2_inter'].mean())
#for df in [df48, df49, df50, df51, df52, df1]: 
#    conc.append(df['conc'].mean())
print(background)
print(prior)
'''

#ds = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20191220_20191221/predictions_CO_GFED_regr.pkl')
#ds0 = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20191223_20191228/predictions_CO_GFED_regr.pkl')
#ds1 = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20191231_20191231/predictions_CO_GFED_regr.pkl')
#ds2 = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20191214_20191217/predictions_CO_GFED_regr.pkl')
#ds3 = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20191201_20191210/predictions_CO_GFED_regr.pkl')
#ds4 = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20200101/predictions_CO_GFED_regr.pkl')
#ds5 = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20200107/predictions_CO_GFED_regr.pkl')
#ds6 = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20191129/predictions_CO_GFED_regr.pkl')
#ds7 = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20191125_20191126/predictions_CO_GFED_regr.pkl')

#ds_tot = pd.concat([ds, ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7])
#ds_tot.to_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions_CO_GFED_regr.pkl')
#ds = pd.read_pickle("/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions_CO_GFED_regr.pkl")
#print(ds)
#print(ds['xco2_inter'])
#print(ds.keys())
#print(ds['time'].values.max())
#print(ds['time'].values.min())
#print(ds.time.values)
#datapath = '/work/bb1170/RUN/b382105/Dataframes/GFEDs/'
#from columnflexpart.utils.utilsCO import load_cams_data
#df = load_cams_data('/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/Regridded_1x1/', datetime(year=2019, month=12, day = 1), datetime(year=2020, month = 1, day = 31))
#print(df['latitude'])
#print(df)
#gdf = pd.read_pickle(datapath+'DF_regridded_CO_AU20192020.pkl')
#gdf = xr.open_dataset("/work/bb1170/RUN/b382105/Dataframes/GFEDs/DF_CO_GFED_cut_to_AU.nc")
#gdf = xr.open_dataset("/work/bb1170/RUN/b382105/Dataframes/GFEDs/GFED2_2019_2020_regr1x1_date_total_emission.nc")
###print(gdf)
gdf = gdf[(gdf['Year']==2019)&(gdf['Month']==12)]
gdf = gdf[(gdf['Longround']>110)&(gdf['Lat']<-10)&(gdf['Lat']>-50)&(gdf['Longround']<155)]
gdf = gdf.set_index(['Longround', 'Lat']).to_xarray()
gdf['emission'][:].plot(x = 'Longround', y='Lat')#x='Longround', y = 'Lat')
#gdf['total_emission'][11,:,:].plot(x='longitude', y='latitude')
plt.savefig('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/GFED_old_regridded')

#ds = xr.open_dataset('/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/CAMS-GLOB-BIO_v3.1_carbon-monoxide_2019.nc', decode_times = False)#, drop_variables="time_components")
#print(ds)
#print(ds['emiss_bio'])
from columnflexpart.utils.utilsCO import load_GFED_data

#load_GFED_data(datetime(year=2019, month =12, day= 15), datetime(year=2020, month = 1, day = 15))

def plot_predictions(ds):
    ds48 = ds[ds['time']<datetime(year=2019, month = 12, day=2)]
    ds49 = ds[(ds['time']>datetime(year=2019, month = 12, day=1, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=9))]
    ds50 = ds[(ds['time']>datetime(year=2019, month = 12, day=8, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=16))]
    ds51 = ds[(ds['time']>datetime(year=2019, month = 12, day=15, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=23))]
    ds52 = ds[(ds['time']>datetime(year=2019, month = 12, day=22, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=30))]
    ds1 = ds[(ds['time']>datetime(year=2019, month = 12, day=29, hour = 23))&(ds['time']<datetime(year=2020, month = 1, day=6))]

    background = []
    measurement = []
    prior = []
    conc = []
    for ds in [ds48, ds49, ds50, ds51, ds52, ds1]: 
        background.append(ds['background_inter'].mean())
        measurement.append(ds['xco2_measurement'].mean())
        prior.append(ds['xco2_inter'].mean())
    #for df in [df48, df49, df50, df51, df52, df1]: 
    #    conc.append(df['conc'].mean())

    weeks = [48, 49, 50, 51, 52, 53]
    plt.rcParams.update({'font.size':14})   
    plt.rcParams.update({'errorbar.capsize': 5})
    #fig, (ax, ax2) = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'height_ratios': [4, 1]},figsize = (14,10))
    fig, ax = plt.subplots(1,1, figsize = (14,10))#plt.figure(figsize = (14,10))
    plt.plot(weeks, background, marker = 'o', label = 'Background')
    plt.plot(weeks, measurement, marker = 'o', label = 'Measurement')
    plt.plot(weeks, prior,  marker = 'o',label = 'Prior')
    plt.legend()
    plt.xticks([48, 49, 50, 51, 52, 53])
    ax.set_xticklabels(['48', '49', '50', '51', '52', '1'])
    plt.xlim((47.5, 53.5))
    #plt.xlim(left =  datetime(year = 2019, month = 11, day=30), right = datetime(year = 2020, month = 1, day=8))
    plt.grid(axis = 'both')
    plt.ylabel('concentration [ppb]')
    #plt.errorbar(x= datetime(year =2020, month = 1, day = 7), y = 407.5, yerr = ds['measurement_uncertainty'].mean(), marker = '.',markersize = 7,linestyle='None',color = 'dimgrey')
    plt.xlabel('weeks')
    plt.title('Weekly mean concentration')
    plt.savefig('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/predictions_weekly.png')
    #plt.plot(weeks, conc,  marker = 'o',label= 'Posterior')



#plot_predictions(ds)

'''
fd = FlexDatasetCO('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20191220_20191221/RELEASES_0_5/' , ct_dir='/work/bb1170/RUN/b382105/Data/CAMS/Concentration/regridded_1x1/', ct_name_dummy='something', chunks=dict(time=20, pointspec=4))
fd.load_measurement('/work/bb1170/RUN/b382105/Data/TCCON_wg_data/wg20080626_20200630.public.nc', 'TCCON')
tr = fd.trajectories
print(tr)
#tr.load_cams_data()
tr.ct_endpoints(boundary = [110, 155, -45, -10])
#tr.ct_endpoints(boundary=args.boundary)
tr.co_from_endpoints(boundary=[110, 155, -45, -10])
tr.save_endpoints()
#ds = xr.open_dataset('/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Flux/CT2022.flux1x1.20200331.nc', drop_variables="time_components")
#print(ds.attrs)
#print(ds.time)

#tr.load_endpoints()
flux_path = '/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/Regridded_1x1/'
#enhancement_inter = fd.enhancement(ct_file=args.flux_file, boundary=args.boundary, allow_read=args.read_only, interpolate=True)
enhancement = fd.enhancement(ct_file=flux_path, boundary=[110, 170, -45,-10], allow_read=False, interpolate=True)
#fd.background(allow_read=False, boundary=[110, 155, -45,-10], interpolate=True)
#xco2 = fd.total(ct_file=flux_path, allow_read=False, boundary=[110, 155, -45,-10], chunks=dict(time=20, pointspec=4), interpolate=False)
#print(xco2)
'''
'''
from columnflexpart.classes.flexdataset import FlexDataset
fd = FlexDataset('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/20191220_20191221/RELEASES_0_5/' , 
                 ct_dir='/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Concentration/', 
                 ct_name_dummy='CT2022.molefrac_glb3x2_', chunks=dict(time=20, pointspec=4))
fd.load_measurement('/work/bb1170/RUN/b382105/Data/TCCON_wg_data/wg20080626_20200630.public.nc', 'TCCON')
tr = fd.trajectories
tr.load_ct_data()
tr.ct_endpoints(boundary = [110, 170, -45, -10])
enhancement = fd.enhancement(ct_file='/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Flux/CT2022.flux1x1.',
                              boundary=[110, 170, -45,-10], allow_read=False, interpolate=False)
'''