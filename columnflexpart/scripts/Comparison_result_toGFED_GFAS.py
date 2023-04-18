import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime 
import numpy as np 
def create_mask_ak_and_co_based(alpha, datapathCO, datapathCO2 , week, wnum):
    akco = xr.open_dataset(datapathCO+'test/'+str("{:e}".format(alpha))+'_CO_ak_CO_spatially_week_'+str(week)+'.nc')
    akco2 = xr.open_dataset(datapathCO2+str("{:e}".format(alpha))+'_CO2_ak_CO2_spatially_week_'+str(week)+'.nc')
    
    ds_co = xr.open_dataset(datapathCO+'test/'+str("{:e}".format(alpha))+'_CO_spatial_results_week_'+str(week)+'.nc')
    ds_co2 = xr.open_dataset(datapathCO2+'{:e}'.format(alpha)+'_CO2_spatial_results_week_'+str(week)+'.nc')
    ds_co_mask = xr.where((akco2>1e-3)&(akco>1e-3)&(ds_co.__xarray_dataarray_variable__>0.5),1, False)#&(akco2>1e-3)&(ds_co.__xarray_dataarray_variable__.values>0.5),1, False)
    print(ds_co_mask)
    ds_co_mask.to_netcdf(datapathCO+'test/mask_co_co2_for_GFED_GFAS_week_'+str(week)+'.nc')
    return ds_co_mask

def calculate_united_mask(datapathmasks):
    total_mask = xr.open_dataset(datapathmasks+'mask_co_co2_for_GFED_GFAS_week_48.nc')
    
    for week in [49,50,51,52,1]:
        mask = xr.open_dataset(datapathmasks+'mask_co_co2_for_GFED_GFAS_week_'+str(week)+'.nc')
        #print(mask)
        total_mask = xr.where((total_mask== 1)|(mask==1), 1, False)
    return total_mask


def multiply_masks_with_GFED(datapathmasks,year, month, molecule,datapath_and_name_GFED, mask_monthly = False,just_land_mask = False): 
    if molecule == 'CO': 
        factor = 12/28
    elif molecule =='CO2': 
        factor = 12/44
    GFED = pd.read_pickle(datapath_and_name_GFED)
    GFED.rename(columns ={'Long': 'longitude', 'Lat':'latitude'}, inplace = True)
    GFED = GFED[(GFED['Year']==year)&(GFED['longitude']>=110)&(GFED['latitude']<=-10)&(GFED['latitude']>=-45)&(GFED['longitude']<=155)&(GFED['Month']==month)]#stimmt mit Literaturwert überein 
    print(GFED.total_emission.sum())
    xrGFED = GFED.set_index(['longitude', 'latitude']).to_xarray()
    print(xrGFED)
    if mask_monthly == True: 
        total_mask = calculate_united_mask(datapathmasks)
        print(total_mask)
        maskedGFED = xrGFED.total_emission*total_mask.__xarray_dataarray_variable__*factor
    else: 
        weekly_mean_values = []
        if just_land_mask == False: 
            for week in [48,49,50,51,52,1]:
                mask = xr.open_dataset(datapathmasks+'mask_co_co2_for_GFED_GFAS_week_'+str(week)+'.nc')
                maskedGFED = xrGFED.total_emission*mask.__xarray_dataarray_variable__
                if week == 48: 
                    weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(1/31))
                elif week == 1: 
                    weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(2/31))
                else: 
                    weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(7/31))
        else: 
            for week in [48,49,50,51,52,1]:
                mask = xr.open_dataset(datapathmasks+'Mask_AU_land_bioclasses_based.nc')
                mask = mask.rename({'Long': 'longitude', 'Lat':'latitude'})
                maskedGFED = xrGFED.total_emission#*mask.bioclass
                if week == 48: 
                    weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(1/31))
                elif week == 1: 
                    weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(2/31))
                else: 
                    weekly_mean_values.append(maskedGFED.sum(['latitude', 'longitude'])*factor*(7/31))

        print(weekly_mean_values)# CO: [0.04749974,0,0.40146725,0.64328656,1.09471446,0.04122866] - total: 2.2281966650221396]
        # CO2: [0.57445635,0,5.06674055,7.90527242,13.35645642,0.51227383] - total : 27.415199568118155
        print(np.array(weekly_mean_values).sum())



    #maskedGFED = xrGFED.total_emission*total_mask.__xarray_dataarray_variable__
    #print(maskedGFED.sum(['latitude', 'longitude'])) # in TgC/month #62.94144557 für CO2 #5.07877246 für CO
    #maskedGFED.plot(x = 'longitude', y='latitude')
    #plt.savefig('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/test/masked_GFED_'+molecule+'CO_total_emission.png')
    #total_mask.__xarray_dataarray_variable__.plot(x = 'longitude', y='latitude')
    #plt.savefig('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/test/total_mask.png')
        

def multiply_masks_with_GFAS(datapathmasks,year, month, molecule,datapath_and_name_GFAS, just_land_mask = False): 
    if molecule == 'CO': 
        factor = 12/28
        variable = 'cofire'
    elif molecule =='CO2': 
        factor = 12/44
        variable = 'co2fire'
    GFAS = pd.read_pickle(datapath_and_name_GFAS)
    GFAS = GFAS[(GFAS['longitude']>=110)&(GFAS['latitude']<=-10)&(GFAS['latitude']>=-45)&(GFAS['longitude']<=155)]
    GFAS = GFAS[(pd.to_datetime(GFAS['date'])>=datetime(year= 2019, month = 12, day=1))&(pd.to_datetime(GFAS['date'])<=datetime(year= 2020, month = 1, day=1))]
    print(GFAS)
    days_in_week = dict({'48': [1,2], '49': [2,3,4,5,6,7,8,9], '50': [9,10,11,12,13,14,15,16], '51': [16,17,18,19,20,21,22,23], '52':[23,24,25,26,27,28,29,30], '1':[30,31,32]})
    weekly_mean_values = []
    for week in [48,49,50,51,52]:
        GFAS_week = GFAS[(pd.to_datetime(GFAS['date'])>=datetime(year= 2019, month = 12, day=days_in_week[str(week)][0]))]
        GFAS_week= GFAS_week[(pd.to_datetime(GFAS_week['date'])<=datetime(year= 2019, month = 12, day=days_in_week[str(week)][-1]))]
        GFAS_week = GFAS_week.set_index(['date','latitude','longitude']).to_xarray()
        if just_land_mask == False: 
            mask = xr.open_dataset(datapathmasks+'mask_co_co2_for_GFED_GFAS_week_'+str(week)+'.nc')
            maskedGFAS = GFAS_week[variable]*mask.__xarray_dataarray_variable__
        else: 
            mask = xr.open_dataset(datapathmasks+'Mask_AU_land_bioclasses_based.nc')
            maskedGFAS = GFAS_week[variable]*mask.bioclass

        #GFAS_week = GFAS_week.set_index(['longitude','latitude']).to_xarray()
        #
        if week == 48: 
            weekly_mean_values.append(maskedGFAS.mean()*7692024*10**6*10**(-9)*factor*(1*24*60*60))
        else: 
            weekly_mean_values.append(maskedGFAS.mean()*7692024*10**6*10**(-9)*factor*(7*24*60*60))

    for week in [1]: 
        GFAS_week = GFAS[(pd.to_datetime(GFAS['date'])>=datetime(year= 2019, month = 12, day=days_in_week[str(week)][0]))]
        GFAS_week = GFAS_week[(pd.to_datetime(GFAS_week['date'])<=datetime(year= 2020, month = 1, day=1))]
        GFAS_week = GFAS_week.set_index(['date','latitude','longitude']).to_xarray()
        if just_land_mask == False: 
            mask = xr.open_dataset(datapathmasks+'mask_co_co2_for_GFED_GFAS_week_'+str(week)+'.nc')
            maskedGFAS = GFAS_week[variable]*mask.__xarray_dataarray_variable__
        else: 
            mask = xr.open_dataset(datapathmasks+'Mask_AU_land_bioclasses_based.nc')
            maskedGFAS = GFAS_week[variable]*mask.bioclass
        #maskedGFAS = GFAS_week.co2fire*mask.__xarray_dataarray_variable__
        
        weekly_mean_values.append(maskedGFAS.mean()*7692024*10**6*10**(-9)*factor*(2*24*60*60))
    print(weekly_mean_values)# für CO: [0.0035973,0.,0.08051799,0.39373375,0.15907385,0.02576907] # TgC/month
        # für CO2: [0.03412136,0,1.19239929,4.94659389,1.62545039,0.24486561]
    #plt.plot([48,49,50,51,52,1],weekly_mean_values)
    #plt.savefig(datapathmasks+'GFAS_CO_weekly_time.png')


datapathCO = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/'
#multiply_masks_with_GFED(datapathCO+'test/',2019, 12,'CO2','/work/bb1170/RUN/b382105/Dataframes/GFEDs/GDF_CO2_AU20192020_regr1x1.pkl', just_land_mask=True)
#multiply_masks_with_GFED(datapathCO+'test/',2019, 12,'CO','/work/bb1170/RUN/b382105/Dataframes/GFEDs/GDF_CO_AU20192020_regr1x1.pkl', just_land_mask=True)

#multiply_masks_with_GFAS(datapathCO+'test/',2019, 12,'CO2','/work/bb1170/RUN/b382105/Dataframes/GFAS/GDF_Test2_GFAS_AU_2009_2020.pkl')
multiply_masks_with_GFAS(datapathCO+'test/',2019, 12,'CO','/work/bb1170/RUN/b382105/Dataframes/GFAS/GDF_CO_Test2_GFAS_AU_2019_2020.pkl', just_land_mask=True)
#for creating weekly masks:
'''
for wnum, week in enumerate([48,49,50,51,52,1]):
    mask = create_mask_ak_and_co_based(1e-5, '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/', 
                              '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/', week, wnum)
    plt.figure()
    mask.__xarray_dataarray_variable__.plot(x = 'longitude', y = 'latitude')
    plt.savefig('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/test/mask_week_'+str(week)+'.png')
    plt.close()
    '''