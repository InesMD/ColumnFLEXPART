
from columnflexpart.scripts.coupled_inversion import CoupledInversion
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def plot_spatial_flux_results_or_diff_to_prior(savepath,  Inversion,molecule_name, week_min, week_max,alpha,vminv=None, diff =False):
    factor = 12*10**6
    #total_spatial_result = xr.Dataset()
    plt.rcParams.update({'font.size':15})   
    predictionsCO2 = Inversion.predictions[:,Inversion.predictions['bioclass']<=Inversion.predictions['bioclass'].shape[0]/2]
    predictionsCO = Inversion.predictions[:,Inversion.predictions['bioclass']>Inversion.predictions['bioclass'].shape[0]/2]
    predictionsCO = predictionsCO.assign_coords(bioclass = np.arange(0,len(predictionsCO['bioclass'])))
    print(predictionsCO)
    for week in range(week_min,week_max+1): 
        plt.figure()
        print(Inversion.predictions)
        spatial_result = Inversion.map_on_grid(predictionsCO2[(predictionsCO2['week']==week)])
        spatial_resultCO = Inversion.map_on_grid(predictionsCO[(predictionsCO['week']==week)])
        print(spatial_resultCO)
        ax = plt.axes(projection=ccrs.PlateCarree())  
        if diff== True: 
            spatial_flux = Inversion.map_on_grid(Inversion.flux[:,Inversion.flux['week']==week])
            spatial_flux = spatial_flux *factor
            spatial_result = (spatial_result*factor)-spatial_flux
            spatial_result.plot(x = 'longitude', y = 'latitude',ax = ax, cmap = 'seismic',vmin = -10, vmax = 10, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':0.835})
            plt.scatter(x = 150.8793,y =-34.4061,color="black")
            ax.coastlines()
            #plt.title('Week '+str(week))
            plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'Diff_to_prior_week_'+str(week)+'.png', bbox_inches = 'tight')
        else: 
            spatial_resultCO = (spatial_resultCO*factor)
            spatial_result = (spatial_result*factor)
            print(spatial_result)
            #spatial_result = spatial_result.where(spatial_result.values>0.5)
            spatial_result.plot(x = 'longitude', y = 'latitude',ax = ax, cmap = 'seismic',vmin = -250, vmax = 250, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
            plt.scatter(x = 150.8793,y =-34.4061,color="black")
            ax.coastlines()
            #plt.title('Week '+str(week))
            plt.savefig(savepath+str("{:e}".format(alpha))+'_Spatial_results_CO2_week_'+str(week)+'.png', bbox_inches = 'tight', dpi = 450)
            plt.close()
            plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())  
            spatial_resultCO.plot(x = 'longitude', y = 'latitude',ax = ax, cmap = 'seismic',vmin = -10, vmax = 10, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
            plt.scatter(x = 150.8793,y =-34.4061,color="black")
            ax.coastlines()
            #plt.title('Week '+str(week))
            plt.savefig(savepath+str("{:e}".format(alpha))+'_Spatial_results_CO_week_'+str(week)+'.png', bbox_inches = 'tight', dpi = 450)
            #spatial_result.to_netcdf(path =savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_spatial_results_week_'+str(week)+'.nc')
        plt.close()
    #total_spatial_result.to_netcdf(path =savepath+'spatial_results_week_'+str(week_min)+'_'+str(week_max)+'.pkl')



def do_everything(savepath, Inversion, molecule_name, mask_datapath_and_name, week_min, week_max, week_num, 
                  err):
    #class_num = plot_input_mask(savepath,mask_datapath_and_name)
    #plot_prior_spatially(Inversion,molecule_name,week_min, week_max, savepath)
    #l1, l2 = find_two_optimal_lambdas(Inversion,[1e-7,10], 1e-14, err)# 1e-8
    #print(l1)
    l3 = 1e-5
    for l in [l3]:#1, l2]:#,l2,l3]:
        predictions = Inversion.fit(alpha = l, xerr = err) 
        #plot_l_curve(Inversion,err,molecule_name,savepath, l)
        print('Plotting spatial results')
        #plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion,molecule_name, week_min, week_max,l,vminv=None, diff =True)
       # print('Plotting spatial difference of results to prior')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, molecule_name,week_min, week_max,l,vminv=None, diff =False)
        #print('Plotting averaging kernels')
        #plot_averaging_kernel(Inversion,molecule_name, l, class_num, week_num,savepath, plot_spatially=False)# class_num
        #plot_averaging_kernel(Inversion,molecule_name, l, class_num, week_num,savepath, plot_spatially=True,weekly = True)
        #plot_averaging_kernel(Inversion,molecule_name, l, class_num, week_num,savepath, plot_spatially=True)
        #print('Plotting concentrations')
        #calc_concentrations(Inversion,  'CO',l, savepath)
        #plot_weekly_concentrations(Inversion,'CO',l, savepath)


def calc_errors_flat_area_weighted_scaled_to_mean_flux(flux_mean, area_bioreg): 

    flat_errors = np.ones((len(flux_mean.bioclass.values), len(flux_mean.week.values)))
    area_bioreg[0] = area_bioreg[0]*10000000 # ocean smaller
    final_error = np.ones(flat_errors.shape)
    for w in range(len(flux_mean.week.values)): 
        area_weighted_errors = flat_errors[:,w]/area_bioreg
        scaling_factor = flux_mean[1:, w].mean()/area_weighted_errors[1:].mean()
        final_error[:,w] = scaling_factor.values*area_weighted_errors

    err_scaled = xr.DataArray(data=final_error, coords=dict({ 'bioclass': ('bioclass',flux_mean.bioclass.values),# [0,1,2,3,4,5,6]),
                                                                'week': ('week',flux_mean.week.values)}))
    return err_scaled


######################### adapt stuff from here on ####################################

savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/sectors_not_splitted_no_correlation/12/Gridded/'#Setup_AK_based2/'
mask = "/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1"#Ecosystems_AK_based_split_with_21_and_20_larger.nc"
non_equal_region_size = False
#Ecosystemes, AK_split with 21 and 20 larger 
area_bioreg = np.array([8.95512935e+13, 4.27021383e+11, 4.01383356e+12, 1.51305732e+12,
 4.42067398e+11, 9.81395187e+09, 3.95111921e+10, 3.18183296e+10,
 5.52121748e+10, 5.55915961e+10, 5.41460509e+10, 4.29887311e+10,
 3.13764949e+10, 1.05335427e+10, 9.31980461e+10, 5.11758810e+10,
 4.00158388e+10, 2.99495636e+10, 3.91250878e+10, 4.90667320e+10,
 1.54614599e+11, 9.65077342e+10, 6.07622615e+10, 6.98377101e+10,
 1.77802351e+11, 1.95969571e+11])

area_bioreg = np.concatenate([area_bioreg, area_bioreg])

Inversion = CoupledInversion(
    result_pathCO = "/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions3_CO.pkl" ,
    result_pathCO2 = "/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions.pkl" ,
    flux_pathCO = "/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/Regridded_1x1/", 
    flux_pathCO2 = "/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Flux/CT2022.flux1x1.",
    bioclass_path = mask,
    month = '2019-12',
    date_min = datetime.datetime(year = 2019, month = 12, day = 1),
    date_max = datetime.datetime(year= 2020, month = 1, day = 7),
    boundary=[110.0, 155.0, -45.0, -10.0], 
       ) 

# prior errors
if non_equal_region_size == True: 
    print('per region')
    err = calc_errors_flat_area_weighted_scaled_to_mean_flux(Inversion.flux, area_bioreg) # adapt!!!!!!!!!!!!!!!!!!!!!
    print('maximum error value: '+str(err.max()))
else: 
    print('not per region')
    err = Inversion.get_land_ocean_error(1/100000) # adapt CO2 and CO errors separately check 
    print(err)
    print('maximum error value: '+str(err.max()))
print('Initlaizing done')

do_everything(savepath, Inversion,'CO',mask, 52, 52, 6, err)
#do_everything(savepath, Inversion, 'CO',mask, 1, 1, 6, err)

#FAKTOR 10**-9 und 10**-6 der jetzt bei fit davor passiert ist, habe ich erstmal rausgenommen. Muss ich bei Fit vermutlich nohc einfügen!!!!!!!!!!!!!!!!!