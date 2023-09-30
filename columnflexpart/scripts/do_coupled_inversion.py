
from columnflexpart.classes.coupled_inversion import CoupledInversion #_no_corr_no_split_fluxes import CoupledInversion
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
from columnflexpart.scripts.plotting_routine_coupled_split_inversion import plot_spatial_flux_results_or_diff_to_prior, plot_averaging_kernel,plot_input_mask,plot_weekly_concentrations, plot_single_concentrations
from columnflexpart.scripts.plotting_routine_coupled_split_inversion import split_eco_flux, split_predictions,  plot_prior_spatially, plot_single_total_concentrations, calculate_and_plot_averaging_kernel, plot_l_curve, split_gridded_flux
from columnflexpart.scripts.plotting_routine_coupled_split_inversion import plot_emission_ratio, split_output, plot_spatial_result,plot_difference_of_posterior_concentrations_to_measurements

def do_everything(savepath, Inversion, molecule_name, mask_datapath_and_name, week_min, week_max, week_num):
    class_num = plot_input_mask(savepath,mask_datapath_and_name, selection = [12])
    #plot_prior_spatially(Inversion,molecule_name,week_min, week_max, savepath)
    #l1, l2 = find_two_optimal_lambdas(Inversion,[1e-7,10], 1e-14, err)# 1e-8

    #l1 = 1e-4#3.3e-3
    #l2 = 1e-3
    #l3 = 1e-2
    l4 = 1e-1
    #l5 = 1e1
    #l6 = 1e2
    #l7 = 1e3
    #l8 = 1e4
    #l9 = 1e5

    for l in [l4]:
        print('fitting')
        Inversion.fit(alpha = l)#, xerr = err) 
        #print(Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week])
        #print(Inversion.predictions_flat.values * Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week].squeeze())
        
        # save predictions
        pred = pd.DataFrame(data =Inversion.predictions_flat.values, columns = ['predictions'])
        pred.insert(loc = 1, column = 'prior_flux_eco', value = Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week].squeeze())
        pred.insert(loc = 1, column ='posterior_err_std', value = Inversion.prediction_errs_flat.values)
        pred.insert(loc = 1, column ='prior_err_cov', value = np.diagonal(Inversion.flux_errs_flat.values))
        pred.insert(loc = 1, column ='prior_err_std_with_lambda', value = np.sqrt(np.diagonal(Inversion.flux_errs_flat.values)/l))
        pred.to_pickle(savepath + str("{:e}".format(l))+'_predictions.pkl')
        
        Inversion.predictions_flat = Inversion.predictions_flat.rename(dict(new = 'bioclass'))
        #plot_l_curve(Inversion,err,molecule_name,savepath, l)
        predictions_list = split_output(Inversion.predictions_flat, 'bioclass')
        fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio = split_eco_flux(Inversion) ### change to use split_output too 
        post_std_list = split_output(Inversion.prediction_errs_flat, 'new')
        prior_std = xr.DataArray(np.sqrt(np.diagonal(Inversion.flux_errs_flat.values)/l),coords = [np.arange(0,len(np.diagonal(Inversion.flux_errs_flat.values)))], dims = 'new')
        prior_std_list = split_output(prior_std, 'new')

        flux_list = [fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio]
        name_list = ['CO2_fire', 'CO2_ant_bio', 'CO_fire', 'CO_ant_bio']  

        fluxCO2_fire_grid, fluxCO2_bio_grid, fluxCO_fire_grid, fluxCO_bio_grid = split_gridded_flux(Inversion)

        flux_list_grid = [fluxCO2_fire_grid, fluxCO2_bio_grid, fluxCO_fire_grid, fluxCO_bio_grid]
        name_list_grid = ['CO2_fire_grid', 'CO2_ant_bio_grid', 'CO_fire_grid', 'CO_ant_bio_grid']  
      
        plot_difference_of_posterior_concentrations_to_measurements(Inversion, savepath,l)

        for idx,predictions in enumerate(predictions_list):
            print('Plotting prior spatially')
            plot_prior_spatially(Inversion, flux_list[idx],name_list[idx], idx, savepath)
            #plot_prior_spatially(Inversion, flux_list_grid[idx],name_list_grid[idx], idx,savepath, False)

            print('Plotting spatial difference of results to prior')
            plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l, diff =False)
            
            print('Plotting spatial results')
            plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l, diff =True)

            print('Plot posterior std deviation')
            plot_spatial_result(Inversion.map_on_grid(post_std_list[idx]), savepath, str("{:e}".format(l))+'_post_std_'+name_list[idx]+'.png', 'pink_r', vmax =None, vmin =0, 
                                cbar_kwargs = {'shrink':  0.835, 'label' : r'posterior standard deviation'}, norm = None)

            #(prior variance(lambda considered) - posterior variance)/prior variance = (prior variance/lambda - post variance)/prior variance
            print('Plot error variance reduction/Uncertainty reduction')
            plot_spatial_result(Inversion.map_on_grid((prior_std_list[idx]-post_std_list[idx])/prior_std_list[idx]), savepath,
                                       str("{:e}".format(l))+'_uncertainty_reduction_'+name_list[idx]+'.png', 'bone_r', vmax =1, vmin =-0.25, 
                                cbar_kwargs = {'shrink':  0.835, 'label' : r'uncertainty reduction'}, norm = None)

            print('Plotting averaging kernels') # not yet working?! CHeck results!!!!!!!!!!!!!!!! 
            #plot_averaging_kernel(Inversion, l, class_num, week_num,savepath, plot_spatially=False)# class_num
            #plot_averaging_kernel(Inversion, l, class_num, week_num,savepath, plot_spatially=True,weekly = True)
            #plot_averaging_kernel(Inversion, l, class_num, week_num,savepath, plot_spatially=True)
        
        plot_emission_ratio(savepath, Inversion, predictions_list[0], predictions_list[2], l, fluxCO2_fire, fluxCO_fire)
        print('Plotting concentrations')
        total_CO2, total_CO = plot_single_concentrations(Inversion, l, savepath)
        #plot_single_concentrations(Inversion, predictionsCO2_fire, predictionsCO2_bio, fluxCO2_fire,fluxCO2_bio, 'CO2',l, savepath)
        CO2, CO = plot_single_total_concentrations(Inversion,l, savepath)

        #plot_single_total_concentrations(Inversion, predictionsCO_fire, predictionsCO_bio, fluxCO_fire, fluxCO_bio, 'CO',l, savepath)
                
        calculate_and_plot_averaging_kernel(Inversion,savepath,l)
        plot_l_curve(Inversion, molecule_name, savepath, l, err = None)
        
 
        #plot_single_concentrations(Inversion,  'CO',l, savepath)
        #plot_weekly_concentrations(Inversion,'CO',l, savepath) # ist ziemlich hartgecoded gerade für Dezember!!!!!!!!!!
        #plot_weekly_concentrations(Inversion,'CO2',l, savepath)


######################### adapt stuff from here on ####################################

savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/Working_Setup/CO2_200_CO_200/CO2_2_CO_6/Corr_0.995/'
mask = "/home/b/b382105/ColumnFLEXPART/resources/Ecosystems_AK_based_split_with_21_and_20_larger_and_4_split.nc"#OekomaskAU_Flexpart_version8_all1x1"#Ecosystems_AK_based_split_with_21_and_20_larger.nc"#OekomaskAU_Flexpart_version8_all1x1"#Ecosystems_AK_based_split_with_21_and_20_larger.nc"
non_equal_region_size = True


#Ecosystems, AK split
area_bioreg = np.array([8.95512935e+13, 4.27021383e+11, 4.01383356e+12, 1.51305732e+12,
 4.42067398e+11, 9.81395187e+09, 3.95111921e+10, 3.18183296e+10,
 5.52121748e+10, 5.55915961e+10, 5.41460509e+10, 4.29887311e+10,
 3.13764949e+10, 1.05335427e+10, 9.31980461e+10, 5.11758810e+10,
 4.00158388e+10, 2.99495636e+10, 3.91250878e+10, 4.90667320e+10,
 1.95434996e+11, 6.07622615e+10, 6.98377101e+10, 1.77802351e+11,
1.95969571e+11])


#Ecosystemes, AK_split with 21 and 20 larger 
area_bioreg = np.array([8.95512935e+13, 4.27021383e+11, 4.01383356e+12, 1.51305732e+12,
 4.42067398e+11, 9.81395187e+09, 3.95111921e+10, 3.18183296e+10,
 5.52121748e+10, 5.55915961e+10, 5.41460509e+10, 4.29887311e+10,
 3.13764949e+10, 1.05335427e+10, 9.31980461e+10, 5.11758810e+10,
 4.00158388e+10, 2.99495636e+10, 3.91250878e+10, 4.90667320e+10,
 1.54614599e+11, 9.65077342e+10, 6.07622615e+10, 6.98377101e+10,
 1.77802351e+11, 1.95969571e+11])
#area_bioreg = np.concatenate([area_bioreg, area_bioreg])
#area bioreg for reion 4 split 21 and 20 larger: 
area_bioreg = [8.95512935e+13, 4.27021383e+11, 4.01383356e+12, 1.45736998e+12,
 2.15745316e+11, 9.81395187e+09, 3.95111921e+10, 3.18183296e+10,
 5.52121748e+10, 5.55915961e+10, 5.41460509e+10, 4.29887311e+10,
 3.13764949e+10, 1.05335427e+10, 9.31980461e+10, 5.11758810e+10,
 4.00158388e+10, 2.99495636e+10, 3.91250878e+10, 4.90667320e+10,
 1.54614599e+11, 9.65077342e+10, 6.07622615e+10, 6.98377101e+10,
 1.77802351e+11, 1.95969571e+11, 2.26322082e+11]


Inversion = CoupledInversion(
    result_pathCO = "/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions3_CO.pkl" ,
    result_pathCO2 = "/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions.pkl" ,
    flux_pathCO = "/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/Regridded_1x1/", 
    flux_pathCO2 = "/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Flux/CT2022.flux1x1.",
    bioclass_path = mask,
    month = '2019-12',
    date_min = datetime.datetime(year = 2019, month = 12, day = 23),
    area_bioreg = area_bioreg,
    week = 52,
    non_equal_region_size=non_equal_region_size,
    date_max = datetime.datetime(year= 2020, month = 1, day = 7),
    boundary=[110.0, 155.0, -45.0, -10.0], 
       ) 
print('Initialization done')


#print(Inversion.gridded_mask)
#img = plt.imshow(Inversion.gridded_mask)
#bar = plt.colorbar(img)
#plt.savefig('/home/b/b382105/test/ColumnFLEXPART/columnflexpart/scripts/gridded_mask.png')
#predictions = Inversion.fit(alpha = 5e-4)
#plot_l_curve(Inversion, 'CO', savepath, 5e-4, err = None)
#print(Inversion.compute_l_curve(alpha = [5e-4, 5e-3, 5e-2]))
#print(Inversion.get_averaging_kernel())
#print(Inversion.get_averaging_kernel().shape)
#Inversion.multiply_H_and_F()
#predictions = Inversion.fit()
#Inversion.predictions_flat.to_netcdf('/home/b/b382105/test/ColumnFLEXPART/columnflexpart/scripts/coupled_predictions.nc')
#print(Inversion.bioclass_mask.values.max())
#Inversion.get_grid_cell_indices_for_final_regions()
#Inversion.get_prior_covariace_matrix(non_equal_region_size, area_bioreg)

#err = Inversion.get_prior_flux_errors( non_equal_region_size, area_bioreg)

do_everything(savepath, Inversion,'CO',mask, 52, 52, 6)
#do_everything(savepath, Inversion, 'CO',mask, 1, 1, 6, err)

#FAKTOR 10**-9 und 10**-6 der jetzt bei fit davor passiert ist, habe ich erstmal rausgenommen. Muss ich bei Fit vermutlich nohc einfügen!!!!!!!!!!!!!!!!!