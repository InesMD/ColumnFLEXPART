from columnflexpart.classes.coupled_inversion import CoupledInversion #_no_corr_no_split_fluxes import CoupledInversion
import datetime
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import pickle
import netCDF4
import pandas as pd
import cartopy.crs as ccrs
from columnflexpart.scripts.plotting_routine_coupled_split_inversion import plot_spatial_flux_results_or_diff_to_prior, plot_averaging_kernel,plot_input_mask,plot_weekly_concentrations, plot_single_concentrations
from columnflexpart.scripts.plotting_routine_coupled_split_inversion import split_eco_flux, split_predictions,  plot_prior_spatially, plot_single_total_concentrations, calculate_and_plot_averaging_kernel, plot_l_curve, split_gridded_flux
from columnflexpart.scripts.plotting_routine_coupled_split_inversion import plot_emission_ratio, split_output, plot_spatial_result,plot_difference_of_posterior_concentrations_to_measurements, calculate_ratio_error
from columnflexpart.scripts.plotting_routine_coupled_split_inversion import calc_single_conc_by_multiplication_with_K, plot_single_concentrations_measurements_and_errors, create_mask_based_on_regions


def save_predictions(savepath, l, lCO): 
    print('Saving predictions')
    pred = pd.DataFrame(data =Inversion.predictions_flat.values, columns = ['predictions'])
    pred.insert(loc = 1, column = 'prior_flux_eco', value = Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week].squeeze())
    pred.insert(loc = 1, column ='posterior_err_std', value = Inversion.prediction_errs_flat.values)
    #pred.insert(loc=1, column = 'posterior_cov', value = Inversion.prediction_errs.values)
    pred.insert(loc = 1, column ='prior_err_cov', value = np.diagonal(Inversion.flux_errs_flat.values))
    pred.to_pickle(savepath + str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_predictions.pkl')

def split_predictions_and_flux_and_priors(): 
    Inversion.predictions_flat = Inversion.predictions_flat.rename(dict(new = 'bioclass'))
    predictions_list = split_output(Inversion.predictions_flat, 'bioclass')

    fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio = split_eco_flux(Inversion) ### change to use split_output too 
    
    post_std_list = split_output(Inversion.prediction_errs_flat, 'new')
    
    prior_std = xr.DataArray(np.sqrt(np.diagonal(Inversion.flux_errs_flat.values)),coords = [np.arange(0,len(np.diagonal(Inversion.flux_errs_flat.values)))], dims = 'new')
    prior_std_list = split_output(prior_std, 'new')

    flux_list = [fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio]
    name_list = ['CO2_fire', 'CO2_ant_bio', 'CO_fire', 'CO_ant_bio']  
    
    fluxCO2_fire_grid, fluxCO2_bio_grid, fluxCO_fire_grid, fluxCO_bio_grid = split_gridded_flux(Inversion)

    flux_list_grid = [fluxCO2_fire_grid, fluxCO2_bio_grid, fluxCO_fire_grid, fluxCO_bio_grid]
    name_list_grid = ['CO2_fire_grid', 'CO2_ant_bio_grid', 'CO_fire_grid', 'CO_ant_bio_grid']  

    return flux_list_grid, name_list_grid, flux_list, name_list, post_std_list, prior_std_list, predictions_list


def plot_spatial_prior_and_prior_errors(flux_list, name_list, flux_list_grid, name_list_grid, prior_std_list, l, lCO, without_errors, with_errors):
    if without_errors == True: 
        print('     Plotting prior spatially')
        plot_prior_spatially(Inversion, flux_list[idx],name_list[idx], idx, savepath)
        plot_prior_spatially(Inversion, flux_list_grid[idx],name_list_grid[idx], idx,savepath, False)
    if with_errors == True: 
        print('     Plotting prior errors spatially')
        plot_spatial_result(Inversion.map_on_grid(prior_std_list[idx]), savepath, str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_prior_std_'+name_list[idx]+'.png', 'pink_r', vmax =None, vmin =0, 
                        cbar_kwargs = {'shrink':  0.835, 'label' : r'prior standard deviation'}, norm = None )
        
def plot_spatial_posterior_and_posterior_errors(predictions, flux_list, name_list, l, lCO, post_std_list, without_errors, with_errors):
    if without_errors == True: 
        print('     Plotting spatial difference of posterior to prior spatially')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l,lCO, diff =False)
        
        print('     Plotting posterior spatially')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l,lCO, diff =True)
    if with_errors == True: 
        print('     Plotting posterior standard deviation spatially')
        plot_spatial_result(Inversion.map_on_grid(post_std_list[idx]), savepath, str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_post_std_'+name_list[idx]+'.png', 'pink_r', vmax =None, vmin =0, 
                            cbar_kwargs = {'shrink':  0.835, 'label' : r'posterior standard deviation'}, norm = None)


def plot_uncertainty_reduction(post_std_list, prior_std_list, name_list, lCO, l):
    print('     Plotting uncertainty reduction spatially')
    plot_spatial_result(Inversion.map_on_grid((prior_std_list[idx]-post_std_list[idx])/prior_std_list[idx]), savepath,
                            str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_uncertainty_reduction_'+name_list[idx]+'.png', 'bone_r', vmax =1, vmin = -0,
                        cbar_kwargs = {'shrink':  0.835, 'label' : r'uncertainty reduction'}, norm = None)
    plot_spatial_result(Inversion.map_on_grid((prior_std_list[idx]-post_std_list[idx])/prior_std_list[idx]), savepath,
                            str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_uncertainty_reduction_zero_enforced_'+name_list[idx]+'.png', 'bone_r', vmax =1, vmin = 0,
                        cbar_kwargs = {'shrink':  0.835, 'label' : r'uncertainty reduction'}, norm = None)

def plot_emission_ratio_and_errors(predictions_list, post_std_list, fluxCO2_fire, fluxCO_fire, flux_list, l, lCO,with_errors, without_errors): 
    if without_errors == True:    
        print('Plotting emission ratio spatially')    
        plot_emission_ratio(savepath, Inversion, predictions_list[0], predictions_list[2], l,lCO, fluxCO2_fire, fluxCO_fire)   

    if with_errors == True: 
        print('Plotting emission ratio errors spatially')
        ratio_err = calculate_ratio_error(post_std_list[0], post_std_list[2], predictions_list[0], predictions_list[2])
        plot_spatial_result(Inversion.map_on_grid(ratio_err), savepath,
                                    str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_ratio_err_'+str(Inversion.week)+'.png', 'pink_r', vmax =None, vmin =0, 
                                cbar_kwargs = {'shrink':  0.835, 'label' : r'$standard deviation'}, norm = None)
        plot_spatial_result(Inversion.map_on_grid(ratio_err), savepath,
                                    str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_ratio_err_cbar_cut_'+str(Inversion.week)+'.png', 'pink_r', vmax =5, vmin =0, 
                                cbar_kwargs = {'shrink':  0.835, 'label' : r'standard deviation'}, norm = None)
        plot_spatial_result(Inversion.map_on_grid(ratio_err*flux_list[0]/flux_list[2]*44/28), savepath,
                                    str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_ratio_err_flux_cbar_cut_'+str(Inversion.week)+'.png', 'pink_r', vmax =140, vmin =0, 
                                cbar_kwargs = {'shrink':  0.835, 'label' : r'standard deviation [gCO$_2$/gCO]'}, norm = None)
    

def plot_concentrations(l, lCO, prior_std, without_errors, with_errors): 
    if without_errors == True: 
        print('Plotting concentrations without errors')
        total_CO2, total_CO = plot_single_concentrations(Inversion, l,lCO, savepath)
        CO2, CO = plot_single_total_concentrations(Inversion,l, lCO, savepath)
    if with_errors == True: 
        print('Plotting concentrations with errors')
        plot_single_concentrations_measurements_and_errors(Inversion, savepath, l, lCO,prior_std)
    #plot_difference_of_posterior_concentrations_to_measurements(Inversion, savepath,l, lCO)   


def do_everything(savepath, Inversion, mask_datapath_and_name, alpha_CO2, alpha_CO, plot_spatial_input_mask, plot_spatial_results,
                                plot_spatial_posterior_errors, plot_spatial_prior ,plot_spatial_prior_errors , plot_uncertainty_reduction, plot_spatial_emission_ratio , 
                                plot_spatial_emission_ratio_errors , plot_spatial_averaging_kernel , plot_concentrations_with_errors ,
                                plot_concentrations_without_errors ,save_prediction_pkl , save_l_curve_information):
    '''
    Function plotting and saving all relevant results of the coupled Inversion 
    
    Parameters

    savepath :                  path where to save data and images
    Inversion :                 Instance of class Inversion 
    mask_datapath_and_name :    path and name of mask of spatial output resolution 
    alpha_CO2 :                 CO2 regularization parameter 
    alpha_CO :                  CO regularization parameter 

    ''' 
    print('fitting')
    Inversion.fit(alpha = 1, xerr = Inversion.flux_errs_flat)
    
    if plot_spatial_input_mask == True: 
        class_num = plot_input_mask(savepath,mask_datapath_and_name)

    # save predictions
    #Inversion.predictions_errs.to_netcdf(savepath +str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_posterior_cov_matrix.nc')
    if save_prediction_pkl == True: 
        save_predictions(savepath, alpha_CO2, alpha_CO)

    flux_list_grid, name_list_grid, flux_list, name_list, post_std_list, prior_std_list, predictions_list = split_predictions_and_flux_and_priors()

    for idx,predictions in enumerate(predictions_list):
        print('Plotting '+ "\033[1m"+name_list[idx]+"\033[0;0m")
        if plot_spatial_prior == True or plot_spatial_prior_errors == True: 
            plot_spatial_prior_and_prior_errors(flux_list, name_list, flux_list_grid, name_list_grid, prior_std_list, alpha_CO2, alpha_CO, plot_spatial_prior, plot_spatial_prior_errors)
        if plot_spatial_results == True or plot_spatial_posterior_errors == True: 
            plot_spatial_posterior_and_posterior_errors(predictions, flux_list, name_list, alpha_CO2, alpha_CO, post_std_list,plot_spatial_results, plot_spatial_posterior_errors)

        if plot_uncertainty_reduction == True: 
            plot_spatial_result(Inversion.map_on_grid((prior_std_list[idx]-post_std_list[idx])/prior_std_list[idx]), savepath,
                                    str("{:.2e}".format(alpha_CO2))+"_CO2_"+"{:.2e}".format(alpha_CO)+'_CO_uncertainty_reduction_'+name_list[idx]+'.png', 'bone_r', vmax =1, vmin = -0,
                                cbar_kwargs = {'shrink':  0.835, 'label' : r'uncertainty reduction'}, norm = None)
            plot_spatial_result(Inversion.map_on_grid((prior_std_list[idx]-post_std_list[idx])/prior_std_list[idx]), savepath,
                                    str("{:.2e}".format(alpha_CO2))+"_CO2_"+"{:.2e}".format(alpha_CO)+'_CO_uncertainty_reduction_zero_enforced_'+name_list[idx]+'.png', 'bone_r', vmax =1, vmin = 0,
                                cbar_kwargs = {'shrink':  0.835, 'label' : r'uncertainty reduction'}, norm = None)
    if plot_spatial_emission_ratio == True or plot_spatial_emission_ratio_errors == True: 
        plot_emission_ratio_and_errors( predictions_list, post_std_list, flux_list[0], flux_list[2],flux_list, alpha_CO2, alpha_CO,with_errors = plot_spatial_emission_ratio_errors, without_errors = plot_spatial_emission_ratio) 

    if plot_spatial_averaging_kernel == True: 
        calculate_and_plot_averaging_kernel(Inversion,savepath,alpha_CO2, alpha_CO, Inversion.week) 

    if plot_concentrations_with_errors == True or plot_concentrations_without_errors == True: 
        prior_std = xr.DataArray(np.sqrt(np.diagonal(Inversion.flux_errs_flat.values)),coords = [np.arange(0,len(np.diagonal(Inversion.flux_errs_flat.values)))], dims = 'new')
        plot_concentrations(alpha_CO2, alpha_CO, prior_std, without_errors = plot_concentrations_without_errors, with_errors = plot_concentrations_with_errors)
    if save_l_curve_information == True: 
        inv_result = Inversion.reg.compute_l_curve(alpha_list =[1])
        with open(savepath+str("{:.2e}".format(alphaCO2))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_inv_result.pickle', 'wb') as handle:
            pickle.dump(inv_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

#################################################### adapt from here on ####################################################################
# Inversion is done weekly for weeks in 2019/2020 specified in week_list:
# if week_list starts at > 35, index of start week in list: [35, 36, 37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,1,2,3,4,5,6,7,8,9]
# must be added to for definition of start_date, end_date, month str (below)
week_list = [48,49,50,51,52,1]#[35, 36, 37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,1,2,3,4,5,6,7,8,9]

# hardcoded list of start and end dates for week 35 2019 to week 9 2020, end dates of week + 10 days to load all footprints that contribute to the week of interest 
start_dates = np.array([datetime.datetime(year = 2019, month = 9, day = 1), datetime.datetime(year = 2019, month = 9, day = 2), datetime.datetime(year = 2019, month = 9, day = 9), datetime.datetime(year = 2019, month = 9, day = 16), 
               datetime.datetime(year = 2019, month = 9, day = 23), datetime.datetime(year = 2019, month = 10, day = 1),datetime.datetime(year = 2019, month = 10, day = 7), datetime.datetime(year = 2019, month = 10, day = 14), 
               datetime.datetime(year = 2019, month = 10, day = 21), datetime.datetime(year = 2019, month = 11, day = 1), datetime.datetime(year = 2019, month = 11, day = 4), datetime.datetime(year = 2019, month = 11, day = 11),
               datetime.datetime(year = 2019, month = 11, day = 18), datetime.datetime(year = 2019, month = 12, day = 1),datetime.datetime(year = 2019, month = 12, day = 2), datetime.datetime(year = 2019, month = 12, day = 9),
               datetime.datetime(year = 2019, month = 12, day = 16), datetime.datetime(year = 2019, month = 12, day = 23), datetime.datetime(year = 2019, month = 12, day = 30),datetime.datetime(year = 2020, month = 1, day = 6), 
               datetime.datetime(year = 2020, month = 1, day = 13), datetime.datetime(year = 2020, month = 1, day = 20), datetime.datetime(year = 2020, month = 1, day = 27), 
               datetime.datetime(year = 2020, month = 2, day = 3), datetime.datetime(year = 2020, month = 2, day = 10),datetime.datetime(year = 2020, month = 2, day = 17), datetime.datetime(year = 2020, month = 2, day = 24)])

end_dates = np.array([datetime.datetime(year = 2019, month = 9, day = 11), datetime.datetime(year = 2019, month = 9, day = 18), datetime.datetime(year = 2019, month = 9, day = 25), datetime.datetime(year = 2019, month = 10, day = 2), 
               datetime.datetime(year = 2019, month = 10, day = 9), datetime.datetime(year = 2019, month = 10, day = 16),datetime.datetime(year = 2019, month = 10, day = 23), datetime.datetime(year = 2019, month = 10, day = 30), 
               datetime.datetime(year = 2019, month = 11, day = 6), datetime.datetime(year = 2019, month = 11, day = 13), datetime.datetime(year = 2019, month = 11, day = 20), datetime.datetime(year = 2019, month = 11, day = 27),
               datetime.datetime(year = 2019, month = 12, day = 4), datetime.datetime(year = 2019, month = 12, day = 11),datetime.datetime(year = 2019, month = 12, day = 18), datetime.datetime(year = 2019, month = 12, day = 25),
               datetime.datetime(year = 2020, month = 1, day = 1), datetime.datetime(year = 2020, month = 1, day = 8), datetime.datetime(year = 2020, month = 1, day = 7),datetime.datetime(year = 2020, month = 1, day = 22), 
               datetime.datetime(year = 2020, month = 1, day = 29), datetime.datetime(year = 2020, month = 2, day = 5), datetime.datetime(year = 2020, month = 2, day = 8), 
               datetime.datetime(year = 2020, month = 2, day = 19), datetime.datetime(year = 2020, month = 2, day = 26),datetime.datetime(year = 2020, month = 2, day = 29), datetime.datetime(year = 2020, month = 2, day = 29)])# 10 days must be added to cpnsider the footprints then too 

month_list = np.array(['2019-09','2019-09','2019-09','2019-09','2019-09','2019-10','2019-10','2019-10','2019-10','2019-11','2019-11','2019-11','2019-11','2019-11','2019-12','2019-12', '2019-12',
                       '2019-12','2019-12','2020-01','2020-01','2020-01','2020-01','2020-02','2020-02','2020-02','2020-02'])
# regularization parameters, can be a list of several parameters.
#  Then, the inversion will be done for each combination of regularization parameters. 
alpha_list = [1e-2] # CO2
alpha_listCO = [2.12] # CO

# plotting options. Set to False if it should not be printed/saved
plot_spatial_input_mask = True

plot_spatial_results = True
plot_spatial_posterior_errors = True

plot_spatial_prior = True
plot_spatial_prior_errors = True

plot_uncertainty_reduction_spatially = True

plot_spatial_emission_ratio = True
plot_spatial_emission_ratio_errors = True

plot_spatial_averaging_kernel = True

plot_concentrations_with_errors = True
plot_concentrations_without_errors = True

save_prediction_pkl = True 

save_l_curve_information = True ###########


for idx,week in enumerate(week_list): 
############ +13 to start with week 48 ##############
    start_date = start_dates[idx+13] 
    end_date = end_dates[idx+13]
    month_str = month_list[idx+13]
    for alphaCO in alpha_listCO:
        for alphaCO2 in alpha_list:
            for prior_err_CO in [1]: # 1 is 100% error 
                for prior_err_CO2 in [1]: # adapt savepaths for different prior errors
                    for meas_err_CO in [1]: # in ppb # 1  means that the true tccon uncertainties will be used, values different from 1 will be set to measurement errors
                        for meas_err_CO2 in [1]: # in ppm # see comment for meas_err_CO
                            for correlation in [0.7]:
                                # adapt savepath 
                                savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_'+str(prior_err_CO2*100)+'_CO_'+str(prior_err_CO*100)+'/CO2_'+str(meas_err_CO2)+'_CO_'+str(meas_err_CO)+'/Corr_'+str(correlation)+'/All_weeks/'+str(week)+'/'
                                if not os.path.isdir(savepath): os.makedirs(savepath) 
                                else: print('WARNING: savepath already exists')
                                # give path to spatial output mask (spatial resolution)
                                    # Final output mask of this thesis: Ecosystems_AK_based_split_with_21_and_20_larger_and_4_split.nc
                                    # 1x1 degree grid covering Australia: OekomaskAU_Flexpart_version8_all1x1
                                mask = "/home/b/b382105/ColumnFLEXPART/resources/Ecosystems_AK_based_split_with_21_and_20_larger_and_4_split.nc"
                                # create_mask_based_on_regions(savepath,mask,selection = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
                                # area bioreg for region 4 split 21 and 20 larger (final output mask), required for initialization of class, not excplicitly used: 
                                area_bioreg = [8.95512935e+13, 4.27021383e+11, 4.01383356e+12, 1.45736998e+12,
                                2.15745316e+11, 9.81395187e+09, 3.95111921e+10, 3.18183296e+10,
                                5.52121748e+10, 5.55915961e+10, 5.41460509e+10, 4.29887311e+10,
                                3.13764949e+10, 1.05335427e+10, 9.31980461e+10, 5.11758810e+10,
                                4.00158388e+10, 2.99495636e+10, 3.91250878e+10, 4.90667320e+10,
                                1.54614599e+11, 9.65077342e+10, 6.07622615e+10, 6.98377101e+10,
                                1.77802351e+11, 1.95969571e+11, 2.26322082e+11]

                                # initialize an instance of class coupled inversion
                                Inversion = CoupledInversion(
                                    result_pathCO = "/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/predictions_CO_2019_2020.pkl", # enhancements from calc_resultsCO (adapted from MA Christopher)
                                    result_pathCO2 = "/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/predictionsCO2_2019_2020.pkl", # enhancements from calc_results ( from MA Christopher)
                                    flux_pathCO = "/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/Regridded_1x1/", # prior flux on 1x1 grid
                                    flux_pathCO2 = "/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Flux/CT2022.flux1x1.",
                                    bioclass_path = mask,
                                    month = month_str,
                                    date_min = start_date,
                                    area_bioreg = area_bioreg,
                                    week = week,
                                    non_equal_region_size=True, # not really used in class, can be omitted by adapting class
                                    date_max = end_date,
                                    boundary=[110.0, 155.0, -45.0, -10.0], 
                                    meas_err_CO = meas_err_CO,
                                    meas_err_CO2 = meas_err_CO2,
                                    prior_err_CO_fire = prior_err_CO/np.sqrt(alphaCO), 
                                    prior_err_CO2_fire = prior_err_CO2/np.sqrt(alphaCO2),
                                    correlation = correlation, 
                                    ) 
                                print('Initialization done')

                                # fuction to plot and save all specified results: 
                                do_everything(savepath, Inversion, mask, alphaCO2, alphaCO, plot_spatial_input_mask, plot_spatial_results,
                                plot_spatial_posterior_errors, plot_spatial_prior ,plot_spatial_prior_errors , plot_uncertainty_reduction_spatially, plot_spatial_emission_ratio , 
                                plot_spatial_emission_ratio_errors , plot_spatial_averaging_kernel , plot_concentrations_with_errors ,
                                plot_concentrations_without_errors ,save_prediction_pkl , save_l_curve_information)            