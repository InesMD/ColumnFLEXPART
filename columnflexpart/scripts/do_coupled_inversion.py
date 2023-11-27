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
    pred = pd.DataFrame(data =Inversion.predictions_flat.values, columns = ['predictions'])
    pred.insert(loc = 1, column = 'prior_flux_eco', value = Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week].squeeze())
    pred.insert(loc = 1, column ='posterior_err_std', value = Inversion.prediction_errs_flat.values)
    #pred.insert(loc=1, column = 'posterior_cov', value = Inversion.prediction_errs.values)
    pred.insert(loc = 1, column ='prior_err_cov', value = np.diagonal(Inversion.flux_errs_flat.values))
    #pred.insert(loc = 1, column ='prior_err_std_with_lambda', value = np.sqrt(np.diagonal(Inversion.flux_errs_flat.values*reg)))# STIMMT VLLT NICHT!!!!!!!!!!!!!!!!!!! reg
    pred.to_pickle(savepath + str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_predictions.pkl')


def split_predictions_and_flux_and_priors(): 
    Inversion.predictions_flat = Inversion.predictions_flat.rename(dict(new = 'bioclass'))
    #plot_l_curve(Inversion,err,molecule_name,savepath, l)
    predictions_list = split_output(Inversion.predictions_flat, 'bioclass')
    fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio = split_eco_flux(Inversion) ### change to use split_output too 
    post_std_list = split_output(Inversion.prediction_errs_flat, 'new')
    prior_std = xr.DataArray(np.sqrt(np.diagonal(Inversion.flux_errs_flat.values)),coords = [np.arange(0,len(np.diagonal(Inversion.flux_errs_flat.values)))], dims = 'new')
    prior_std_list = split_output(prior_std, 'new')

    flux_list = [fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio]
    name_list = ['CO2_fire', 'CO2_ant_bio', 'CO_fire', 'CO_ant_bio']  
    

    ##predictions_errs_flat = Inversion.predictions_errs.rename(dict(new = 'bioclass'))
    #plot_l_curve(Inversion,err,molecule_name,savepath, l)
    #prediction_errs_list = split_output(predictions_errs_flat, 'bioclass')
    #sum = np.sqrt(np.sum(np.diag(fluxCO2_fire**2*prediction_errs_list[0]))+ np.sum(np.diag(fluxCO2_bio**2*prediction_errs_list[1]))+np.sum(np.diag(fluxCO_fire**2*prediction_errs_list[2]))
    #+ np.sum(np.diag(fluxCO_bio**2*prediction_errs_list[3]))+Inversion.predictions_errs*flux)

    fluxCO2_fire_grid, fluxCO2_bio_grid, fluxCO_fire_grid, fluxCO_bio_grid = split_gridded_flux(Inversion)

    flux_list_grid = [fluxCO2_fire_grid, fluxCO2_bio_grid, fluxCO_fire_grid, fluxCO_bio_grid]
    name_list_grid = ['CO2_fire_grid', 'CO2_ant_bio_grid', 'CO_fire_grid', 'CO_ant_bio_grid']  

def plot_spatial_prior_and_posterior_results():
    print('Plotting prior spatially')
    plot_prior_spatially(Inversion, flux_list[idx],name_list[idx], idx, savepath)
    plot_prior_spatially(Inversion, flux_list_grid[idx],name_list_grid[idx], idx,savepath, False)

    print('Plotting spatial difference of results to prior')
    plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l,lCO, diff =False)
    plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx]+'_45_',idx, l,lCO, vmin = -45, vmax = 45, diff =False)
    
    print('Plotting spatial results')
    plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l,lCO, diff =True)

def plot_prior_and_posterior_errors():
    print('Plot posterior std deviation')
    plot_spatial_result(Inversion.map_on_grid(post_std_list[idx]), savepath, str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_post_std_'+name_list[idx]+'.png', 'pink_r', vmax =0.5, vmin =0, 
                        cbar_kwargs = {'shrink':  0.835, 'label' : r'posterior standard deviation'}, norm = None)
    plot_spatial_result(Inversion.map_on_grid(prior_std_list[idx]), savepath, str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_prior_std_'+name_list[idx]+'.png', 'pink_r', vmax =None, vmin =0, 
                    cbar_kwargs = {'shrink':  0.835, 'label' : r'prior standard deviation'}, norm = None )
    print('Plot uncertainty reduction')
    plot_spatial_result(Inversion.map_on_grid((prior_std_list[idx]-post_std_list[idx])/prior_std_list[idx]), savepath,
                            str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_uncertainty_reduction_'+name_list[idx]+'.png', 'bone_r', vmax =1, vmin = -0,
                        cbar_kwargs = {'shrink':  0.835, 'label' : r'uncertainty reduction'}, norm = None)
    plot_spatial_result(Inversion.map_on_grid((prior_std_list[idx]-post_std_list[idx])/prior_std_list[idx]), savepath,
                            str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_uncertainty_reduction_zero_enforced_'+name_list[idx]+'.png', 'bone_r', vmax =1, vmin = 0,
                        cbar_kwargs = {'shrink':  0.835, 'label' : r'uncertainty reduction'}, norm = None)

def plot_emission_ratio_and_errors(): 
    
    print('Plot emission ratio and errors')
       
    plot_emission_ratio(savepath, Inversion, predictions_list[0], predictions_list[2], l,lCO, fluxCO2_fire, fluxCO_fire)   

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
 

def plot_concentrations(): 
    print('Plotting concentrations')
    total_CO2, total_CO = plot_single_concentrations(Inversion, l,lCO, savepath)
    CO2, CO = plot_single_total_concentrations(Inversion,l, lCO, savepath)
    plot_single_concentrations_measurements_and_errors(Inversion, savepath, l, lCO,prior_std)
    plot_difference_of_posterior_concentrations_to_measurements(Inversion, savepath,l, lCO)   


def do_everything(savepath, Inversion, mask_datapath_and_name, alpha_CO2, alpha_CO, plot_spatial_results,
                                plot_spatial_posterior_errors, plot_spatial_prior ,plot_spatial_prior_errors , plot_spatial_emission_ratio , 
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
    class_num = plot_input_mask(savepath,mask_datapath_and_name)
    Inversion.fit(alpha = 1, xerr = Inversion.flux_errs_flat)

    for l in [alpha_CO2]:
        #savepath = savepath2 + 'CO2_' + str("{:.2e}".format(l)) + '/'
        if not os.path.isdir(savepath): os.makedirs(savepath)

        for lCO in [alpha_CO]: 
            print('fitting')
            #workaround for having several regularization parameters: alpha = 1 and xerr = xerr *unit matrix with 1/desired alpha on diagonal
            reg  = xr.zeros_like(Inversion.flux_errs_flat)
            for i in range(int(reg.shape[0]/2)): 
                reg[i,i] = 1/l
            for i in range(int(reg.shape[0]/2),int(reg.shape[0]) ): 
                reg[i,i] = 1/lCO
             

            # save predictions
            #Inversion.predictions_errs.to_netcdf(savepath +str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_posterior_cov_matrix.nc')
    
            pred = pd.DataFrame(data =Inversion.predictions_flat.values, columns = ['predictions'])
            pred.insert(loc = 1, column = 'prior_flux_eco', value = Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week].squeeze())
            pred.insert(loc = 1, column ='posterior_err_std', value = Inversion.prediction_errs_flat.values)
            #pred.insert(loc=1, column = 'posterior_cov', value = Inversion.prediction_errs.values)
            pred.insert(loc = 1, column ='prior_err_cov', value = np.diagonal(Inversion.flux_errs_flat.values))
            pred.insert(loc = 1, column ='prior_err_std_with_lambda', value = np.sqrt(np.diagonal(Inversion.flux_errs_flat.values*reg)))# STIMMT VLLT NICHT!!!!!!!!!!!!!!!!!!! reg
            pred.to_pickle(savepath + str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_predictions.pkl')
            
            Inversion.predictions_flat = Inversion.predictions_flat.rename(dict(new = 'bioclass'))
            #plot_l_curve(Inversion,err,molecule_name,savepath, l)
            predictions_list = split_output(Inversion.predictions_flat, 'bioclass')
            fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio = split_eco_flux(Inversion) ### change to use split_output too 
            post_std_list = split_output(Inversion.prediction_errs_flat, 'new')
            prior_std = xr.DataArray(np.sqrt(np.diagonal(Inversion.flux_errs_flat.values)),coords = [np.arange(0,len(np.diagonal(Inversion.flux_errs_flat.values)))], dims = 'new')
            prior_std_list = split_output(prior_std, 'new')

            flux_list = [fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio]
            name_list = ['CO2_fire', 'CO2_ant_bio', 'CO_fire', 'CO_ant_bio']  
           

            ##predictions_errs_flat = Inversion.predictions_errs.rename(dict(new = 'bioclass'))
            #plot_l_curve(Inversion,err,molecule_name,savepath, l)
            #prediction_errs_list = split_output(predictions_errs_flat, 'bioclass')
            #sum = np.sqrt(np.sum(np.diag(fluxCO2_fire**2*prediction_errs_list[0]))+ np.sum(np.diag(fluxCO2_bio**2*prediction_errs_list[1]))+np.sum(np.diag(fluxCO_fire**2*prediction_errs_list[2]))
            #+ np.sum(np.diag(fluxCO_bio**2*prediction_errs_list[3]))+Inversion.predictions_errs*flux)

            fluxCO2_fire_grid, fluxCO2_bio_grid, fluxCO_fire_grid, fluxCO_bio_grid = split_gridded_flux(Inversion)

            flux_list_grid = [fluxCO2_fire_grid, fluxCO2_bio_grid, fluxCO_fire_grid, fluxCO_bio_grid]
            name_list_grid = ['CO2_fire_grid', 'CO2_ant_bio_grid', 'CO_fire_grid', 'CO_ant_bio_grid']  
        

            for idx,predictions in enumerate(predictions_list):
                print('Plotting prior spatially')
                plot_prior_spatially(Inversion, flux_list[idx],name_list[idx], idx, savepath)
                plot_prior_spatially(Inversion, flux_list_grid[idx],name_list_grid[idx], idx,savepath, False)

                print('Plotting spatial difference of results to prior')
                plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l,lCO, diff =False)
                plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx]+'_45_',idx, l,lCO, vmin = -45, vmax = 45, diff =False)
                
                print('Plotting spatial results')
                plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l,lCO, diff =True)

                print('Plot posterior std deviation')
                plot_spatial_result(Inversion.map_on_grid(post_std_list[idx]), savepath, str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_post_std_'+name_list[idx]+'.png', 'pink_r', vmax =0.5, vmin =0, 
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'posterior standard deviation'}, norm = None)
                # flux: 
                if idx>1:
                    plot_spatial_result(Inversion.map_on_grid(post_std_list[idx]*flux_list[idx]*12*10**6), savepath, str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_post_std_flux_1_max'+name_list[idx]+'_'+str(Inversion.week)+'.png', 'pink_r', vmax =0.8, vmin =0, 
                                        cbar_kwargs = {'shrink':  0.835, 'label' : r'standard deviation [$\mu$gC m$^{-2}$s$^{-1}$]'}, norm = None)
                else:
                    plot_spatial_result(Inversion.map_on_grid(post_std_list[idx]*flux_list[idx]*12*10**6), savepath, str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_post_std_flux_1_max'+name_list[idx]+'_'+str(Inversion.week)+'.png', 'pink_r', vmax =150, vmin =0, 
                                        cbar_kwargs = {'shrink':  0.835, 'label' : r'standard deviation [$\mu$gC m$^{-2}$s$^{-1}$]'}, norm = None)
           
                plot_spatial_result(Inversion.map_on_grid(prior_std_list[idx]), savepath, str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_prior_std_'+name_list[idx]+'.png', 'pink_r', vmax =None, vmin =0, 
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'prior standard deviation'}, norm = None )

                plot_spatial_result(Inversion.map_on_grid((prior_std_list[idx]-post_std_list[idx])/prior_std_list[idx]), savepath,
                                        str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_uncertainty_reduction_'+name_list[idx]+'.png', 'bone_r', vmax =1, vmin = -0,
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'uncertainty reduction'}, norm = None)
                plot_spatial_result(Inversion.map_on_grid((prior_std_list[idx]-post_std_list[idx])/prior_std_list[idx]), savepath,
                                        str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_uncertainty_reduction_zero_enforced_'+name_list[idx]+'.png', 'bone_r', vmax =1, vmin = 0,
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'uncertainty reduction'}, norm = None)

            
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
            
            calculate_and_plot_averaging_kernel(Inversion,savepath,l, lCO, Inversion.week) 
            plot_emission_ratio(savepath, Inversion, predictions_list[0], predictions_list[2], l,lCO, fluxCO2_fire, fluxCO_fire)   
   
            print('Plotting concentrations')
            total_CO2, total_CO = plot_single_concentrations(Inversion, l,lCO, savepath)
            CO2, CO = plot_single_total_concentrations(Inversion,l, lCO, savepath)
            plot_single_concentrations_measurements_and_errors(Inversion, savepath, l, lCO,prior_std)
            plot_difference_of_posterior_concentrations_to_measurements(Inversion, savepath,l, lCO)   

            #plot_l_curve(Inversion, molecule_name, savepath, l, err = None)

#################################################### adapt from here on ####################################################################
# Inversion is done weekly for weeks in 2019/2020:
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
# regularization parameters
alpha_list = [1e-2]
alpha_listCO = [2.12]

# plotting options. Set to False if it should not be printed/saved
plot_spatial_results = True
plot_spatial_posterior_errors = True

plot_spatial_prior = True
plot_spatial_prior_errors = True

plot_spatial_emission_ratio = True
plot_spatial_emission_ratio_errors = True

plot_spatial_averaging_kernel = True

plot_concentrations_with_errors = True
plot_concentrations_without_errors = True

save_prediction_pkl = True ###############

save_l_curve_information = True ###########


for idx,week in enumerate(week_list): 
############ +13 to start with week 48 ##############
    start_date = start_dates[idx+13] 
    end_date = end_dates[idx+13]
    month_str = month_list[idx+13]
    for alphaCO in alpha_listCO:
        for alphaCO2 in alpha_list:
            for prior_err_CO in [1]:
                for prior_err_CO2 in [1]: # adapt savepaths for different prior errors
                    for meas_err_CO in [1]:
                        for meas_err_CO2 in [1]:
                            for correlation in [0.7]:
                                # adapt savepath 
                                savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_'+str(prior_err_CO2*100)+'_CO_'+str(prior_err_CO*100)+'/CO2_'+str(meas_err_CO2)+'_CO_'+str(meas_err_CO)+'/Corr_'+str(correlation)+'/All_weeks/'+str(week)+'/'
                                if not os.path.isdir(savepath): os.makedirs(savepath) 
                                else: print('WARNING: savepath already exists')
                                # load spatial output mask
                                # Final output mask of this thesis: Ecosystems_AK_based_split_with_21_and_20_larger_and_4_split.nc
                                # 1x1 degree grid covering Australia: OekomaskAU_Flexpart_version8_all1x1
                                mask = "/home/b/b382105/ColumnFLEXPART/resources/Ecosystems_AK_based_split_with_21_and_20_larger_and_4_split.nc"#OekomaskAU_Flexpart_version8_all1x1"
                                # plot_input_mask(savepath,mask)
                                # create_mask_based_on_regions(savepath,mask,selection = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
                                # area bioreg for region 4 split 21 and 20 larger: 
                                area_bioreg = [8.95512935e+13, 4.27021383e+11, 4.01383356e+12, 1.45736998e+12,
                                2.15745316e+11, 9.81395187e+09, 3.95111921e+10, 3.18183296e+10,
                                5.52121748e+10, 5.55915961e+10, 5.41460509e+10, 4.29887311e+10,
                                3.13764949e+10, 1.05335427e+10, 9.31980461e+10, 5.11758810e+10,
                                4.00158388e+10, 2.99495636e+10, 3.91250878e+10, 4.90667320e+10,
                                1.54614599e+11, 9.65077342e+10, 6.07622615e+10, 6.98377101e+10,
                                1.77802351e+11, 1.95969571e+11, 2.26322082e+11]

                                Inversion = CoupledInversion(
                                    result_pathCO = "/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/predictions_CO_2019_2020.pkl",#"/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions3_CO.pkl" ,
                                    result_pathCO2 = "/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/predictionsCO2_2019_2020.pkl",# "/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions.pkl" ,
                                    flux_pathCO = "/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/Regridded_1x1/", 
                                    flux_pathCO2 = "/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Flux/CT2022.flux1x1.",
                                    bioclass_path = mask,
                                    month = month_str,
                                    date_min = start_date,
                                    area_bioreg = area_bioreg,
                                    week = week,
                                    non_equal_region_size=True,
                                    date_max = end_date,
                                    boundary=[110.0, 155.0, -45.0, -10.0], 
                                    meas_err_CO = meas_err_CO,
                                    meas_err_CO2 = meas_err_CO2,
                                    prior_err_CO_fire = prior_err_CO/np.sqrt(alphaCO), 
                                    prior_err_CO2_fire = prior_err_CO2/np.sqrt(alphaCO2),
                                    correlation = correlation, 
                                    ) 
                                print('Initialization done')

                                #Inversion.fit(alpha = 1, xerr = Inversion.flux_errs_flat)
                                #inv_result = Inversion.reg.compute_l_curve(alpha_list =[1])
                                #with open(savepath+str("{:.2e}".format(alphaCO2))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_inv_result.pickle', 'wb') as handle:
                                #    pickle.dump(inv_result, handle, protocol=pickle.HIGHEST_PROTOCOL)#

                                #with open(savepath+str("{:.2e}".format(alphaCO2))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_inv_result.pickle', 'rb') as handle:
                                #    b = pickle.load(handle)
                                
                                # script to plot and save all results: 
                                do_everything(savepath, Inversion, mask, alphaCO2, alphaCO, plot_spatial_results,
                                plot_spatial_posterior_errors, plot_spatial_prior ,plot_spatial_prior_errors , plot_spatial_emission_ratio , 
                                plot_spatial_emission_ratio_errors , plot_spatial_averaging_kernel , plot_concentrations_with_errors ,
                                plot_concentrations_without_errors ,save_prediction_pkl , save_l_curve_information)            