
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
from columnflexpart.scripts.plotting_routine_coupled_split_inversion import calc_single_conc_by_multiplication_with_K, plot_single_concentrations_measurements_and_errors
def do_everything(savepath, Inversion, molecule_name, mask_datapath_and_name, week_min, week_max, week_num, alpha_list, alpha_listCO):
    class_num = plot_input_mask(savepath,mask_datapath_and_name, selection = [12])
    Inversion.fit(alpha = 1, xerr = Inversion.flux_errs_flat)

    for l in [alpha_list]:
        #savepath = savepath2 + 'CO2_' + str("{:.2e}".format(l)) + '/'
        if not os.path.isdir(savepath): os.makedirs(savepath)

        for lCO in [alpha_listCO]: 
            print('fitting')
            # ugly workaround for having several regularization parameters: alpha = 1 and xerr = xerr *unit matrix with 1/desired alpha on diagonal
            reg  = xr.zeros_like(Inversion.flux_errs_flat)
            for i in range(int(reg.shape[0]/2)): 
                reg[i,i] = 1/l
            for i in range(int(reg.shape[0]/2),int(reg.shape[0]) ): 
                reg[i,i] = 1/lCO
             

            # save predictions
            pred = pd.DataFrame(data =Inversion.predictions_flat.values, columns = ['predictions'])
            pred.insert(loc = 1, column = 'prior_flux_eco', value = Inversion.flux_eco[:,Inversion.flux_eco.week == Inversion.week].squeeze())
            pred.insert(loc = 1, column ='posterior_err_std', value = Inversion.prediction_errs_flat.values)
            pred.insert(loc = 1, column ='prior_err_cov', value = np.diagonal(Inversion.flux_errs_flat.values))
            pred.insert(loc = 1, column ='prior_err_std_with_lambda', value = np.sqrt(np.diagonal(Inversion.flux_errs_flat.values*reg)))# STIMMT VLLT NICHT!!!!!!!!!!!!!!!!!!! reg
            pred.to_pickle(savepath + str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_predictions.pkl')
            
            Inversion.predictions_flat = Inversion.predictions_flat.rename(dict(new = 'bioclass'))
            #plot_l_curve(Inversion,err,molecule_name,savepath, l)
            predictions_list = split_output(Inversion.predictions_flat, 'bioclass')
            fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio = split_eco_flux(Inversion) ### change to use split_output too 
            post_std_list = split_output(Inversion.prediction_errs_flat, 'new')
            prior_std = xr.DataArray(np.sqrt(np.diagonal(Inversion.flux_errs_flat.values*reg)),coords = [np.arange(0,len(np.diagonal(Inversion.flux_errs_flat.values)))], dims = 'new')
            prior_std_list = split_output(prior_std, 'new')

            flux_list = [fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio]
            name_list = ['CO2_fire', 'CO2_ant_bio', 'CO_fire', 'CO_ant_bio']  

            fluxCO2_fire_grid, fluxCO2_bio_grid, fluxCO_fire_grid, fluxCO_bio_grid = split_gridded_flux(Inversion)

            flux_list_grid = [fluxCO2_fire_grid, fluxCO2_bio_grid, fluxCO_fire_grid, fluxCO_bio_grid]
            name_list_grid = ['CO2_fire_grid', 'CO2_ant_bio_grid', 'CO_fire_grid', 'CO_ant_bio_grid']  
        

            for idx,predictions in enumerate(predictions_list):
                print('Plotting prior spatially')
                plot_prior_spatially(Inversion, flux_list[idx],name_list[idx], idx, savepath)
                #plot_prior_spatially(Inversion, flux_list_grid[idx],name_list_grid[idx], idx,savepath, False)

                print('Plotting spatial difference of results to prior')
                plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l,lCO, diff =False)
                
                print('Plotting spatial results')
                plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l,lCO, diff =True)

                print('Plot posterior std deviation')
                plot_spatial_result(Inversion.map_on_grid(post_std_list[idx]), savepath, str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_post_std_'+name_list[idx]+'.png', 'pink_r', vmax =None, vmin =0, 
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'posterior standard deviation'}, norm = None)
                # flux: 
                plot_spatial_result(Inversion.map_on_grid(post_std_list[idx]*flux_list[idx]*12*10**6), savepath, str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_post_std_flux_'+name_list[idx]+'.png', 'pink_r', vmax =None, vmin =0, 
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'posterior standard deviation [$\mu$gC m$^{-2}$ s$^{-1}$]'}, norm = None)
                
                plot_spatial_result(Inversion.map_on_grid(prior_std_list[idx]), savepath, str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_prior_std_'+name_list[idx]+'.png', 'pink_r', vmax =None, vmin =0, 
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'prior standard deviation '}, norm = None )

                #(prior variance(lambda considered) - posterior variance)/prior variance = (prior variance/lambda - post variance)/prior variance
                print('Plot error variance reduction/Uncertainty reduction')
                print(prior_std_list[0])
                print(post_std_list[0])
                print(prior_std_list[0]-post_std_list[0])
                print((prior_std_list[0]-post_std_list[0])/prior_std_list[0])
                plot_spatial_result(Inversion.map_on_grid((prior_std_list[idx]-post_std_list[idx])/prior_std_list[idx]), savepath,
                                        str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_uncertainty_reduction_'+name_list[idx]+'.png', 'bone_r', vmax =1, vmin =0, 
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'uncertainty reduction'}, norm = None)

            
            ratio_err = calculate_ratio_error(post_std_list[0], post_std_list[2], predictions_list[0], predictions_list[2])
            #print(ratio_err)
            plot_spatial_result(Inversion.map_on_grid(ratio_err), savepath,
                                        str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_ratio_err.png', 'pink_r', vmax =None, vmin =0, 
                                   cbar_kwargs = {'shrink':  0.835, 'label' : r'$\Delta$CO$_2$/$\Delta$CO standard deviation'}, norm = None)
            plot_spatial_result(Inversion.map_on_grid(ratio_err), savepath,
                                        str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_ratio_err_cbar_cut.png', 'pink_r', vmax =5, vmin =0, 
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'$\Delta$CO$_2$/$\Delta$CO standard deviation'}, norm = None)
            plot_spatial_result(Inversion.map_on_grid(ratio_err*flux_list[0]/flux_list[2]*44/28), savepath,
                                        str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_ratio_err_flux_cbar_cut.png', 'pink_r', vmax =None, vmin =0, 
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'$\Delta$CO$_2$/$\Delta$CO standard deviation [gCO$_2$/gCO]'}, norm = None)
            
            plot_single_concentrations_measurements_and_errors(Inversion, savepath, l, lCO,prior_std)
            plot_emission_ratio(savepath, Inversion, predictions_list[0], predictions_list[2], l,lCO, fluxCO2_fire, fluxCO_fire)

            #print('Plotting concentrations')
            total_CO2, total_CO = plot_single_concentrations(Inversion, l,lCO, savepath)
            CO2, CO = plot_single_total_concentrations(Inversion,l, lCO, savepath)
            plot_difference_of_posterior_concentrations_to_measurements(Inversion, savepath,l, lCO)

            calculate_and_plot_averaging_kernel(Inversion,savepath,l, lCO)
            #inv_result = Inversion.reg.compute_l_curve(alpha_list =[1])
            #f = netCDF4.Dataset(savepath+str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_inv_result.nc', 'w')
            #parms = f.createGroup('parameters')
            #for k,v in inv_result.items():
            #    setattr(parms, k, v)
            #with open(savepath+str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_inv_result.pickle', 'wb') as handle:
            #    pickle.dump(inv_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #with open(savepath+str("{:.2e}".format(l))+"_CO2_"+"{:.2e}".format(lCO)+'_CO_inv_result.pickle', 'rb') as handle:
            #    b = pickle.load(handle)


            #plot_l_curve(Inversion, molecule_name, savepath, l, err = None)

######################### adapt stuff from here on ####################################
alpha_list = [1e-3]
alpha_listCO = [1e-1]#1e-3,1e-2,1e-1,5e-1,1]
for alphaCO in alpha_listCO:
    for alphaCO2 in alpha_list:
        for prior_err_CO in [1]:
            for prior_err_CO2 in [1]: # adapt savepaths for different prior errors
                for meas_err_CO in [6]:
                    for meas_err_CO2 in [1]:
                        for correlation in [0.7]:
                            savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/CO_like_CO2_prior/2_reg_params_coupled/CO2_'+str(prior_err_CO2*100)+'_CO_'+str(prior_err_CO*100)+'/CO2_'+str(meas_err_CO2)+'_CO_'+str(meas_err_CO)+'/Corr_'+str(correlation)+'/'
                            if not os.path.isdir(savepath): os.makedirs(savepath) 
                            else: print('WARNING: savepath already exists')
                            mask = "/home/b/b382105/ColumnFLEXPART/resources/Ecosystems_AK_based_split_with_21_and_20_larger_and_4_split.nc"#OekomaskAU_Flexpart_version8_all1x1"#Ecosystems_AK_based_split_with_21_and_20_larger.nc"#OekomaskAU_Flexpart_version8_all1x1"#Ecosystems_AK_based_split_with_21_and_20_larger.nc"
                            non_equal_region_size = True

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
                                meas_err_CO = meas_err_CO,
                                meas_err_CO2 = meas_err_CO2,
                                prior_err_CO_fire = prior_err_CO/np.sqrt(alphaCO), 
                                prior_err_CO2_fire = prior_err_CO2/np.sqrt(alphaCO2),
                                correlation = correlation, 
                                ) 
                            print('Initialization done')

                            #print(Inversion.flux_eco_flat)

                            #Inversion.fit(alpha = 1, xerr = Inversion.flux_errs_flat)
                            #inv_result = Inversion.reg.compute_l_curve(alpha_list =[1])
                            #with open(savepath+str("{:.2e}".format(alphaCO2))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_inv_result.pickle', 'wb') as handle:
                            #    pickle.dump(inv_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

                            #with open(savepath+str("{:.2e}".format(alphaCO2))+"_CO2_"+"{:.2e}".format(alphaCO)+'_CO_inv_result.pickle', 'rb') as handle:
                            #    b = pickle.load(handle)
                            do_everything(savepath, Inversion,'CO',mask, 52, 52, 6, alphaCO2, alphaCO)
            