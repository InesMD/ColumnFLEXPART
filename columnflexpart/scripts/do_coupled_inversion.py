
from columnflexpart.classes.coupled_inversion import CoupledInversion #_no_corr_no_split_fluxes import CoupledInversion
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from columnflexpart.scripts.plotting_routine_coupled_split_inversion import plot_spatial_flux_results_or_diff_to_prior, plot_averaging_kernel,plot_input_mask,plot_weekly_concentrations, plot_single_concentrations
from columnflexpart.scripts.plotting_routine_coupled_split_inversion import split_eco_flux, split_predictions,  plot_prior_spatially


def do_everything(savepath, Inversion, molecule_name, mask_datapath_and_name, week_min, week_max, week_num):
    class_num = plot_input_mask(savepath,mask_datapath_and_name)
    #plot_prior_spatially(Inversion,molecule_name,week_min, week_max, savepath)
    #l1, l2 = find_two_optimal_lambdas(Inversion,[1e-7,10], 1e-14, err)# 1e-8
    #print(l1)
    l3 = 5e-4
    for l in [l3]:#1, l2]:#,l2,l3]:
        print('fitting')
        predictions = Inversion.fit(alpha = l)#, xerr = err) 
        Inversion.predictions_flat = Inversion.predictions_flat.rename(dict(new = 'bioclass'))
        #plot_l_curve(Inversion,err,molecule_name,savepath, l)
        predictionsCO2_fire, predictionsCO2_bio, predictionsCO_fire, predictionsCO_bio = split_predictions(Inversion)
        fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio = split_eco_flux(Inversion)

        flux_list = [fluxCO2_fire, fluxCO2_bio, fluxCO_fire, fluxCO_bio]
        name_list = ['CO2_fire_small_prior', 'CO2_ant_bio_small_prior_', 'CO_fire_small_prior_', 'CO_ant_bio_small_prior']  
        for idx,predictions in enumerate([predictionsCO2_fire,predictionsCO2_bio,predictionsCO_fire,predictionsCO_bio]):
            print('Plotting prior spatially')
            plot_prior_spatially(Inversion, flux_list[idx],name_list[idx], idx, savepath)

            print('Plotting spatial difference of results to prior')
            plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l, diff =False)
            
            print('Plotting spatial results')
            plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, predictions, flux_list[idx], name_list[idx],idx, l, diff =True)

            print('Plotting averaging kernels') # not yet working?! CHeck results!!!!!!!!!!!!!!!! 
            #plot_averaging_kernel(Inversion, l, class_num, week_num,savepath, plot_spatially=False)# class_num
            #plot_averaging_kernel(Inversion, l, class_num, week_num,savepath, plot_spatially=True,weekly = True)
            #plot_averaging_kernel(Inversion, l, class_num, week_num,savepath, plot_spatially=True)

        print('Plotting concentrations')
        plot_single_concentrations(Inversion, predictionsCO_fire, predictionsCO_bio, fluxCO_fire, fluxCO_bio, 'CO',l, savepath)
        plot_single_concentrations(Inversion, predictionsCO2_fire, predictionsCO2_bio, fluxCO2_fire,fluxCO2_bio, 'CO2',l, savepath)
        #plot_single_concentrations(Inversion,  'CO',l, savepath)
        #plot_weekly_concentrations(Inversion,'CO',l, savepath) # ist ziemlich hartgecoded gerade für Dezember!!!!!!!!!!
        #plot_weekly_concentrations(Inversion,'CO2',l, savepath)


######################### adapt stuff from here on ####################################

savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/Images_coupled_Inversion/everything_splitted_first_correlation_setup_weekly_inversion/52/'
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