
from columnflexpart.scripts.coupled_inversion import CoupledInversion
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def plot_spatial_result(spatial_result, molecule_name, savepath, savename):
    '''
    for weekly spatial plotting of fluxes, saves plot 
    spatial_result : xarray to plot with latitude and longitude as coordinates
    molecule_name : Either "CO2" or "CO"
    savepath : path to save output image, must exist
    savename : Name of output image
    '''
    if molecule_name == "CO2" : 
        vmax = 250
    elif molecule_name == "CO": 
        vmax = 10
    vmin = -vmax

    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())  
    spatial_result.plot(x = 'longitude', y = 'latitude',ax = ax, cmap = 'seismic',vmin = vmin, vmax = vmax, 
                            cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
    plt.scatter(x = 150.8793,y =-34.4061,color="black")
    ax.coastlines()
    #plt.title('Week '+str(week))
    plt.savefig(savepath+savename, bbox_inches = 'tight', dpi = 450)
    plt.close()


def plot_spatial_flux_results_or_diff_to_prior(savepath,  Inversion, week_min, week_max,alpha, diff =False):
    factor = 12*10**6
    plt.rcParams.update({'font.size':15})   
    print(Inversion.predictions['bioclass'])
    predictionsCO2 = Inversion.predictions[:,Inversion.predictions['bioclass']<Inversion.predictions['bioclass'].shape[0]/2]
    predictionsCO = Inversion.predictions[:,Inversion.predictions['bioclass']>=Inversion.predictions['bioclass'].shape[0]/2]
    predictionsCO = predictionsCO.assign_coords(bioclass = np.arange(0,len(predictionsCO['bioclass'])))

    for week in range(week_min,week_max+1):  
        spatial_resultCO2 = Inversion.map_on_grid(predictionsCO2[(predictionsCO2['week']==week)])
        spatial_resultCO = Inversion.map_on_grid(predictionsCO[(predictionsCO['week']==week)])
        
        if diff== True: 

            spatial_fluxCO2 = Inversion.map_on_grid(Inversion.flux[:int((len(Inversion.flux.bioclass.values))/2),Inversion.flux['week']==week])
            spatial_fluxCO2 = spatial_fluxCO2 *factor
            spatial_resultCO2 = (spatial_resultCO2*factor)-spatial_fluxCO2
            savename = str("{:e}".format(alpha))+'_CO2_Diff_to_prior_week_'+str(week)+'.png'
            plot_spatial_result(spatial_resultCO2, "CO2", savepath, savename)
           
            spatial_fluxCO = Inversion.map_on_grid(Inversion.flux[int((len(Inversion.flux.bioclass.values))/2):,Inversion.flux['week']==week])
            spatial_fluxCO = spatial_fluxCO *factor
            spatial_resultCO = (spatial_resultCO*factor)-spatial_fluxCO
            savename = str("{:e}".format(alpha))+'_CO_Diff_to_prior_week_'+str(week)+'.png'
            plot_spatial_result(spatial_resultCO, "CO", savepath, savename)
          
        else: 
          
            spatial_resultCO = (spatial_resultCO*factor)
            spatial_resultCO2 = (spatial_resultCO2*factor) 

            savename = str("{:e}".format(alpha))+"_Spatial_results_CO_"+str(week)+".png"
            plot_spatial_result(spatial_resultCO,"CO", savepath, savename)

            savename = str("{:e}".format(alpha))+"_Spatial_results_CO2_"+str(week)+".png"
            plot_spatial_result(spatial_resultCO2, "CO2", savepath, savename)

    #total_spatial_result.to_netcdf(path =savepath+'spatial_results_week_'+str(week_min)+'_'+str(week_max)+'.pkl')
