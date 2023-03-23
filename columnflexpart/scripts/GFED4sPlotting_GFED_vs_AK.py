
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from columnflexpart.utils import select_boundary, optimal_lambda
from columnflexpart.classes.flexdataset import FlexDataset
from functools import partial
import geopandas as gpd
import cartopy.crs as ccrs
from typing import Optional, Literal, Union, Callable, Iterable, Any
from pathlib import Path
import bayesinverse
from columnflexpart.classes.inversion import InversionBioclass
from columnflexpart.utils.utils import optimal_lambda
from matplotlib.colors import LogNorm 
from matplotlib.colors import LinearSegmentedColormap
#from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D




def plot_averaging_kernel(Inversion, alpha, class_num, week_num,savepath, plot_spatially, weekly = False):
    # get ak 
    print('Calculating Averaging Kernel')
    flux_mean, flux_err = Inversion.get_flux()
    err = Inversion.get_land_ocean_error(1/10)
    predictions = Inversion.fit(alpha = alpha, xerr = err) 
    ak = Inversion.get_averaging_kernel()
    plt.rcParams.update({'font.size':13})
    #sum ak 
    ak_sum = np.zeros((week_num*class_num,class_num))   
    for i in range(0,class_num):#ecosystem number 
        list_indices = list(i+class_num*np.arange(0,week_num))
        for idx in list_indices: 
            ak_sum[:,i] += ak[:,idx]

    ak_final = np.zeros((class_num,class_num))
    for i in range(0,class_num):
        list_indices = list(i+class_num*np.arange(0,week_num))
        for idx in list_indices:
            ak_final[i] += ak_sum[idx]
        ak_final[i] = ak_final[i]/len(list_indices)
    
    if plot_spatially ==  True and weekly == False: 
        #print(ak_final.diagonal())
        #ak_final = ak_final(np.where(ak_final>1e-3)[0])
        #print(ak_final)
        ak_xr = xr.DataArray(data = ak_final.diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))#, week = [1,48,49,50,51,52])))
        #print(ak_xr)
        '''
        selection = np.where(ak_xr>1e-3)[0]
        #print(np.where(ak_xr>1e-3)[0])
        ds0 = ak_xr.where(ak_xr.bioclass==selection[0], drop = True).rename('ak')
        #print(ds0)
        for i in selection[1:]: 
            dsi =ak_xr.where(ak_xr.bioclass==i, drop = True).rename('ak')
            #print(dsi)
            ds0 = xr.merge([ds0,dsi])
        ak_xr = ds0
        #print(ak_xr)
        '''
        #print(len(ak_xr))
        #print(len(ak_xr.ak))
        #print(ak_xr.ak)       
        ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, len(ak_xr))#ak_xr.ak, len(ak_xr.ak)
        print(ak_spatial)
        print('Done')
        return ak_spatial
        print(ak_spatial)
        ax = plt.axes(projection=ccrs.PlateCarree())  
        orig_map=plt.cm.get_cmap('gist_heat')
        reversed_map = orig_map.reversed()
        ak_spatial.plot(norm = LogNorm(vmin = 1e-3), x='longitude', y='latitude',ax = ax,cmap = reversed_map)
        ax.coastlines()
        plt.title('Averaging kernel for 2019/12')
        plt.savefig(savepath+'ak_spatial_gist_heat_reversed'+str(alpha)+'log_1e-3_no_ocean.png')
    #sk_shape # Seite 47
    elif plot_spatially == True and weekly == True: 
        week_list = [1,48,49,50,51,52]
        for week in range(0,week_num): 
            ak_xr = xr.DataArray(data = ak_sum[week*class_num:(week+1)*class_num].diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))
            ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, class_num)
            fig = plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())  
            orig_map=plt.cm.get_cmap('gist_heat')
            reversed_map = orig_map.reversed()
            ak_spatial.plot(x='longitude', y='latitude',ax = ax, vmax = 0.2, vmin = 0, cmap = reversed_map)
            ax.coastlines()
            plt.title('Averaging kernel for 2019 week '+str(week_list[week]))
            fig.savefig(savepath+'ak_spatial_gist_heat_reversed'+str(alpha)+'_week'+str(week_list[week])+'_no_ocean.png')

from matplotlib.collections import LineCollection
def add_iso_line(ax, x, y, z, value, **kwargs): 
    v = np.diff(z > value, axis=1) #1
    print(v)
    h = np.diff(z > value, axis=0)#0
    l = np.argwhere(v) #v.T
    print(l)
    print(x.shape)
    print(y.shape)
    print(l.shape)
    print(x[l[:, 0] + 1])
    print(y[l[:, 1]])
    vlines = np.array(list(zip(np.stack((x[l[:, 0] + 1], y[l[:, 1]])).T, #T
                                np.stack((x[l[:, 0] + 1], y[l[:, 1] + 1])).T))) #T
    l = np.argwhere(h) #h.T
    hlines = np.array(list(zip(np.stack((x[l[:, 0]], y[l[:, 1] + 1])).T,#T
                                np.stack((x[l[:, 0] + 1], y[l[:, 1] + 1])).T))) #T
    lines = np.vstack((vlines, hlines)) #vlines, hlines
    ax.add_collection(LineCollection(lines, **kwargs))



def plotFireMonthlyMeanSpatially(savepath, syear, smonth, gdf, Inversion, savefig=True):
    print('Processing GFED')
    sdf = gdf[(gdf['Year']==syear)&(gdf['Month']==smonth)].reset_index()

    #sdf = sdf[(sdf['Longround']>166)]
    #sdf1 = sdf[( sdf['Lat']<154)]
    sdf = sdf.set_index(['Long', 'Lat'])   
    #print(sdf)
    xsdf = xr.Dataset.from_dataframe(sdf)
    foo = xr.DataArray(xsdf['total_emission'][:]*12/44)#in TgC

    foo1 = foo[(foo['Long']>166)]
    foo2 = foo[( foo['Long']<154)]
    print(foo2)
    foo2 = foo2.where(foo2>0.0005)
    #foo2 = foo2[(foo2['emission']>0)]
    #print(foo2)

    ak_to_plot = plot_averaging_kernel(Inversion, 0.0006228666734243309, 699, 6,savepath, True)
    
    #ak_to_plot = ak_to_plot>1e-3
    #print(ak_to_plot.where(ak_to_plot>1e-4))
   
    ak_equal =  xr.where((ak_to_plot>=5e-2), 5e-2,  ak_to_plot)
    print('Step 1')
    print(ak_equal.min())
    print(ak_equal.max())
    ak_equal = xr.where((ak_equal<5e-2)&(ak_equal>=1e-2), 1e-2, ak_equal)
    print('Step 2')
    print(ak_equal.min())
    print(ak_equal.max())
    ak_equal = xr.where((ak_equal<1e-2)&(ak_equal>=1e-3), 1e-3,ak_equal)
    print('Step 3')
    print(ak_equal.min())
    print(ak_equal.max())
    ak_equal = ak_equal.where(ak_equal>=1e-3)#*1
    print('Step 4')
    print(ak_equal.min())
    print(ak_equal.max())
    



    plt.figure(figsize=(14, 10))
    plt.rcParams.update({'font.size': 13})    
    ax = plt.axes(projection=ccrs.PlateCarree())    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    
    cmap0 = LinearSegmentedColormap.from_list('', ['white','orange','orangered', 'firebrick', 'darkred'])#'white',
    cmap_ak = LinearSegmentedColormap.from_list('', ['white','lightgrey','silver','darkgrey',  'dimgrey'])#, 'black'])#'white',#'gray',
    #colors = ["darkorange", "gold", "lawngreen", "lightseagreen"]
    #cmap1 = LinearSegmentedColormap.from_list("", colors)

    #add_iso_line(ax = ax, x = ak_equal.longitude.values-0.5, y = ak_equal.latitude.values-0.5,
    #              z = ak_equal.values, value = 0.5, color="black", linewidth=0.2)
    #ak_equal.plot.contour(x='longitude', y = 'latitude', ax = ax, levels = 1, colors = 'k')# cmap = 'Greys',norm = LogNorm(vmin = 1e-3),cbar_kwargs=dict(orientation='vertical',
                   #   shrink=0.75, label='Averaging kernel') )
   
    ak_equal.plot(norm = LogNorm(vmin = 3e-4), x='longitude', y = 'latitude', ax = ax, alpha = 0.9,  cmap = cmap_ak, add_colorbar = False)#alpha = 0.6, #'cmap_akGreys', cbar_kwargs=dict(orientation='vertical',
               #       shrink=0.75, label='Averaging Kernel', ticks = [1e-4,1e-3, 1e-2, 5e-2]))
    print('ak_plotted')

    foo2.plot.pcolormesh(x='Long', y = 'Lat',extend = 'max', cmap = cmap0, vmax = 0.2,vmin = 0, ax = ax,  cbar_kwargs=dict(orientation='vertical',
                      shrink=0.9, label='CO$_2$ Emissions [TgC/month]', ticks = np.arange(0,0.3,0.1)))#ax= ax,  transform = ccrs.PlateCarree() cmap = 'OrRd',
    print('GFED plotted')
    #foo2.plot.pcolormesh(x='Longround', y = 'Lat',cmap = cmap0, vmax = 0.2,ax = ax, add_colorbar = False)
 

    #plt.plot([], [], ' ',color = 'lightgrey',  label=r">= 1$\cdot$10$^{-3}$")
    #plt.plot([], [], ' ',color = 'grey',  label=r">= 1$\cdot$10$^{-2}$")
    #plt.plot([], [], ' ',color = 'darkgrey', label=r">= 5$\cdot$10$^{-2}$")
    legend_elements = [Line2D([0], [0], color='grey', lw=5, label=r"$\geq 5\cdot 10^{-2}$"),
                       Line2D([0], [0], color='darkgrey', lw=5, label=r"$< 5\cdot 10^{-2} & \geq 10^{-2}$"),
                       Line2D([0], [0], color='lightgrey', lw=5, label=r"$< 10^{-2} & \geq 10^{-3}$")]
                       
    plt.legend(handles = legend_elements, title = 'Averaging kernel', loc = 'lower left')
    #bar_10_100 = ax.bar(np.arange(0,10), np.arange(30,40), bottom=np.arange(1,11), color="darkgrey")
    #bar_0_10 = ax.bar(np.arange(0,10), np.arange(1,11), color="grey")
    #bar_10_100 = ax.bar(np.arange(0,10), np.arange(30,40), bottom=np.arange(1,11), color="lightgrey")
    # create blank rectangle
    #extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    #ax.legend([extra, bar_0_10, bar_10_100], ("Averaging kernel", r">= 5$\cdot$10$^{-2}$",r">= 1$\cdot$10$^{-2}$", r">= 1$\cdot$10$^{-3}$"))

    ax.coastlines()
    plt.title('GFED CO$_2$ Emissions '+str(smonth).zfill(2)+'/'+str(syear).zfill(2), size =15)
    plt.savefig(savepath+"GFED_AK_CO2_AU_greys_no_ocean_ak_1e-3.png")
    #plt.title('Fire CO$_2$ Emissions '+str(smonth).zfill(2)+'/'+str(syear).zfill(2), size =15)
    #if savefig == True: 
    #    plt.savefig(savepath+"GFED_CO2_AK_AU_{}_{}_smallerFires.png".format(syear,smonth),facecolor='w',edgecolor='none')
    plt.show()


'''
def MonthlySumSpatially(gdf, startyear, endyear):
    gdf['date'] = pd.to_datetime(gdf['date'])
    gdf.insert(loc = 1, column = 'year', value = gdf['date'].dt.year)
    gdf.insert(loc = 1, column = 'month', value = gdf['date'].dt.month)
    gdf.insert(loc = 1, column = 'day', value = gdf['date'].dt.day)
    gdf = gdf[(gdf['year']>= startyear)&(gdf['year']<= endyear)]
    gdf['co2fire'][:] =gdf['co2fire'][:]* 10**(-9)*(24*60*60)*np.cos(np.deg2rad(gdf['latitude'][:]))*111319*111000 
    gdf['day'] = np.ones(gdf['day'].shape[0])*15
    output = gdf.groupby(['latitude', 'longitude', 'year', 'month'])['co2fire'].sum().reset_index()

    return output
'''

def MonthlySumSpatially(gdf, startyear, endyear):
    gdf['Date'] = pd.to_datetime(gdf['Date'])
    gdf.insert(loc = 1, column = 'year', value = gdf['Date'].dt.year)
    gdf.insert(loc = 1, column = 'month', value = gdf['Date'].dt.month)
    gdf.insert(loc = 1, column = 'day', value = gdf['Date'].dt.day)
    gdf = gdf[(gdf['year']>= startyear)&(gdf['year']<= endyear)]
    gdf['emission'][:] =gdf['emission'][:]* 10**(-9)*(24*60*60)*np.cos(np.deg2rad(gdf['Lat'][:]))*111319*111000 
    gdf['day'] = np.ones(gdf['day'].shape[0])*15
    output = gdf.groupby(['Lat', 'Long', 'year', 'month'])['emission'].sum().reset_index()

    return output


datapath = '/work/bb1170/RUN/b382105/Dataframes/GFEDs/'
savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/'

output = pd.read_pickle(datapath+'GDF_CO2_AU20162021.pkl')
#print(output.keys())
#output = MonthlySumSpatially(output, 2019, 2019)

savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/'

Inversion = InversionBioclass(
    result_path="/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions.pkl",
    month="2019-12", 
    flux_path="/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Flux/CT2022.flux1x1.",
    bioclass_path= "/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1",
    time_coarse = None,
    boundary=[110.0, 155.0, -45.0, -10.0],
    data_outside_month=False
)

plotFireMonthlyMeanSpatially(savepath, 2019, 12, output, Inversion, savefig=True)