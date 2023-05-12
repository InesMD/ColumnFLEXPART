#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:20:03 2022

@author: eschoema
"""

import geopandas
import pandas as pd
import numpy as np
from functions import getGdfOfRectGrid, getAreaOfGrid
import xarray as xr

Lat_min = -54
Lat_max = -1
Long_min = 10#101
Long_max = 179#164

#read shapefiles with polygons of the oekoregions in AUstralia
reg0 = geopandas.read_file('/home/b/b382105/ColumnFLEXPART/resources/Oekoregionen_PolygoneAU.shp')
reg = reg0.to_crs(4326) #transform to correct coordinate reference system

#get 1x1 grid to Australian region
try:
    print('get grid') 
    grid = pd.read_pickle("/home/b/b382105/ColumnFLEXPART/resources/Grid1_1"+'_'+str(Lat_min)+'_'+str(Lat_max)+'_'+str(Long_min)+'_'+str(Long_max)+".pkl")
except:
    print('create grid')
    grid = getGdfOfRectGrid(Lat_min,Lat_max,Long_min,Long_max)

#spatial selection of gird cells in individual oekoregions    
gdf = geopandas.overlay(grid, reg, how='difference')
gdf.insert(loc = 1, value = np.zeros(len(gdf.Lat)), column = 'bioclass')

for i in range(int(reg.BioKlassen.min()), int(reg.BioKlassen.max())+1):
    grid2= geopandas.overlay(grid, reg[(reg.BioKlassen == i)], how='intersection')    
    grid2.insert(loc = 1, value = np.ones(len(grid2.Lat))*i, column = 'bioclass')
    gdf = gdf.append(grid2)
    del grid2
    
gdf = gdf[(gdf.duplicated(subset=['Lat','Long'],keep='last') == False)] # remove dublicates
gdf = gdf.drop(columns = ['index','GridID', 'AreaGrid', 'geomPoint', 'geomPoly','Shape_Leng', 'Shape_Area', 'BioKlassen']).reset_index()

## further split up class 1,4,6 into the spatially separated regions and region 3 into two pieces
gdf.insert(loc=1,column='bioclass0',value=gdf.bioclass)
gdf.loc[(gdf.bioclass == 1)&(gdf.Long > 130),['bioclass0']] = 7
gdf.loc[(gdf.bioclass == 3)&(gdf.Lat < -21),['bioclass0']] = 8
gdf.loc[(gdf.bioclass == 4)&(gdf.Long < 130),['bioclass0']] = 9
gdf.loc[(gdf.bioclass == 4)&(gdf.Long > 140),['bioclass0']] = 10
#gdf.loc[(gdf.bioclass == 6)&(gdf.Lat < -40),['bioclass0']] = 11

#version 0, split region 2 into 4 pieces
gdf0 = gdf.copy()
gdf0 = gdf0.drop(columns='bioclass')
gdf0.insert(loc=1,column='bioclass',value=gdf0.bioclass0)
gdf0.loc[(gdf0.bioclass0 == 2)&(gdf0.Lat < -27)&(gdf0.Long < 131),['bioclass']] = 12
gdf0.loc[(gdf0.bioclass0 == 2)&(gdf0.Lat >= -27)&(gdf0.Long < 131),['bioclass']] = 13
gdf0.loc[(gdf0.bioclass0 == 2)&(gdf0.Lat >= -27)&(gdf0.Long >= 131),['bioclass']] = 14

#version 1, split region 2 into 6 pieces
gdf1 = gdf.copy()
gdf1 = gdf1.drop(columns='bioclass')
gdf1.insert(loc=1,column='bioclass',value=gdf1.bioclass0)
gdf1.loc[(gdf1.bioclass0 == 2)&(gdf1.Lat < -27)&(gdf1.Long < 125),['bioclass']] = 12
gdf1.loc[(gdf1.bioclass0 == 2)&(gdf1.Lat < -27)&(gdf1.Long >= 125)&(gdf1.Long < 136),['bioclass']] = 13
gdf1.loc[(gdf1.bioclass0 == 2)&(gdf1.Lat < -27)&(gdf1.Long >= 136),['bioclass']] = 14
gdf1.loc[(gdf1.bioclass0 == 2)&(gdf1.Lat >= -27)&(gdf1.Long < 125),['bioclass']] = 15
gdf1.loc[(gdf1.bioclass0 == 2)&(gdf1.Lat >= -27)&(gdf1.Long >= 136),['bioclass']] = 16

#version 2, split region 2 into 9 pieces
gdf2 = gdf.copy()
gdf2 = gdf2.drop(columns='bioclass')
gdf2.insert(loc=1,column='bioclass',value=gdf2.bioclass0)
gdf2.loc[(gdf2.bioclass0 == 2)&(gdf2.Lat < -30)&(gdf2.Long < 125),['bioclass']] = 12
gdf2.loc[(gdf2.bioclass0 == 2)&(gdf2.Lat < -30)&(gdf2.Long >= 125)&(gdf2.Long < 136),['bioclass']] = 13
gdf2.loc[(gdf2.bioclass0 == 2)&(gdf2.Lat < -30)&(gdf2.Long >= 136),['bioclass']] = 14
gdf2.loc[(gdf2.bioclass0 == 2)&(gdf2.Lat >= -25)&(gdf2.Long < 125),['bioclass']] = 15
gdf2.loc[(gdf2.bioclass0 == 2)&(gdf2.Lat >= -25)&(gdf2.Long >= 125)&(gdf2.Long < 136),['bioclass']] = 16
gdf2.loc[(gdf2.bioclass0 == 2)&(gdf2.Lat >= -25)&(gdf2.Long >= 136),['bioclass']] = 17
gdf2.loc[(gdf2.bioclass0 == 2)&(gdf2.Lat >= -30)&(gdf2.Lat < -25)&(gdf2.Long < 125),['bioclass']] = 18
gdf2.loc[(gdf2.bioclass0 == 2)&(gdf2.Lat >= -30)&(gdf2.Lat < -25)&(gdf2.Long >= 136),['bioclass']] = 19

#version 3, split region 2 into 3 lat pieces
gdf3 = gdf.copy()
gdf3 = gdf3.drop(columns='bioclass')
gdf3.insert(loc=1,column='bioclass',value=gdf3.bioclass0)
gdf3.loc[(gdf3.bioclass0 == 2)&(gdf3.Lat < -30),['bioclass']] = 12
gdf3.loc[(gdf3.bioclass0 == 2)&(gdf3.Lat >= -25),['bioclass']] = 13

#version 2, split region 2 into 16 pieces and region 3 in 2 pieces
gdf4 = gdf.copy()
gdf4 = gdf4.drop(columns='bioclass')
gdf4.insert(loc=1,column='bioclass',value=gdf4.bioclass0)
Latcut=[-40,-31,-27,-23,-10]
Longcut = [110,121,131,138,150]
gdf4.loc[(gdf4.bioclass0 == 3)&(gdf4.Long < 136),['bioclass']] = 12
num = 13
for j in range(len(Latcut)-1):
    for k in range(len(Longcut)-1):
        gdf4.loc[(gdf4.bioclass0 == 2)&
                 (gdf4.Lat >= Latcut[j])&
                 (gdf4.Lat < Latcut[j+1])&
                 (gdf4.Long >= Longcut[k])&
                 (gdf4.Long < Longcut[k+1]),['bioclass']] = num
        num = num + 1
gdf4.loc[(gdf4.bioclass0 == num-1),['bioclass']] = 2

# version 4 but with region 6 splitted into 4 pieces
gdf5 = gdf4.copy()
gdf5.loc[(gdf5.bioclass == 6)&(gdf5.Lat > -32),['bioclass']] = 29
gdf5.loc[(gdf5.bioclass == 6)&(gdf5.Lat < -40),['bioclass']] = 30
gdf5.loc[(gdf5.bioclass == 6)&(gdf5.Long >= 147),['bioclass']] = 31

# version 4 but with region 6 splitted into 4 pieces
gdf5 = gdf4.copy()
gdf5.loc[(gdf5.bioclass == 6)&(gdf5.Lat > -32),['bioclass']] = 29
gdf5.loc[(gdf5.bioclass == 6)&(gdf5.Lat < -40),['bioclass']] = 30
gdf5.loc[(gdf5.bioclass == 6)&(gdf5.Long >= 147),['bioclass']] = 31

# ecoregions, but take region 5,6,7,8,16,29,30,31 of version 4 seperately
gdf6 = gdf5.copy()

gdf6.loc[(gdf6.bioclass == 16),['bioclass0']] = 16
gdf6.loc[(gdf6.bioclass0 == 2),['bioclass']] = 2
gdf6.loc[(gdf6.bioclass0 == 3),['bioclass']] = 3
gdf6.loc[(gdf6.bioclass0 == 4)|(gdf6.bioclass0 == 9)|(gdf6.bioclass0 == 10),['bioclass']] = 4

gdf6.loc[(gdf6.bioclass == 29),['bioclass']] = 10
gdf6.loc[(gdf6.bioclass == 30),['bioclass']] = 11
gdf6.loc[(gdf6.bioclass == 31),['bioclass']] = 12
gdf6.loc[(gdf6.bioclass == 16),['bioclass']] = 9

# like version 6 but additonally subdividing region and adding region from version 4
gdf7 = gdf5.copy()

gdf7.loc[(gdf7.bioclass == 16),['bioclass0']] = 16
gdf7.loc[(gdf7.bioclass == 20),['bioclass0']] = 20
gdf7.loc[(gdf7.bioclass0 == 2),['bioclass']] = 2
gdf7.loc[(gdf7.bioclass0 == 3),['bioclass']] = 3
gdf7.loc[(gdf7.bioclass0 == 4)|(gdf7.bioclass0 == 9)|(gdf7.bioclass0 == 10),['bioclass']] = 4

gdf7.loc[(gdf7.bioclass == 29),['bioclass']] = 10
gdf7.loc[(gdf7.bioclass == 30),['bioclass']] = 11
gdf7.loc[(gdf7.bioclass == 31),['bioclass']] = 12
gdf7.loc[(gdf7.bioclass == 16),['bioclass']] = 9
gdf7.loc[(gdf7.bioclass == 20),['bioclass']] = 14
gdf7.loc[(gdf7.bioclass == 12)&(gdf7.Long >= 149),['bioclass']] = 13

#all gridcell own bioclass
gdf8 = gdf0.copy()
j = 0
for lat in range(-1,-54,-1):
    for long in range(10,180):
        if gdf8.loc[(gdf8.Lat == lat + 0.5)&(gdf8.Long == long + 0.5),['bioclass']].values[0][0] != 0:
            gdf8.loc[(gdf8.Lat == lat + 0.5)&(gdf8.Long == long + 0.5),['bioclass']] = j
            j += 1

gdf9 = gdf0[(gdf0.Lat >= -45)&(gdf0.Lat <= -10)&(gdf0.Long >= 110)&(gdf0.Long <= 155)].copy()
gdf9.loc[(gdf9.bioclass >= 0),['bioclass']] = 1

gdf10 = gdf0[(gdf0.Lat >= -48)&(gdf0.Lat <= -7)&(gdf0.Long >= 107)&(gdf0.Long <= 158)].copy()
gdf10.loc[(gdf10.bioclass >= 0),['bioclass']] = 1
gdf10.plot()

gdf11 = gdf8.copy()
#for overview
#pivot14 = pd.pivot(gdf14, values='bioclass', index='Lat', columns='Long')
gdf11.drop(columns = 'bioclass0', inplace= True)
gdf11.insert(loc=1, column = 'bioclass0',value=np.zeros(len(gdf11)))    
gdf11.loc[(gdf11.bioclass >= 693)&(gdf11.bioclass <= 698),['bioclass0']] = 1
gdf11.loc[(gdf11.bioclass >= 653)&(gdf11.bioclass <= 654)|
          (gdf11.bioclass >= 660)&(gdf11.bioclass <= 667)|
          (gdf11.bioclass >= 671)&(gdf11.bioclass <= 677)|
          (gdf11.bioclass >= 681)&(gdf11.bioclass <= 687)|
          (gdf11.bioclass >= 691)&(gdf11.bioclass <= 692)
        ,['bioclass0']] = 2
gdf11.loc[(gdf11.bioclass >= 688)&(gdf11.bioclass <= 690)|
          (gdf11.bioclass >= 678)&(gdf11.bioclass <= 680)|
          (gdf11.bioclass >= 668)&(gdf11.bioclass <= 670)|
          (gdf11.bioclass == 655)
        ,['bioclass0']] = 3
gdf11.loc[(gdf11.bioclass >= 656)&(gdf11.bioclass <= 658)|
          (gdf11.bioclass >= 636)&(gdf11.bioclass <= 638)|
          (gdf11.bioclass >= 612)&(gdf11.bioclass <= 615)
        ,['bioclass0']] = 4        
gdf11.loc[(gdf11.bioclass >= 583)&(gdf11.bioclass <= 586)|
          (gdf11.bioclass >= 548)&(gdf11.bioclass <= 549)
        ,['bioclass0']] = 5   
gdf11.loc[(gdf11.bioclass >= 546)&(gdf11.bioclass <= 547)|
          (gdf11.bioclass >= 507)&(gdf11.bioclass <= 510)|
          (gdf11.bioclass >= 470)&(gdf11.bioclass <= 471)
        ,['bioclass0']] = 6   
gdf11.loc[(gdf11.bioclass >= 649)&(gdf11.bioclass <= 652)|
          (gdf11.bioclass >= 631)&(gdf11.bioclass <= 635)|
          (gdf11.bioclass >= 606)&(gdf11.bioclass <= 611)|
          (gdf11.bioclass >= 577)&(gdf11.bioclass <= 582)|
          (gdf11.bioclass >= 542)&(gdf11.bioclass <= 545)
        ,['bioclass0']] = 7
gdf11.loc[(gdf11.bioclass == 505)|(gdf11.bioclass == 506)|
          (gdf11.bioclass >= 463)&(gdf11.bioclass <= 469)|
          (gdf11.bioclass >= 424)&(gdf11.bioclass <= 432)|
          (gdf11.bioclass >= 386)&(gdf11.bioclass <= 393)|
          (gdf11.bioclass >= 350)&(gdf11.bioclass <= 354)
        ,['bioclass0']] = 8
gdf12 = gdf11.copy(deep=True)

gdf11.loc[(gdf11.bioclass0 == 0)&(gdf11.bioclass > 0)&
          (gdf11.Long >= 147)&(gdf11.Long <= 153)
        ,['bioclass0']] = 9  
gdf11.drop(columns= 'bioclass',inplace=True)
gdf11.rename(columns={'bioclass0':'bioclass'},inplace=True)


gdf12.loc[(gdf12.bioclass0 == 0)&(gdf12.bioclass > 0)
        ,['bioclass0']] = 10 
gdf12.drop(columns= 'bioclass',inplace=True)
gdf12.rename(columns={'bioclass0':'bioclass'},inplace=True)

gdf13 = gdf12.copy()
gdf13.loc[(gdf13.bioclass > 0)&(gdf13.bioclass < 10)
        ,['bioclass']] = 1  
gdf13.loc[(gdf13.bioclass == 10)
        ,['bioclass']] = 2

gdf14 = gdf12.copy()
gdf14.loc[(gdf14.bioclass == 10)&(gdf14.Long > 152)
        ,['bioclass']] = 6  

#add area
dfArea = getAreaOfGrid()
gdf120 =pd.merge(gdf12,dfArea,on = 'Lat', how = 'left')
AreasRegions = gdf120.groupby('bioclass')['Area'].sum().reset_index()

gdf130 =pd.merge(gdf13,dfArea,on = 'Lat', how = 'left')
AreasRegions13 = gdf130.groupby('bioclass')['Area'].sum().reset_index()

gdf1_0 =pd.merge(gdf1,dfArea,on = 'Lat', how = 'left')
AreasRegions1_0 = gdf1_0.groupby('bioclass')['Area'].sum().reset_index()

gdf_0 =pd.merge(gdf,dfArea,on = 'Lat', how = 'left')
AreasRegions_0 = gdf_0.groupby('bioclass')['Area'].sum().reset_index()

#pivot11_2 = pd.pivot(gdf11, values='bioclass', index='Lat', columns='Long')
'''
gdf0 = gdf0.drop(columns = ['geometry','bioclass0'])
gdf0_0 = gdf0.set_index(['Lat','Long'])
ds0 = gdf0_0.to_xarray()
#ds0.to_netcdf('/home/eschoema/OekomaskAU_Flexpart_version0_Tasmania')

gdf1 = gdf1.drop(columns = ['geometry','bioclass0'])
gdf1_0 = gdf1.set_index(['Lat','Long'])
ds1 = gdf1_0.to_xarray()
#ds1.to_netcdf('/home/eschoema/OekomaskAU_Flexpart_version1_Tasmania')

gdf2 = gdf2.drop(columns = ['geometry','bioclass0'])
gdf2_0 = gdf2.set_index(['Lat','Long'])
ds2 = gdf2_0.to_xarray()
#ds2.to_netcdf('/home/eschoema/OekomaskAU_Flexpart_version2_Tasmania')

gdf3 = gdf3.drop(columns = ['geometry','bioclass0'])
gdf3_0 = gdf3.set_index(['Lat','Long'])
ds3 = gdf3_0.to_xarray()
#ds3.to_netcdf('/home/eschoema/OekomaskAU_Flexpart_version3_Tasmania')

gdf4 = gdf4.drop(columns = ['geometry','bioclass0'])
gdf4_0 = gdf4.set_index(['Lat','Long'])
ds4 = gdf4_0.to_xarray()
#ds4.to_netcdf('/home/eschoema/OekomaskAU_Flexpart_version4_Tasmania')

gdf5 = gdf5.drop(columns = ['geometry','bioclass0'])
gdf5_0 = gdf5.set_index(['Lat','Long'])
ds5 = gdf5_0.to_xarray()
ds5.to_netcdf('/home/eschoema/OekomaskAU_Flexpart_version5_Tasmania')

gdf6 = gdf6.drop(columns = ['geometry','bioclass0'])
gdf6_0 = gdf6.set_index(['Lat','Long'])
ds6 = gdf6_0.to_xarray()
ds6.to_netcdf('/home/eschoema/OekomaskAU_Flexpart_version6_Tasmania')

gdf7 = gdf7.drop(columns = ['geometry','bioclass0'])
gdf7_0 = gdf7.set_index(['Lat','Long'])
ds7 = gdf7_0.to_xarray()
ds7.to_netcdf('/home/eschoema/OekomaskAU_Flexpart_version7_Tasmania')

gdf8 = gdf8.drop(columns = ['geometry','bioclass0'])
gdf8_0 = gdf8.set_index(['Lat','Long'])
ds8 = gdf8_0.to_xarray()
ds8.to_netcdf('/home/eschoema/OekomaskAU_Flexpart_version8_all1x1')

gdf9 = gdf9.drop(columns = ['geometry','bioclass0'])
gdf9_0 = gdf9.set_index(['Lat','Long'])
ds9 = gdf9_0.to_xarray()
ds9.to_netcdf('/home/eschoema/OekomaskAU_innerAU_Box')

gdf10 = gdf10.drop(columns = ['geometry','bioclass0'])
gdf10_0 = gdf10.set_index(['Lat','Long'])
ds10 = gdf10_0.to_xarray()
ds10.to_netcdf('/home/eschoema/OekomaskAU_innerAU_Box_buffer3degrees')

gdf11 = gdf11.drop(columns = ['geometry'])
gdf11_0 = gdf11.set_index(['Lat','Long'])
ds11 = gdf11_0.to_xarray()
ds11.to_netcdf('/home/eschoema/OekomaskAU_AKbased_1')

gdf12 = gdf12.drop(columns = ['geometry'])
gdf12_0 = gdf12.set_index(['Lat','Long'])
ds12 = gdf12_0.to_xarray()
ds12.to_netcdf('/home/eschoema/OekomaskAU_AKbased_2')

gdf13 = gdf13.drop(columns = ['geometry'])
gdf13_0 = gdf13.set_index(['Lat','Long'])
ds13 = gdf13_0.to_xarray()
ds13.to_netcdf('/home/eschoema/OekomaskAU_AKbased_3')

gdf14 = gdf14.drop(columns = ['geometry'])
gdf14_0 = gdf14.set_index(['Lat','Long'])
ds14 = gdf14_0.to_xarray()
ds14.to_netcdf('/home/eschoema/OekomaskAU_AKbased_4')

'''