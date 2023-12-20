#!/usr/bin/env python
# functions for the GOSAT Scrips
# Author: Eva-Marie Schoemann


import numpy as np
import pandas as pd
import sys
import datetime
import geopandas
import xarray as xr
#import h5py 
from shapely.geometry import Polygon
#from RegionParam import getRegion

# funtion to create a 2dim array with the gas concentration values
# input:
# -- df: pandas dataframe containing columns with latitude, longitude and gas concentrations
# -- latname: name of the columne containing the latitude
# -- longname: name of the columne containing the longitude
# -- valuename: name of the columne containing the concentrations
#
# output:
# -- xyarray: numpy 2D array(latitude, longitude) with the concentration values
def createXYArray(df, latname, longname, valuename):
    df["Lat_round"] = np.floor(np.array(df[latname])) 
    df["Long_round"] = np.floor(np.array(df[longname]))
    space = df.groupby(['Lat_round','Long_round'])[valuename].mean().reset_index()
    xyarray = np.zeros([180,360])
    for j in range(-90,90,1):
        print(str(j+90)+" of 180")
        for i in range(-180,180,1):
            if list(space.loc[(space.Lat_round == j) & (space.Long_round == i)][valuename].values) == []:
                xyarray[89-j][i+180] = np.nan
            else:
                val = list(space.loc[(space.Lat_round == j) & (space.Long_round == i)][valuename].values)
                xyarray[89-j][i+180] = val[0]
    df.drop(columns=['Lat_round', 'Long_round'])
    
    return xyarray

# function to create a pandas dataframe out of a dictionary from a .out file
# input:
# -- dic: dictionary from .out file created by "read_output_file" function of the "read_remotec_out" paackage
#
# output:
# -- df: pandas DataFrame containing the variables 
# -- -- CO2 and CH4 from the 'REMOTEC-OUTPUT_X_COLUMN' as 'CO2' and 'CH4', 
# -- -- 'GOSAT_TIME_DAY' as 'Day'
# -- -- 'GOSAT_TIME_MONTH', as 'Month'
# -- -- 'GOSAT_TIME_YEAR' as 'Year'
# -- -- 'GOSAT_LATITUDE' as 'Lat'
# -- -- 'GOSAT_LONGITUDE' as 'Long'
# -- -- the down rounded latitude as 'Lat_round'
# -- -- the down rounded longitude as 'Long_round'
    
def createPandasDateFrame(dic):
    
    d = {'CO2': dic['REMOTEC-OUTPUT_X_COLUMN_CORR'].T[0], 
         'CH4': dic['REMOTEC-OUTPUT_X_COLUMN_CORR'].T[1],
         'CO2_uncorr': dic['REMOTEC-OUTPUT_X_COLUMN'].T[0],
         'CH4_uncorr': dic['REMOTEC-OUTPUT_X_COLUMN'].T[1],
         'Sec': dic['GOSAT_TIME_SEC'],
         'Min': dic['GOSAT_TIME_MIN'], 
         'Hour': dic['GOSAT_TIME_HOUR'],
         'Day': dic['GOSAT_TIME_DAY'], 
         'Month': dic['GOSAT_TIME_MONTH'],
         'Year': dic['GOSAT_TIME_YEAR'],
         'Lat':dic['GOSAT_LATITUDE'],
         'Long':dic['GOSAT_LONGITUDE'],
         'CT_CO2': dic['REMOTEC-OUTPUT_X_APR_COLUMN'].T[0],
         'CT_CH4': dic['REMOTEC-OUTPUT_X_APR_COLUMN'].T[1],
         'CT_error': dic['REMOTEC-OUTPUT_X_APR_COLUMN_ERR'].T[0],
         'CO2_error': dic['REMOTEC-OUTPUT_X_COLUMN_ERR'].T[0],
         'meas_geom' : dic['(0=NADIR,1=GLINT)'],
         'quality' : dic['REMOTEC-OUTPUT_FLAG_QUALITY'], #NICHT VERWENDEN, DA NICHT ZUVERLÄSSIG
         'gain': dic["GOSAT_GAIN"]}
    df = pd.DataFrame(data=d)
    
    df["Lat_round"] = np.floor(np.array(df["Lat"])) 
    df["Long_round"] = np.floor(np.array(df["Long"]))

    return df
            

def getDayMeans(gdf,
                 year_min,month_min,day_min,
                 year_max,month_max,day_max,
                 Long_min,Long_max,
                 Lat_min,Lat_max,
                 ValueName,Error_name = '',UnCorr_name=''):
    """get the mean of all values of one day

    """
    output_all = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month','Day'])[ValueName].mean().reset_index()

    output = output_all.copy(); print('Caution, min number of mean value = 0')
    #output = output_all[(output_all.number >= 10)]
    print(len(output_all.Year))
    print(len(output.Year))
    date = output.apply(lambda x: datetime.date(int(x.Year),int(x.Month),int(x.Day)),axis=1)
    output.insert(loc=1,column='Date',value=date)
    return output


#get the mean of all values of one day
def getDaySum(gdf,
                 year_min,month_min,day_min,
                 year_max,month_max,day_max,
                 Long_min,Long_max,
                 Lat_min,Lat_max,
                 ValueName,Error_name = '',UnCorr_name=''):
    output_all = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month','Day'])[ValueName].sum().reset_index()

    output = output_all.copy(); print('Caution, min number of mean value = 0')
    #output = output_all[(output_all.number >= 10)]
    print(len(output_all.Year))
    print(len(output.Year))
    date = output.apply(lambda x: datetime.date(int(x.Year),int(x.Month),int(x.Day)),axis=1)
    output.insert(loc=1,column='Date',value=date)
    return output

# function to create monthly means of CO2 for different paper
# input:
# -- gdf: Geodataframe containing the data
# -- year_max, month_max, day_max: enddate
# -- year_min,month_min, day_min: startdate
# -- Long_min, Long_max: Longitude range
# -- Lat_min, Lat_max: Latitude range
#
# output:
# -- gdf with monthly CO2 values

def getMonthMeans(gdf,year_min,month_min,day_min,year_max,month_max,day_max,Long_min,Long_max,Lat_min,Lat_max,ValueName,Error_name = '',UnCorr_name=''):
    output_mean = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month'])[ValueName].mean().reset_index()
    output_count = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month'])[ValueName].count().reset_index()
    output_count.rename(columns = {ValueName:'number'}, inplace = True)
    output_all = pd.merge(output_mean, output_count, on=['Year','Month'])
    '''
    if 'SIF' not in ValueName:
        output_count1 = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Hour >= 0)
                        & (gdf.Hour < 6)
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month'])[ValueName].count().reset_index()
        output_count1.rename(columns = {ValueName:'number_0_6'}, inplace = True)
        output_all = pd.merge(output_all, output_count1,how='left', on=['Year','Month'])

        output_count2 = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Hour >= 6)
                        & (gdf.Hour < 12)
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month'])[ValueName].count().reset_index()
        output_count2.rename(columns = {ValueName:'number_6_12'}, inplace = True)
        output_all = pd.merge(output_all, output_count2,how='left', on=['Year','Month'])


        output_count3 = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Hour >= 12)
                        & (gdf.Hour < 19)
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month'])[ValueName].count().reset_index()
        output_count3.rename(columns = {ValueName:'number_12_18'}, inplace = True)
        output_all = pd.merge(output_all, output_count3,how='left', on=['Year','Month'])


        output_count4 = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Hour >= 18)
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month'])[ValueName].count().reset_index()
        output_count4.rename(columns = {ValueName:'number_18_24'}, inplace = True)
        output_all = pd.merge(output_all, output_count4,how='left', on=['Year','Month'])
   '''
    

    output_std = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month'])[ValueName].std().reset_index()
    output_std.rename(columns = {ValueName:'StDev'}, inplace = True)
    output_all = pd.merge(output_all, output_std, on=['Year','Month'])
    output_all.insert(loc=1,column='StError',value= output_all.StDev/np.sqrt(output_all.number))

    if Error_name:
        output_error_all = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)]
        output_error_all.insert(loc=1,column='error_sq',value= output_error_all[Error_name]*output_error_all[Error_name])
        output_error = output_error_all.groupby(['Year','Month'])['error_sq'].sum().reset_index()
        output_all = pd.merge(output_all, output_error, on=['Year','Month']) 
        output_all.insert(loc=1,column=Error_name,value=np.sqrt(output_all.error_sq)/output_all.number)

    if UnCorr_name:
        output_raw_mean = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month'])[UnCorr_name].mean().reset_index()
        output_all = pd.merge(output_all, output_raw_mean, on=['Year','Month'])
    
    output = output_all.copy(); print('Caution, min number of mean value = 0')
    #output = output_all[(output_all.number >= 10)]
    
    print(len(output_all.Year))
    print(len(output.Year))



    return output



# function to create monthly means of TCCON CO2 
# input:
# -- gdf: Geodataframe containing the data
# -- year_max, month_max, day_max: enddate
# -- year_min,month_min, day_min: startdate
#
# output:
# -- gdf with monthly CO2 values

def getMonthMeansTCCON(gdf,year_min,month_min,day_min,year_max,month_max,day_max,ValueName,minMeas,Error_name = '',UnCorr_name=''):
    output_mean = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))].groupby(['Year','Month'])[ValueName].mean().reset_index()
    #print(output_mean)
    output_count = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))].groupby(['Year','Month'])[ValueName].count().reset_index()
    
    output_Ndays = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))].groupby(['Year','Month'])['Day'].unique().reset_index()
    output_Ndays.insert(loc=1,column='numberDays',value=output_Ndays.Day.apply(lambda x: len(x)))
    output_Ndays.drop(['Day'],axis=1)

    #print(output_count)
    output_count.rename(columns = {ValueName:'number'}, inplace = True)
    output_all = pd.merge(output_mean, output_count, on=['Year','Month'])
    output_all = pd.merge(output_all, output_Ndays, on=['Year','Month'])
    output_std = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))].groupby(['Year','Month'])[ValueName].std().reset_index()
    output_std.rename(columns = {ValueName:'StDev'}, inplace = True)
    output_all = pd.merge(output_all, output_std, on=['Year','Month'])
    output_all.insert(loc=1,column='StError',value= output_all.StDev/np.sqrt(output_all.number))

    if Error_name:
        output_error_all = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))]
        output_error_all.insert(loc=1,column='error_sq',value= output_error_all[Error_name]*output_error_all[Error_name])
        output_error = output_error_all.groupby(['Year','Month'])['error_sq'].sum().reset_index()
        output_all = pd.merge(output_all, output_error, on=['Year','Month'])
        output_all.insert(loc=1,column=Error_name,value=np.sqrt(output_all.error_sq)/output_all.number)



    output = output_all[(output_all.number >= minMeas)]
    output = output_all[(output_all.numberDays >= 11)]
    print('Caution, min number of days per mean = 10')
    #print('Caution, min number of mean value = 0')
    return output

# function to create monthly sum of indicated column inside given borders
# input:
# -- gdf: Geodataframe containing the data
# -- year_max, month_max, day_max: enddate
# -- year_min,month_min, day_min: startdate
# -- Long_min, Long_max: Longitude range
# -- Lat_min, Lat_max: Latitude range
# -- ValueName: Name of column to sum up
#
# output:
# -- gdf with monthly summed values

def getMonthSum(gdf,year_min,month_min,day_min,year_max,month_max,day_max,Long_min,Long_max,Lat_min,Lat_max,ValueName,Errorname = ''):
   
    output_mean = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month'])[ValueName].sum().reset_index()

    output_count = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month'])[ValueName].count().reset_index()
    output_count.rename(columns = {ValueName:'number'}, inplace = True)
    output_all = pd.merge(output_mean, output_count, on=['Year','Month'])
    if len(Errorname) > 1:
        output_error_all = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)]
        output_error_all.insert(loc=1,column='error_sq',value= output_error_all[Errorname]*output_error_all[Errorname])
        output_error = output_error_all.groupby(['Year','Month'])['error_sq'].sum().reset_index()
        output_all = pd.merge(output_all, output_error, on=['Year','Month']) 
        output_all.insert(loc=1,column=Errorname,value=np.sqrt(output_all.error_sq))
        output_error2 = output_error_all.groupby(['Year','Month'])[Errorname].mean().reset_index()
        output_error2 = output_error2.rename(columns={Errorname:Errorname+'mean'})
        output_all = pd.merge(output_all, output_error2, on=['Year','Month']) 
    
    output = output_all[(output_all.number > 0)]



    return output

# function to create monthly sum of indicated column inside given borders
# input:
# -- gdf: Geodataframe containing the data
# -- year_max, month_max, day_max: enddate
# -- year_min,month_min, day_min: startdate
# -- Long_min, Long_max: Longitude range
# -- Lat_min, Lat_max: Latitude range
# -- ValueName: Name of column to sum up
#
# output:
# -- gdf with monthly summed values

def getMonthSumFCday(gdf,year_min,month_min,day_min,year_max,month_max,day_max,Long_min,Long_max,Lat_min,Lat_max,ValueName):
   
    output_mean2 = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month','Day'])[ValueName].mean().reset_index()

    output_mean = output_mean2.groupby(['Year','Month'])[ValueName].sum().reset_index()


    output_count = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','Month'])[ValueName].count().reset_index()
    output_count.rename(columns = {ValueName:'number'}, inplace = True)
    output_all = pd.merge(output_mean, output_count, on=['Year','Month'])
    output = output_all[(output_all.number > 0)]



    return output


# function to create 10-day means of CO2 for Guerlet paper
# input:
# -- gdf: Geodataframe containing the data
# -- year_max, month_max, day_max: enddate
# -- year_min,month_min, day_min: startdate
# -- Long_min, Long_max: Longitude range
# -- Lat_min, Lat_max: Latitude range
#
# output:
# -- gdf with 10day CO2 values

def getTenDayMeans(gdf,year_min,month_min,day_min,year_max,month_max,day_max,Long_min,Long_max,Lat_min,Lat_max,ValueName):
    ten_num = []
    print('start 10 means')
    for i in range(len(gdf.Day)):
        if gdf.Day.iloc[i]/10 >= 3:
            ten_num.append((gdf.Month.iloc[i]-1)*3+2)
        else:
            ten_num.append((gdf.Month.iloc[i]-1)*3+(gdf.Day.iloc[i]//10))
    gdf['ten_num'] = ten_num
    output_mean = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','ten_num'])[ValueName].mean().reset_index()

    output_count = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','ten_num'])[ValueName].count().reset_index()
    output_count.rename(columns = {ValueName:'number'}, inplace = True)
    output = pd.merge(output_mean, output_count, on=['Year','ten_num'])

    m = []
    for j in range(len(output)):
        m.append((output.ten_num.iloc[j]//3) + 1)
    output['Month'] = m

    TenDate = []
    for k in range(len(output)):
        try:
            TenDate.append(datetime.date(int(output.Year.iloc[k]),int(output.Month.iloc[k]),(output.ten_num.iloc[k]-(3*(output.Month.iloc[k]-1)))*10+5))
        except:
            print(output.Year.iloc[k])
            print(output.Month.iloc[k])
            TenDate.append(datetime.date(int(output.Year.iloc[k]),int(output.Month.iloc[k]),(output.ten_num.iloc[k]-(3*(output.Month.iloc[k]-1)))*10+5))
    output['TenDate'] = TenDate

    output[ValueName][(output.number <= 10)] = np.nan

    return output


# function to create 10-day sum
# input:
# -- gdf: Geodataframe containing the data
# -- year_max, month_max, day_max: enddate
# -- year_min,month_min, day_min: startdate
# -- Long_min, Long_max: Longitude range
# -- Lat_min, Lat_max: Latitude range
#
# output:
# -- gdf with 10day CO2 values

def getTenDaySum(gdf,year_min,month_min,day_min,year_max,month_max,day_max,Long_min,Long_max,Lat_min,Lat_max,ValueName):
    ten_num = []
    print('start 10 means')
    for i in range(len(gdf.Date)):
        if gdf.Date.iloc[i].day/10 >= 3:
            ten_num.append((gdf.Month.iloc[i]-1)*3+2)
        else:
            ten_num.append((gdf.Month.iloc[i]-1)*3+(gdf.Date.iloc[i].day//10))
    gdf['ten_num'] = ten_num
    output_mean = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','ten_num'])[ValueName].sum().reset_index()

    output_count = gdf[(gdf.Date >= datetime.date(year_min,month_min,day_min))
                        & (gdf.Date <= datetime.date(year_max,month_max,day_max))
                        & (gdf.Long >= Long_min)
                        & (gdf.Long <= Long_max)
                        & (gdf.Lat >= Lat_min)
                        & (gdf.Lat <= Lat_max)].groupby(['Year','ten_num'])[ValueName].count().reset_index()
    output_count.rename(columns = {ValueName:'number'}, inplace = True)
    output = pd.merge(output_mean, output_count, on=['Year','ten_num'])

    m = []
    for j in range(len(output)):
        m.append((output.ten_num.iloc[j]//3) + 1)
    output['Month'] = m

    TenDate = []
    for k in range(len(output)):
        try:
            TenDate.append(datetime.date(int(output.Year.iloc[k]),int(output.Month.iloc[k]),int((output.ten_num.iloc[k]-(3*(output.Month.iloc[k]-1)))*10+5)))
        except:
            print(output.Year.iloc[k])
            print(output.Month.iloc[k])
            TenDate.append(datetime.date(int(output.Year.iloc[k]),int(output.Month.iloc[k]),(output.ten_num.iloc[k]-(3*(output.Month.iloc[k]-1)))*10+5))
    output['TenDate'] = TenDate

    output[ValueName][(output.number <= 10)] = np.nan

    return output


# function to detrend the monthly CO2 data
# input
# -- Month_means_CO2: GEoDataFrame containing a "Date" colume with a datetime date, a "CO2" column
# -- year_min, month_min: start month and year of the time series -> needs to be bigger than 04/2009
# -- ten_min: start numbre of 10 days period in range (0,36)
# -- detrendtype: 0= trend with yearly growth rate, 1= 2009-2013 trend, 2=2009-2013 trend
# -- mean_type: monthly means (0) or 10-day means (1)
# -- ValueName: CO2 or CT_CO2
#
# output:
# -- detrend: detrended CO2 values
# -- Monthdate: Datetime with year and month and fixed day (15th)


def DetrendMonthdate(Month_means_CO2,year_min, month_min, ten_min, detrendtype, mean_type, ValueName, Dataset = ''):
    # create background dataset
    rate = [1.58,2.41,1.69,2.41,2.43,2.05,2.94,2.83,2.16,2.33,2.57,2.35,2.60]#updated on 02.06.2022
    #rate = [1.58,2.41,1.69,2.41,2.43,2.05,2.94,2.83,2.14,2.39,2.54]#updated on 18.02.2022
    #rate = [1.61,2.43,1.7,2.39,2.41,2.03,2.97,2.86,2.14,2.38,2.60]#old before 18.02.2022, https://gml.noaa.gov/ccgg/trends/gl_gr.html
    detrend_all_month = []
    detrend_year = []
    detrend_month = []
    offset = 384.5#384.5
    print('Detrended with same offset')
    '''
    print('Detrended with different offset')
    if 'TM5nocs' in Dataset:
        offset = 384.8
    elif 'OCO' in Dataset:
        offset = 384.8
    elif 'RTnew' in Dataset:
        offset = 385#384.5#385
    elif 'CAMSnocs' in Dataset:
        offset = 384.8
    elif 'CTm' in Dataset:
        offset = 384.8
    #elif 'CAMScs' in Dataset:
    #    offset = 385.1
    elif 'TCCONdb' in Dataset:
        offset = 385.2
    elif 'TCCONwg' in Dataset:
        offset = 384
    '''  
    for y in range(2009,2022):
        if y == 2009:
            for m in range(4,13):
                detrend_all_month.append(offset+rate[y-2009]/12*(m-4))
                detrend_month.append(m)
                detrend_year.append(y)
        else:
            for m in range(1,13):
                detrend_all_month.append(detrend_all_month[-1]+rate[y-2009]/12)
                detrend_month.append(m)
                detrend_year.append(y)

    if 'CH4' in ValueName:
        rate = [0.00467,0.00521,0.00485,0.00502,0.00569,0.01276,0.01003,0.00708,0.00687,0.00854,0.01034]
        detrend_all_month = []
        detrend_year = []
        detrend_month = []
        for y in range(2009,2020):
            if y == 2009:
                for m in range(4,13):
                    detrend_all_month.append(1.735+rate[y-2009]/12*(m-4))
                    detrend_month.append(m)
                    detrend_year.append(y)
            else:
                for m in range(1,13):
                    detrend_all_month.append(detrend_all_month[-1]+rate[y-2009]/12)
                    detrend_month.append(m)
                    detrend_year.append(y)
    '''                
    if 'OCO' in Dataset:
        rate = [1.61,2.43,1.7,2.39,2.41,2.03,2.97,2.86,2.14,2.38,2.60]
        #rate = [1.61,2.43,1.7,2.39,2.41,1.88,2.82,2.71,1.99,2.23,2.45]
        #rate = [1.61,2.43,1.7,2.39,2.41,1.93,2.87,2.76,2.04,2.28,2.50] #CAMS detrend
        detrend_all_month = []
        detrend_year = []
        detrend_month = []
        for y in range(2009,2020):
            if y == 2009:
                for m in range(4,13):
                    #detrend_all_month.append(384.5+rate[y-2009]/12*(m-4))
                    detrend_all_month.append(385.5+rate[y-2009]/12*(m-4))# CAMS
                    detrend_month.append(m)
                    detrend_year.append(y)
            else:
                for m in range(1,13):
                    detrend_all_month.append(detrend_all_month[-1]+rate[y-2009]/12)
                    detrend_month.append(m)
                    detrend_year.append(y)
    '''                    
    detrend_all_ten = []
    detrend_year_tens = []
    detrend_tens = []
    for y in range(2009,2022):
        if y == 2009:
            for m in range(10,36):
                detrend_all_ten.append(385+rate[y-2009]/36*(m-10))
                detrend_tens.append(m)
                detrend_year_tens.append(y)
        else:
            for m in range(0,36):
                detrend_all_ten.append(detrend_all_ten[-1]+rate[y-2009]/36)
                detrend_tens.append(m)
                detrend_year_tens.append(y)

    data_month = {'Background' : detrend_all_month,'Year':detrend_year,'Month': detrend_month}
    BG = pd.DataFrame(data = data_month)
    data_tens = {'Background' : detrend_all_ten,'Year':detrend_year_tens,'ten_num': detrend_tens}   
    BG_tens = pd.DataFrame(data = data_tens)

    print('got detrend background')
    if detrendtype == 0:
        if mean_type == 0:
            Month_means_CO2 = pd.merge(Month_means_CO2, BG, on = ['Year','Month'])
            #detrend_all = detrend_all_month
            #offset = ((year_min-2009)*12+month_min-4)
        elif mean_type == 1:
            Month_means_CO2 = pd.merge(Month_means_CO2, BG_tens, on = ['Year','ten_num'])
            #detrend_all = detrend_all_ten
            #offset = ((year_min-2009)*36+ten_min-10)
    elif detrendtype == 1:
        Month_means_CO2['Background']=(385+(Month_means_CO2.Month-4+12*(Month_means_CO2.Year-2009))*22/108)
    elif detrendtype == 2:
        Month_means_CO2['Background'] = (385+(Month_means_CO2.Month-4+12*(Month_means_CO2.Year-2009))*11/60)
    elif detrendtype == 3:
        print('get linear background')
        Month_means_CO2['Background'] = (384.5+(Month_means_CO2.Month-4+12*(Month_means_CO2.Year-2009))*0.194)
    else:
        sys.exit('please enter a detrend type')
    Month_means_CO2 = Month_means_CO2.reset_index()
    Month_means_CO2['Detrend'] = Month_means_CO2[ValueName] - Month_means_CO2['Background']
    
    MonthDate = []
    for m in range(len(Month_means_CO2[ValueName])):
        try:
            MonthDate.append(datetime.date(int(Month_means_CO2.Year.values[m]),int(Month_means_CO2.Month.values[m]),15))
        except:
            print(Month_means_CO2.Year.values[m])
            print(Month_means_CO2.Month.values[m])
            MonthDate.append(datetime.date(int(Month_means_CO2.Year.values[m]),(Month_means_CO2.Month.values[m]),15))
    Month_means_CO2['MonthDate'] = MonthDate
    return Month_means_CO2

def IndividualDetrend(Month_means_CO2,year_min, month_min, ten_min, detrendtype, mean_type, ValueName, Dataset = ''):
    # create background dataset
    rate = ((Month_means_CO2[(Month_means_CO2.Year == 2017)|(Month_means_CO2.Year == 2018)][ValueName].mean())-(Month_means_CO2[(Month_means_CO2.Year == 2010)|(Month_means_CO2.Year == 2011)][ValueName].mean()))/(7)
    detrend_all_month = []
    detrend_year = []
    detrend_month = []
    for y in range(2009,2020):
        if y == 2009:
            for m in range(4,13):
                #detrend_all_month.append(rate/12*(m-4))
                detrend_all_month.append(Month_means_CO2[(Month_means_CO2.Month == 4)&(Month_means_CO2.Year == 2009)][ValueName].values[0]+rate/12*(m-4))
                detrend_month.append(m)
                detrend_year.append(y)
        else:
            for m in range(1,13):
                detrend_all_month.append(detrend_all_month[-1]+rate/12)
                detrend_month.append(m)
                detrend_year.append(y)

    data_month = {'Background2' : detrend_all_month,'Year':detrend_year,'Month': detrend_month}
    BG = pd.DataFrame(data = data_month)

    Month_means_CO2 = pd.merge(Month_means_CO2, BG, on = ['Year','Month'])
    #print(Month_means_CO2.keys())
    try:
        Month_means_CO2['IndDetrend'] = Month_means_CO2[ValueName] - Month_means_CO2['Background2']
    except:
        Month_means_CO2['IndDetrend'] = Month_means_CO2[ValueName] - Month_means_CO2['Background2_x']
    #print(Month_means_CO2['IndDetrend'].values)
    ValueName = 'CO2'
    rate2 = ((Month_means_CO2[(Month_means_CO2.Year == 2019)&(Month_means_CO2.Month == 6)][ValueName].values[0])-(Month_means_CO2[(Month_means_CO2.Year == 2009)&(Month_means_CO2.Month == 4)][ValueName].values[0]))/(len(Month_means_CO2.Year)/12)
    detrend_all_month = []
    detrend_year = []
    detrend_month = []
    for y in range(2009,2020):
        if y == 2009:
            for m in range(4,13):
                #detrend_all_month.append(rate/12*(m-4))
                detrend_all_month.append(Month_means_CO2[(Month_means_CO2.Month == 4)&(Month_means_CO2.Year == 2009)][ValueName].values[0]-0.5+rate2/12*(m-4))
                detrend_month.append(m)
                detrend_year.append(y)
        else:
            for m in range(1,13):
                detrend_all_month.append(detrend_all_month[-1]+rate2/12)
                detrend_month.append(m)
                detrend_year.append(y)

    data_month2 = {'Background3' : detrend_all_month,'Year':detrend_year,'Month': detrend_month}
    BG2 = pd.DataFrame(data = data_month2)

    Month_means_CO2 = pd.merge(Month_means_CO2, BG2, on = ['Year','Month'])
    try:
        Month_means_CO2['IndDetrend2'] = Month_means_CO2[ValueName] - Month_means_CO2['Background3']
    except:
        Month_means_CO2['IndDetrend2'] = Month_means_CO2[ValueName] - Month_means_CO2['Background3_x']

    if detrendtype == 'finn_fwd':
        ValueName = 'Detrend'
        rate3 = 0.3
        detrend_all_month = []
        detrend_year = []
        detrend_month = []
        for y in range(2009,2020):
            if y == 2009:
                for m in range(4,13):
                    #detrend_all_month.append(rate/12*(m-4))
                    detrend_all_month.append(0)
                    detrend_month.append(m)
                    detrend_year.append(y)
            elif y <= 2013:
                for m in range(1,13):
                    detrend_all_month.append(0)
                    detrend_month.append(m)
                    detrend_year.append(y)
            else:
                for m in range(1,13):
                    detrend_all_month.append(detrend_all_month[-1]+rate3/12)
                    detrend_month.append(m)
                    detrend_year.append(y)
        data_month3 = {'Background4' : detrend_all_month,'Year':detrend_year,'Month': detrend_month}
        BG3 = pd.DataFrame(data = data_month3)

        Month_means_CO2 = pd.merge(Month_means_CO2, BG3, on = ['Year','Month'])
        Month_means_CO2['IndDetrend3'] = Month_means_CO2[ValueName] + Month_means_CO2['Background4']

    return Month_means_CO2


# function to add "MonthDate" column to Dtaframe based on column Year and Month

def addMonthDate(DF):
    MonthDate = []
    for m in range(len(DF['Year'])):
        try:
            MonthDate.append(datetime.date(int(DF.Year[m]),int(DF.Month[m]),15))
        except:
            print(DF.Year[m])
            print(DF.Month[m])
            MonthDate.append(datetime.date(int(DF.Year[m]),(DF.Month[m]),15))
    DF.insert(loc = 1,column='MonthDate',value = MonthDate)
    return DF


# function to create a pandas dataframe out of a NBE file
# input:
# -- filepath: file of NBE data
#
# output:
# -- df: pandas DataFrame containing the variables
# -- -- 'flux',
# -- -- 'Month' integer
# -- -- 'Date' datetime object with day = 15
# -- -- 'Year' integer

def createDataFrameNBE(filepath):
    DS = xr.open_mfdataset(filepath,combine='by_coords',concat_dim='None',decode_times=False)

    DayMonthSJ = [31,29,31,30,31,30,31,31,30,31,30,31]
    DayMonth = [31,28,31,30,31,30,31,31,30,31,30,31]

    for d in range(len(DS['flux'])):
        #dfy = pd.DataFrame(data=DS.variables['flux'][d].values,index = DS.variables['latitude'].values,
        #                          columns = DS.variables['longitude'].values, dtype='float' )
        # move coordinates to center coordinates
        dfy = pd.DataFrame(data=DS.variables['flux'][d].values,index = [-90] + list(range(-88,90,4)),
                                  columns = list(np.array(range(-1775,1825,50))/10), dtype='float' )
        dfy = pd.melt(dfy.reset_index(), id_vars='index',
                                  value_name ='flux',var_name = 'Long')
        dfy["Lat"] = dfy['index']
        
        #if d == 0:
        dfy2 = pd.DataFrame(data=DS.variables['area'].values,index = [-90] + list(range(-88,90,4)),
                                  columns =  list(np.array(range(-1775,1825,50))/10), dtype='float' )
        dfy2 = pd.melt(dfy2.reset_index(), id_vars='index',
                                 value_name ='area',var_name = 'Long')
        dfy2["Lat"] = dfy2['index']
        dfy3 = pd.DataFrame(data=DS.variables['LandMask'].values,index = [-90] + list(range(-88,90,4)),
                                  columns =  list(np.array(range(-1775,1825,50))/10), dtype='float' )
        dfy3 = pd.melt(dfy3.reset_index(), id_vars='index',
                                 value_name ='LandMask',var_name = 'Long')
        dfy3["Lat"] = dfy3['index']
        #print(dfy2)
        dfy = pd.merge(dfy, dfy2, on=['Lat','Long'], how = 'outer')
        dfy = pd.merge(dfy, dfy3, on=['Lat','Long'], how = 'outer')
        dfy.insert(loc=1, column='total_flux_sec', value = dfy['area']*dfy.flux*dfy.LandMask)
        #print(dfy)

        date = []
        year = []
        month = []
        tflux = []
        monthdate = DS.variables['nmon'].values #monthnumber since 2010-01-01 start with number 1
        for i in range(len(dfy['flux'])):
            year.append(((monthdate[d]-1)//12)+2010)
            month.append(monthdate[d]-(12*(year[-1]-2010)))
            date.append(datetime.date(int(year[-1]),int(month[-1]),15))
            if year[-1] in [2012,2016,2020]:
                tflux.append(dfy.total_flux_sec[i]*60*60*24*DayMonthSJ[int(month[-1])-1])
            else:
                tflux.append(dfy.total_flux_sec[i]*60*60*24*DayMonth[int(month[-1])-1])
        dfy.insert(loc=1,column='Date',value= date)
        dfy.insert(loc=1,column='Year',value= year)
        dfy.insert(loc=1,column='Month',value= month)
        dfy.insert(loc=1, column='total_flux', value = tflux)

        if d == 0:
            df = dfy.copy()
        else:
            df = df.append(dfy)
        print(d)

    return df



# function to create a pandas dataframe out of a GFED4 file
# input:
# -- filepath: file of GFED4 data
# -- datayear: year of the datafile
# -- datatype: emissions per m^2 (=0) or total emissions (=1)
#
# output:
# -- df: pandas DataFrame containing the variables
# -- -- 'emission',
# -- -- 'Month' integer
# -- -- 'Date' datetime object with day = 15
# -- -- 'Year' integer



def createDataFrameGFED(filepath,datayear,datatype):
    DS = h5py.File(filepath,'r')
    typeL = ['SAVA','BORF','TEMF','DEFO','PEAT','AGRI']
    CO2L = [1686,1489,1647,1643,1703,1585]
    COL = [63,127,88,93,210,102]
    for m in range(1,13):
        mo = str(m)
        if datatype == 0:
            dfy = pd.DataFrame(data=DS['emissions/'+mo.zfill(2)+'/C'][:],index = DS['lat'][:,0],
                                  columns = DS['lon'][0,:], dtype='float' )
            if datayear < 2017:
                dfy_area = pd.DataFrame(data=DS['burned_area/'+mo.zfill(2)+'/burned_fraction'][:],index = DS['lat'][:,0],
                                  columns = DS['lon'][0,:], dtype='float' )
            else:
                arr = np.empty([720,1440])
                arr[:] = np.nan
                dfy_area = pd.DataFrame(data=arr,index = DS['lat'][:,0],
                                  columns = DS['lon'][0,:], dtype='float' )
            dfy_area = pd.melt(dfy_area.reset_index(), id_vars='index',
                                  value_name ='burned_fraction',var_name = 'Long')
            dfy_area["Lat"] = dfy_area['index']
        elif datatype == 1:
            dfy = pd.DataFrame(data=DS['biosphere/'+mo.zfill(2)+'/NPP'][:],index = DS['lat'][:,0],
                                  columns = DS['lon'][0,:], dtype='float' )
        
        elif datatype == 2:
            datae = DS['emissions/'+mo.zfill(2)+'/DM'][:]
            CO2emission = np.zeros((720, 1440))
            for t in range(6):
                contribution = DS['emissions/'+mo.zfill(2)+'/partitioning/DM_'+typeL[t]][:]
                CO2emission += datae * contribution * CO2L[t]

            dfy = pd.DataFrame(data=CO2emission,index = DS['lat'][:,0],
                                  columns = DS['lon'][0,:], dtype='float' )
       
        elif datatype == 3:
            datae = DS['emissions/'+mo.zfill(2)+'/DM'][:]
            CO2emission = np.zeros((720, 1440))
            for t in range(6):
                contribution = DS['emissions/'+mo.zfill(2)+'/partitioning/DM_'+typeL[t]][:]
                CO2emission += datae * contribution * COL[t]

            dfy = pd.DataFrame(data=CO2emission,index = DS['lat'][:,0],
                                  columns = DS['lon'][0,:], dtype='float' )
  
        dfy = pd.melt(dfy.reset_index(), id_vars='index',
                                  value_name ='emission',var_name = 'Long')
        dfy["Lat"] = dfy['index']

        if datatype == 0:
            dfy = pd.merge(dfy, dfy_area, on=['Lat','Long'], how = 'outer')

        date = []
        year = []
        month = []
        for i in range(len(dfy['emission'])):
            year.append(datayear)
            month.append(m)
            date.append(datetime.date(int(year[-1]),int(month[-1]),15))
        dfy.insert(loc=1,column='Date',value= date)
        dfy.insert(loc=1,column='Year',value= year)
        dfy.insert(loc=1,column='Month',value= month)
        if m ==1:
            df = dfy.copy()
        else:
            df = df.append(dfy)
        
    return df

def getMeanAmplitude(df,varName, yearName, minYear, maxYear):
    '''
    #calculates the mean amplitude and its standard deviation of a time series 
    # with customizable year coordinate
    # input
    #   dataframe containing:
    #     value variable
    #     year coordinate
    #   name of value coordinate
    #   name of year coordinate
    #   Year to start the averaging 
    #   Year to end the averaging
    # returns:
    #   mean Amplitude, stDev, minimum and maximum as integers'''
    dfAmpli = df.groupby([yearName])[varName].min().reset_index()
    dfAmpli.rename(columns = {varName:'AMin'},inplace=True)
    dfAmpli2 = df.groupby([yearName])[varName].max().reset_index()
    dfAmpli2.rename(columns = {varName:'AMax'},inplace=True)
    dfAmpli = dfAmpli.merge(dfAmpli2,on=yearName)
    del dfAmpli2
    dfAmpli.insert(loc = 1, column = 'AAmpli',value = dfAmpli.AMax-dfAmpli.AMin)
    amplitude = dfAmpli[(dfAmpli[yearName] >= minYear)&(dfAmpli[yearName] <= maxYear)]['AAmpli'].mean()
    minimum = dfAmpli[(dfAmpli[yearName] >= minYear)&(dfAmpli[yearName] <= maxYear)]['AMin'].mean()
    maximum = dfAmpli[(dfAmpli[yearName] >= minYear)&(dfAmpli[yearName] <= maxYear)]['AMax'].mean()
    StDevAmplitude = dfAmpli[(dfAmpli[yearName] >= minYear)&(dfAmpli[yearName] <= maxYear)]['AAmpli'].std(ddof = 0)
    
    return amplitude, StDevAmplitude, minimum, maximum

def getReferenceDate(year_min,year_max,month_min,month_max):
    yea = []
    mon = []
    for k in range(year_min,year_max +1):
        if k == year_min:
            for p in range(month_min,12+1):
                yea.append(k)
                mon.append(p)
        elif k == year_max:
            for p in range(1,month_max+1):
                yea.append(k)
                mon.append(p)
        else:
            for p in range(1,13):
                yea.append(k)
                mon.append(p)

    dateData = {"Year":yea,"Month":mon}
    DateRef = pd.DataFrame(data=dateData)
    MonthDate2 = []
    for j in range(len(DateRef.Year)):
        MonthDate2.append(datetime.date(DateRef.Year[j],DateRef.Month[j],15))
    DateRef['MonthDate'] = MonthDate2

    #year from july to june index column
    jjYear = []
    for i in range(len(DateRef.Year)):
        if DateRef.Month.values[i] <= 6:
            jjYear.append(DateRef.Year.values[i]-1)
        elif DateRef.Month.values[i] >= 7:
            jjYear.append(DateRef.Year.values[i])
        else:
            print('Error')
    DateRef.insert(loc = 1, column= 'JJYear',value = jjYear)

    return DateRef

def getReferenceDateDay(year_min,year_max,month_min,month_max):
    yea = []
    mon = []
    day = []
    for k in range(year_min,year_max +1):
        if k == year_min:
            for p in range(month_min,12+1):
                for d in range(1,getNumDayOfMonth(k,p)+1):
                    yea.append(k)
                    mon.append(p)
                    day.append(d)
        elif k == year_max:
            for p in range(1,month_max+1):
                for d in range(1,getNumDayOfMonth(k,p)+1):
                    yea.append(k)
                    mon.append(p)
                    day.append(d)
        else:
            for p in range(1,13):
                for d in range(1,getNumDayOfMonth(k,p)+1):
                    yea.append(k)
                    mon.append(p)
                    day.append(d)

    dateData = {"Year":yea,"Month":mon,"Day":day}
    DateRef = pd.DataFrame(data=dateData)
    MonthDate2 = []
    for j in range(len(DateRef.Year)):
        MonthDate2.append(datetime.date(DateRef.Year[j],DateRef.Month[j],DateRef.Day[j]))
    DateRef['Date'] = MonthDate2

    return DateRef


def getNumDayOfMonth(year,month):
    listDays = [31,28,31,30,31,30,31,31,30,31,30,31]
    listDaysl = [31,29,31,30,31,30,31,31,30,31,30,31]
    if year < 1900:
        print('year is out of implemented range, check code')
    elif year in list(range(1904,2100,4)):
        days = listDaysl[month-1]
    else:
        days = listDays[month-1]

    return days


def getReferencesDateDay(year_min,year_max,month_min,month_max,day_min,day_max):
    
    yea = []
    mon = []
    day = []
    for k in range(year_min,year_max +1):
        if k == year_min:
            for p in range(month_min,12+1):
                if p == month_min:
                    for d in range(day_min, getNumDayOfMonth(k,p)+1):
                        day.append(d)
                        yea.append(k)
                        mon.append(p)
                else:
                    for d in range(1, getNumDayOfMonth(k,p)+1):
                        day.append(d)
                        yea.append(k)
                        mon.append(p)
        elif k == year_max:
            for p in range(1,month_max+1):
                if p == month_max:
                    for d in range(1, day_max):
                        day.append(d)
                        yea.append(k)
                        mon.append(p)
                else:
                    for d in range(1, getNumDayOfMonth(k,p)+1):
                        day.append(d)
                        yea.append(k)
                        mon.append(p)
        else:
            for p in range(1,13):
                for d in range(1, getNumDayOfMonth(k,p)+1):
                    day.append(d)
                    yea.append(k)
                    mon.append(p)

    dateData = {"Year":yea,"Month":mon,"Day":day}
    DateRef = pd.DataFrame(data=dateData)
    MonthDate2 = []
    for j in range(len(DateRef.Year)):
        MonthDate2.append(datetime.date(DateRef.Year[j],DateRef.Month[j],DateRef.Day[j]))
    DateRef['MonthDate'] = MonthDate2

    return DateRef

def getAreaOfGrid():
    """Get Dataframe with Area dependend on Latitude for a 1°x1° grid"""
    AreaLat = []
    Lat = range(895,-905,-10)
    for i in Lat:
        geom = [Polygon(zip([100,100,101,101],[i/10-0.5,i/10+0.5,i/10+0.5,i/10-0.5]))]
        GData = geopandas.GeoDataFrame({'data':[1]}, geometry=geom)
        GData.crs = 'epsg:4326'
        GData = GData.to_crs({'proj':'cea'})
        AreaLat.append(GData.geometry.area[0])
    dfArea = pd.DataFrame({'Area':AreaLat,'Lat':np.array(Lat)/10})
    
    return(dfArea)
    
def getGdfOfGrid(Num):
    """Get Geodatframe of 1°x1° Grid for a certain transcom region"""
    Transcom = pd.read_pickle("/home/eschoema/Transcom_Regions.pkl")
    first = True
    AreaLat = []
    RegionName, Long_min, Long_max, Lat_min, Lat_max = getRegion(Num)
    Lat = range(Lat_max*10+5, Lat_min*10-5,-10)
    if Long_max >= 179.5:
        Long_max = Long_max -1
    Long = range(Long_max*10+5, Long_min*10-5,-10)
    for i in Lat:
        for j in Long:
            geom = [Polygon(zip([j/10-0.5,j/10-0.5,j/10+0.5,j/10+0.5],[i/10-0.5,i/10+0.5,i/10+0.5,i/10-0.5]))]
            GData = geopandas.GeoDataFrame({'data':[1]}, geometry=geom)
            GData.crs = 'epsg:4326'
            GData.insert(loc=1,column = 'Lat',value = [i/10])
            GData.insert(loc=1,column = 'Long',value = [j/10])
            if first:
                gdf = GData.copy()
                first = False
            else:
                gdf = gdf.append(GData, ignore_index=True)
            GData = GData.to_crs({'proj':'cea'})
            AreaLat.append(GData.geometry.area[0])
    
    gdf.insert(loc=1, value = AreaLat,column = 'AreaGrid') 
    gdf = gdf.drop(columns = 'data')
    gdf.insert(loc=1, value = gdf.geometry,column = 'geomPoly') 
    gdf.insert(loc=1, column = 'geomPoint', value = geopandas.points_from_xy(gdf.Long, gdf.Lat))
    gdf = gdf.set_geometry(gdf.geomPoint)
    igdf = gdf.within(Transcom[(Transcom.transcom == RegionName)].geometry.iloc[0])
    gdf = gdf.loc[igdf]
    
    gdf = gdf.set_geometry(gdf.geomPoly)
    gdf = gdf.reset_index()
    gdf.insert(loc=1,value=gdf.index,column='GridID')
    gdf.to_pickle("/home/eschoema/Grid1_1"+RegionName+".pkl")
    
    return gdf

def getGdfOfRectGrid(Lat_min,Lat_max,Long_min,Long_max):
    """Get Geodatframe of 1°x1° Grid for a certain rectengular region"""
    first = True
    AreaLat = []
    Lat = range(Lat_max*10+5, Lat_min*10-5,-10)
    if Long_max >= 179.5: #avoid borders
        Long_max = Long_max -1
    Long = range(Long_max*10+5, Long_min*10-5,-10)
    for i in Lat:
        for j in Long:
            geom = [Polygon(zip([j/10-0.5,j/10-0.5,j/10+0.5,j/10+0.5],[i/10-0.5,i/10+0.5,i/10+0.5,i/10-0.5]))]
            GData = geopandas.GeoDataFrame({'data':[1]}, geometry=geom)
            GData.crs = 'epsg:4326' #set lat long coordinate reference system
            GData.insert(loc=1,column = 'Lat',value = [i/10])
            GData.insert(loc=1,column = 'Long',value = [j/10])
            if first:
                gdf = GData.copy()
                first = False
            else:
                gdf = gdf.append(GData, ignore_index=True)
            GData = GData.to_crs({'proj':'cea'}) #project from lat/lon to meter based coordinate system
            AreaLat.append(GData.geometry.area[0]) #get area of indiviual grid cells
    
    gdf.insert(loc=1, value = AreaLat,column = 'AreaGrid') 
    gdf = gdf.drop(columns = 'data')
    gdf.insert(loc=1, value = gdf.geometry,column = 'geomPoly') #save polygon geometry as column
    gdf.insert(loc=1, column = 'geomPoint', value = geopandas.points_from_xy(gdf.Long, gdf.Lat)) #create point geometry
    gdf = gdf.set_geometry(gdf.geomPoint) # set point geometry as active geometry
    gdf = gdf.reset_index()
    gdf.insert(loc=1,value=gdf.index,column='GridID')
    gdf.to_pickle("/home/eschoema/Grid1_1"+'_'+str(Lat_min)+'_'+str(Lat_max)+'_'+str(Long_min)+'_'+str(Long_max)+".pkl")
    
    return gdf
    