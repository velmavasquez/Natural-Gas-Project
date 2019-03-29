import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gmaps
import gmaps.geojson_geometries

from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss

from datetime import datetime
from dateutil.relativedelta import relativedelta
from calendar import monthrange

from matplotlib.pyplot import cm
from pprint import pprint

from matplotlib.cm import viridis
from matplotlib.cm import pink
from matplotlib.colors import to_hex

from census import Census

def f_strToDate(ini_date):
    
    strg = ini_date
    ch = "-"
    vpos = [pos for pos, char in enumerate(strg) if char == ch]
    year = int(strg[:vpos[0]])
    month = int(strg[vpos[0]+1:vpos[1]])
    day = int(strg[vpos[1]+1:])
    
    return datetime(year,month,day).date()

def f_EIA_ExtractData(url,eia_api_key,series_id):
    query = f"{url}?api_key={eia_api_key}&series_id={series_id}"
    response = requests.get(query).json()
    #pprint(response)

    data = response["series"][0]["data"]
    valx = []
    valy = []
    df  = pd.DataFrame(response["series"][0]["data"])
    df = df.dropna(how='any')
    #print(df)
    
    for item, row in df.iterrows():
        vx = row.iloc[0]
        vy = row.iloc[1]
        if series_id[-1]=="M":
            #date_str = vx[4:]+"/15/"+vx[:4]
            date_str = vx[:4]+"/"+vx[4:]+"/1"            
        elif series_id[-1]=="A":
            #date_str = "6/15/"+vx[:4]
            date_str = vx+"/6/1"
        elif series_id[-1]=="D":
            date_str = vx[:4]+"/"+vx[4:6]+"/"+vx[6:]
        
        #date_object = datetime.strptime(date_str,'%m/%d/%Y').date()
        date_object = datetime.strptime(date_str,'%Y/%m/%d').date()
        #print(f"{(date_object)} , {float(x[1])}")
        valx.append(date_object)
        valy.append(float(vy))

    title = response["series"][0]["name"]
    try:
        country = response["series"][0]["geography"]
    except:
        country ="US"
    #if series_id[-1]=="D":
    #    country ="US"
    #else:
    #    country = response["series"][0]["geography"]
    unit = response["series"][0]["units"]
    
    return valx, valy, title, country, unit

def f_EIA_ExtractData_Range(url,eia_api_key,series_id,ini_date,end_date):

    ini_date_d = f_strToDate(ini_date)
    end_date_d = f_strToDate(end_date)
    
    valx, valy, title, country, unit = f_EIA_ExtractData(url,eia_api_key,series_id)

    valx_upt = [];     valy_upt = []
    index = 0
    for x in valx:    
        if  (x>=ini_date_d) & (x<=end_date_d):
            valx_upt.append(x)
            valy_upt.append(valy[index])
        index +=1

    return valx_upt, valy_upt, title, country, unit


def f_EIA_ListSeries(url,eia_api_key,category_id):
    query = f"{url}?api_key={eia_api_key}&category_id={category_id}&out=json"
    data = requests.get(query).json()
    list_chcat = data["category"]["childcategories"]
    list_chser = data["category"]["childseries"]    
    
    list_ser = []    
    if (len(list_chcat) > 0) & (len(list_chser) == 0):
        for x in list_chcat:
            category_id_x = x['category_id']            
            list_ser.extend(f_EIA_ListSeries(url,eia_api_key,category_id_x))           
    elif (len(list_chcat) == 0) & (len(list_chser) > 0):                
        list_ser = list_chser
            
    return list_ser


def f_EIA_PlotData(list_series,url,eia_api_key,title):    
    large = 16; med = 12.5; small = 11
    params = {'axes.titlesize': med,
              'legend.fontsize': small,
              'figure.figsize': (15, 5),
              'axes.labelsize': med,
              'axes.titlesize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    fig, ax = plt.subplots()

    ax.yaxis.set_label_coords(0.00,1.03)
    ax.yaxis.grid(True)

    color=iter(cm.rainbow(np.linspace(0,1,len(list_series))))
    flag = True

    for series_id in list_series:
        valx, valy, label, country, unit = f_EIA_ExtractData(url,eia_api_key,series_id)    
        fig.suptitle(title, fontweight="bold")
        ax.set_ylabel(unit, rotation=0, labelpad=20)

        if flag:
            maxx = max(valx); minx = min(valx); maxy = max(valy); miny = min(valy);
            flag = False
        else:        
            if maxx < max(valx):
                maxx = max(valx)
            if minx > min(valx):
                minx = min(valx)
            if maxy < max(valy):
                maxy = max(valy)
            if miny > min(valy):
                miny = min(valy)

        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy + 2)
        c=next(color)
        ax.plot(valx,valy, linewidth=2,color=c,label=label)
        ax.legend(loc='upper left')
    
    plt.show()

def f_EIA_PlotData_Range(list_series,url,eia_api_key,title,ini_date,end_date):    
    large = 16; med = 12.5; small = 11
    params = {'axes.titlesize': med,
              'legend.fontsize': small,
              'figure.figsize': (15, 5),
              'axes.labelsize': med,
              'axes.titlesize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    fig, ax = plt.subplots()

    ax.yaxis.set_label_coords(0.00,1.03)
    ax.yaxis.grid(True)

    color=iter(cm.rainbow(np.linspace(0,1,len(list_series))))
    flag = True

    for series_id in list_series:
        valx, valy, label, country, unit = f_EIA_ExtractData_Range(url,eia_api_key,series_id,ini_date,end_date)
        fig.suptitle(title, fontweight="bold")
        ax.set_ylabel(unit, rotation=0, labelpad=20)

        if flag:
            maxx = max(valx); minx = min(valx); maxy = max(valy); miny = min(valy);
            flag = False
        else:        
            if maxx < max(valx):
                maxx = max(valx)
            if minx > min(valx):
                minx = min(valx)
            if maxy < max(valy):
                maxy = max(valy)
            if miny > min(valy):
                miny = min(valy)

        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy + 2)
        c=next(color)
        ax.plot(valx,valy, linewidth=2,color=c,label=label)
        ax.legend(loc='upper left')
    
    plt.show()


def f_EIA_PlotData_Stack_Range(list_series,url,eia_api_key,title,ini_date,end_date):    
    large = 16; med = 12.5; small = 11
    params = {'axes.titlesize': med,
              'legend.fontsize': small,
              'figure.figsize': (15, 5),
              'axes.labelsize': med,
              'axes.titlesize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    fig, ax = plt.subplots()

    ax.yaxis.set_label_coords(0.00,1.03)
    ax.yaxis.grid(True)

    color=iter(cm.rainbow(np.linspace(0,1,len(list_series))))
    flag = True

    for series_id in list_series:
        valx, valy, label, country, unit = f_EIA_ExtractData_Range(url,eia_api_key,series_id,ini_date,end_date)
        fig.suptitle(title, fontweight="bold")
        ax.set_ylabel(unit, rotation=0, labelpad=20)

        if flag:
            maxx = max(valx); minx = min(valx); maxy = max(valy); miny = min(valy)
            flag = False
            y = valy
            vlabel = [label]
        else:        
            if maxx < max(valx):
                maxx = max(valx)
            if minx > min(valx):
                minx = min(valx)
            if maxy < max(valy):
                maxy = max(valy)
            if miny > min(valy):
                miny = min(valy)
            y = np.vstack([y,valy])
            vlabel.append(label)

        #ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy + 2)
        c=next(color)
        #ax.plot(valx,valy, linewidth=2,color=c,label=label)
        
    print(vlabel)   
    ax.stackplot(valx,y,labels=vlabel)
    #ax.legend(loc='upper left')
    
    plt.show()

def f_EIA_PlotData_Mod_Range(list_series,list_series_sec,url,eia_api_key,title,ini_date,end_date):
    large = 16; med = 12.5; small = 11
    params = {'axes.titlesize': med,
              'legend.fontsize': small,
              'figure.figsize': (15, 5),
              'axes.labelsize': med,
              'axes.titlesize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    fig, ax = plt.subplots()
    
    ax.yaxis.set_label_coords(0.00,1.03)
    ax.yaxis.grid(True)

    color=iter(cm.rainbow(np.linspace(0,1,len(list_series)+len(list_series_sec))))
    flag = True

    for series_id in list_series:
        valx, valy, label, country, unit = f_EIA_ExtractData_Range(url,eia_api_key,series_id,ini_date,end_date)    
        fig.suptitle(title, fontweight="bold")
        ax.set_ylabel(unit, rotation=0, labelpad=20)

        if flag:
            maxx = max(valx); minx = min(valx); maxy = max(valy); miny = min(valy)
            flag = False
        else:
            if maxx < max(valx):
                maxx = max(valx)
            if minx > min(valx):
                minx = min(valx)
            if maxy < max(valy):
                maxy = max(valy)
            if miny > min(valy):
                miny = min(valy)

        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy + 2)
        c=next(color)
        ax.plot(valx,valy, linewidth=2,color=c,label=label)
        ax.legend(loc='upper left')
    
    ax_sec = ax.twinx()
    
    for series_id in list_series_sec:
        valx, valy, label, country, unit = f_EIA_ExtractData_Range(url,eia_api_key,series_id,ini_date,end_date)    
        #fig.suptitle(title, fontweight="bold")
        #ax.set_ylabel(unit, rotation=0, labelpad=20)

        if maxx < max(valx):
            maxx = max(valx)
        if minx > min(valx):
            minx = min(valx)
        if maxy < max(valy):
            maxy = max(valy)
        if miny > min(valy):
            miny = min(valy)

        #ax_sec.set_xlim(minx, maxx); ax_sec.set_ylim(miny, maxy + 2)
        c=next(color)
        ax_sec.plot(valx,valy, linewidth=2,color=c,label=label)
        ax_sec.legend(loc='center left') 
    
    #plt.show()

def f_EIA_PlotData_Mod(list_series,list_series_sec,url,eia_api_key,title):    
    large = 16; med = 12.5; small = 11
    params = {'axes.titlesize': med,
              'legend.fontsize': small,
              'figure.figsize': (15, 5),
              'axes.labelsize': med,
              'axes.titlesize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    fig, ax = plt.subplots()
    
    ax.yaxis.set_label_coords(0.00,1.03)
    ax.yaxis.grid(True)

    color=iter(cm.rainbow(np.linspace(0,1,len(list_series)+len(list_series_sec))))
    flag = True

    for series_id in list_series:
        valx, valy, label, country, unit = f_EIA_ExtractData(url,eia_api_key,series_id)    
        fig.suptitle(title, fontweight="bold")
        ax.set_ylabel(unit, rotation=0, labelpad=20)

        if flag:
            maxx = max(valx); minx = min(valx); maxy = max(valy); miny = min(valy)
            flag = False
        else:        
            if maxx < max(valx):
                maxx = max(valx)
            if minx > min(valx):
                minx = min(valx)
            if maxy < max(valy):
                maxy = max(valy)
            if miny > min(valy):
                miny = min(valy)

        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy + 2)
        c=next(color)
        ax.plot(valx,valy, linewidth=2,color=c,label=label)
        ax.legend(loc='upper left')
    
    ax_sec = ax.twinx()
    
    for series_id in list_series_sec:
        valx, valy, label, country, unit = f_EIA_ExtractData(url,eia_api_key,series_id)    
        #fig.suptitle(title, fontweight="bold")
        #ax.set_ylabel(unit, rotation=0, labelpad=20)

        if maxx < max(valx):
            maxx = max(valx)
        if minx > min(valx):
            minx = min(valx)
        if maxy < max(valy):
            maxy = max(valy)
        if miny > min(valy):
            miny = min(valy)

        #ax_sec.set_xlim(minx, maxx); ax_sec.set_ylim(miny, maxy + 2)
        c=next(color)
        ax_sec.plot(valx,valy, linewidth=2,color=c,label=label)
        ax_sec.legend(loc='center left') 
    
    plt.show()


def f_WorldWeather_ExtractData(url,api_key,ini_date,end_date,city,out_format,NumHours,unit):
    
#url = "https://api.worldweatheronline.com/premium/v1/past-weather.ashx"
#api_key = "5f0480e908404ffab6e231155191803"
#ini_date = "2009-1-1"
#end_date = "2019-2-1"
#city = "New York, NY"
#out_format = "json"
#NumHours = 24

    params = {
        "q": city,
        "key": api_key,
        "format": out_format,
        "tp": NumHours
    }

    d1 = datetime.strptime(ini_date,'%Y-%m-%d').date()
    d2 = datetime.strptime(end_date,'%Y-%m-%d').date()
    NumMonths = diff_month(d2, d1)

    vpos = [pos for pos, char in enumerate(ini_date) if char == "-"]
    year = int(ini_date[:vpos[0]])
    month = int(ini_date[vpos[0]+1:vpos[1]])
    day = int(ini_date[vpos[1]+1:])               
    endday_month = monthrange(year,month)[1]

    valx = []; valy = []

    for x in range(1,NumMonths+1,1):    
        if x == 1:
            ini_date_x = ini_date
            #end_date_x = datetime(year,month,endday_month)
            end_date_x = str(year) + "-" + str(month) + "-" + str(endday_month)
        else:         
            date_object = datetime.strptime(end_date_x,'%Y-%m-%d').date()
            #date_after_month = date_object + relativedelta(months=1)
            date_after_month = date_object + relativedelta(days=1)
            year = date_after_month.year
            month = date_after_month.month
            day = date_after_month.day
            ini_date_x = str(year) + "-" + str(month) + "-" + str(day)
            endday_month = monthrange(year,month)[1]
            if x == NumMonths:
                end_date_x = end_date
            else:
                end_date_x = str(year) + "-" + str(month) + "-" + str(endday_month)
        
        params["date"] = ini_date_x
        params["enddate"] = end_date_x
        response = requests.get(url, params=params)
        data = response.json()
        #pprint(data)
    
        list_Temp = data["data"]["weather"]    
        for i in range(0,len(list_Temp),1):
            #print(f' {list_Temp[i]["date"]} : {list_Temp[i]["maxtempC"]} C ')
            date_object = datetime.strptime(list_Temp[i]["date"],'%Y-%m-%d').date()
            valx.append(date_object)
            #valx.append(x[i]["date"])
            if unit == "F":
                valy.append(0.5*float(list_Temp[i]["maxtempF"])+0.5*float(list_Temp[i]["mintempF"]))
            elif unit == "C":
                valy.append(0.5*float(list_Temp[i]["maxtempC"])+0.5*float(list_Temp[i]["mintempC"]))
            else:
                print(f'{unit} is not supported !!!!!!')
                break

    return valx, valy

def f_Census_ExtractData(api_key):

    c = Census(api_key)

    # Run Census Search to retrieve data on all states
    # Note the addition of "B23025_005E" for unemployment count
    census_data = c.acs5.get(("NAME", "B19013_001E", "B01003_001E", "B01002_001E",
                              "B19301_001E",
                              "B17001_002E",
                              "B23025_005E"), {'for': 'county:*'})

    # Convert to DataFrame
    census_pd = pd.DataFrame(census_data)

    # Column Reordering
    census_pd = census_pd.rename(columns={"B01003_001E": "Population",
                                          "B01002_001E": "Median Age",
                                          "B19013_001E": "Household Income",
                                          "B19301_001E": "Per Capita Income",
                                          "B17001_002E": "Poverty Count",
                                          "B23025_005E": "Unemployment Count",
                                          "NAME": "Name", "state": "State"})

    # Add in Poverty Rate (Poverty Count / Population)
    census_pd["Poverty Rate"] = 100 * \
        census_pd["Poverty Count"].astype(
            int) / census_pd["Population"].astype(int)

    # Add in Employment Rate (Employment Count / Population)
    census_pd["Unemployment Rate"] = 100 * \
        census_pd["Unemployment Count"].astype(
            int) / census_pd["Population"].astype(int)

    #census_pd["State_Name"] = census_pd["Name"]

    for index,row in census_pd.iterrows():
        state_name = row["Name"]
        census_pd.loc[index,"State_Name"] = state_name[(state_name.find(",")+2):]
        #census_pd["State_Name"] = state_name[:(state_name.find(",")-1)]
        census_pd.loc[index,"Name"] = state_name[:(state_name.find(","))]
    
    
    census_pd["Name"] = census_pd["Name"].str.replace(" Municipio","")
    census_pd["Name"] = census_pd["Name"].str.replace(" County","")



    # Final DataFrame
    census_pd = census_pd[["State", "State_Name", "Name", "Population", "Median Age", "Household Income",
                           "Per Capita Income", "Poverty Count", "Poverty Rate", "Unemployment Rate"]]

    return census_pd

def f_PlotData(valx, valy, title, unit):
    large = 16; med = 14; small = 11
    params = {'axes.titlesize': med,
              'legend.fontsize': med,
              'figure.figsize': (15, 5),
              'axes.labelsize': med,
              'axes.titlesize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    fig, ax = plt.subplots()
    fig.suptitle(title, fontweight="bold")
    ax.set_xlim(min(valx), max(valx))
    ax.set_ylim(min(valy) - 5, max(valy) + 5)

    #ax.set_xlabel("Fake Banana Ages (in days)")
    ax.set_ylabel(unit, rotation=0, labelpad=20)
    ax.yaxis.set_label_coords(0.09,1.03)
    ax.yaxis.grid(True)

    ax.plot(valx,valy, linewidth=2,color="Slateblue",label=title)
    ax.legend(loc='upper left')
    #plt.show()

    return fig, ax

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def calculate_color(par,min_par,max_par):
    #"""
    #Convert the par to a color
    #"""
    # make par a number between 0 and 1
    range_par = max_par - min_par
    normalized_par = (par - min_par) / range_par
    # invert gini so that high inequality gives dark color
    inverse_par = 1.0 - normalized_par
    # transform the gini coefficient to a matplotlib color
    mpl_color = viridis(inverse_par)
    # transform from a matplotlib color to a valid CSS color
    gmaps_color = to_hex(mpl_color, keep_alpha=False)
    return gmaps_color


def adf(ts):
    
    # Determing rolling statistics    
    ts_t = pd.Series.rolling(ts, window=12)
    rolmean = ts_t.mean()
    rolstd = ts_t.std()

    #Plot rolling statistics:
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Calculate ADF factors
    adftest = adfuller(ts, autolag='AIC')
    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','# of Lags Used',
                                              'Number of Observations Used'])
    for key,value in adftest[4].items():
        adfoutput['Critical Value (%s)'%key] = value
    return adfoutput

def RMSE(predicted, actual):
    mse = (predicted - actual)**2
    rmse = np.sqrt(mse.sum()/mse.count())
    return rmse


def calculate_color_1(par,min_par,max_par):
    #"""
    #Convert the par to a color
    #"""
    # make par a number between 0 and 1
    range_par = max_par - min_par
    normalized_par = (par - min_par) / range_par
    # invert gini so that high inequality gives dark color
    inverse_par = 1.0 - normalized_par
    # transform the gini coefficient to a matplotlib color
    mpl_color = pink(inverse_par)
    # transform from a matplotlib color to a valid CSS color
    gmaps_color = to_hex(mpl_color, keep_alpha=False)
    return gmaps_color