import geopandas as gpd
import pandas as pd
import datetime as dt
import pyarrow
import os
from pathlib import Path
import numpy as np
from datawrapper import Datawrapper
from dotenv import load_dotenv


def inc_data_read(start_year = 2014, full_dataset = True, convert_cook_crs = True):
    
    """
    Pulling in City of Chicago Incident data; returns pandas dataframe of incidents.
    start_year: input starting year for requested data
    full_dataset: choose to pull in full dataset or small subset of data
    convert_cook_crs: choose to convert to local espg to match beat data or not 
    """

    

    if full_dataset == True:
        print("Pulling full dataset")
        limit = 20000000
    else:
        print("Small subset")
        limit = 200
    
    # initialize datatime object
    today = dt.date.today()
    
    # getting current year
    current_yr = today.year
    # initialize a list
    inc_df_list = []
    
    # for each year between 2018 and current year, pull in incident data    
    for year in range(start_year, current_yr + 1):
        inc_data_yr = pd.read_csv(
            f'https://data.cityofchicago.org/resource/ijzp-q8t2.csv?$limit={limit}&$where=date%20between%20%27{year}-01-01T00:00:00%27%20and%20%27{year}-12-31T23:59:59%27', storage_options={'verify': False}
        )
        inc_df_list.append(inc_data_yr)
        
    # concat lists of data from each list (dataframe of yearly arrests)
    inc_df = pd.concat(inc_df_list, ignore_index=True)

    # creating a geopandas dataframe from dataframe
    geometry = gpd.points_from_xy(inc_df.longitude, inc_df.latitude, crs="EPSG:4326")
    inc_gdf = gpd.GeoDataFrame(
    inc_df, geometry=geometry 
     )     



    # converting the espg to correct area for cook for beats and incidents to work together 
    if convert_cook_crs == True: 
        inc_gdf = inc_gdf.to_crs(epsg=26916) 
        
  
    print(inc_gdf.crs)
    print(inc_gdf.shape)

    return inc_gdf


def arr_data_read(start_year = 2014, full_dataset = True):
    
    """
    Pulling in City of Chicago Arrest data; returns pandas dataframe of arrests.
    start_year: input starting year for requested data
    full_dataset: choose to pull in full dataset or small subset of data
    """

    

    if full_dataset == True:
        print("Pulling full dataset")
        limit = 20000000
    else:
        print("Small subset")
        limit = 200
    
    # initialize datatime object
    today = dt.date.today()
    
    # getting current year
    current_yr = today.year
    # initialize a list
    arr_df_list = []
    
    # for each year between 2018 and current year, pull in arrest data    
    for year in range(start_year, current_yr + 1):
        arr_data_yr = pd.read_csv(
            f'https://data.cityofchicago.org/resource/dpt3-jri9.csv?$limit={limit}&$where=arrest_date%20between%20%27{year}-01-01T00:00:00%27%20and%20%27{year}-12-31T23:59:59%27'
    )
        arr_df_list.append(arr_data_yr)
        
    # concat lists of data from each list (dataframe of yearly arrests)
    arr_df = pd.concat(arr_df_list, ignore_index=True)
    print(arr_df.shape)

    return arr_df


def street_network_read(full_dataset = True):
    
    """
    Pulling in City of Chicago street network data; returns geopandas dataframe of transportation data.
    full_dataset: choose to pull in full dataset or small subset of data
    """

    if full_dataset == True:
        print("Pulling full dataset")
        limit = 20000000
    else:
        print("Small subset")
        limit = 200

    # pull in data
    street_gdf = gpd.read_file(
        f'https://data.cityofchicago.org/resource/pr57-gg9e.geojson?$limit={limit}'
    )
    street_gdf = street_gdf.to_crs("EPSG:26916")

    print("Read in Chicago's Full Street Network as a geopandas dataframe.")

    return street_gdf


def offense_features(df):
    
    # assign enforcement drive offenses
    enfor_do = ['GAMBLING', 'CONCEALED CARRY LICENSE VIOLATION', 'NARCOTICS', 'WEAPONS VIOLATION', 'OBSCENITY', 'PROSTITUTION', 'INTERFERENCE WITH PUBLIC OFFICER', 'LIQUOR LAW VIOLATION', 'OTHER NARCOTIC VIOLATION']
    df['Enforcement Driven Incidents'] = np.where(df['primary_type'].isin(enfor_do), 1, 0)
    
    #assign domestic battery
    df['Domestic Battery'] = np.where(df['description'].str.lower().str.contains('domestic|dom') == True, 1, 0)
    
    #Assign Domestic Violence
    df['Domestic Violence'] = np.where(
        (df['Domestic Battery'] == 1) |
        ((df['primary_type'] == 'BATTERY') & (df['domestic'] == True)) |
        ((df['primary_type'] == 'ASSAULT') & (df['domestic'] == True)) |
        ((df['primary_type'] == 'CRIM SEXUAL ASSAULT') & (df['domestic'] == True)),
        1, 0

    )
    # Remove simple marijuana possession (under 30g) and distribution/intent to sell (under 10g) from offense differences
    df['simple-cannabis'] =  np.where((df['primary_type'] == 'NARCOTICS') &
                                  (df['description'].isin(['POSS: CANNABIS 30GMS OR LESS', 'MANU/DEL:CANNABIS 10GM OR LESS'])), 1, 0)

    df['primary_type'] = np.where(df['simple-cannabis'] == 1, 'NARCOTICS-CANNABIS', df['primary_type'])
    
    df['is_gun'] = np.where(df['description'].str.lower().str.contains('gun|firearm'), 1, 0)

    # add gun possession variables
    df['gun_possession'] = np.where((df['is_gun'] ==1) & (df['description'].\
                                                        str.lower().str.contains("unlawful poss|possession|register|report") ==True), 1,0)

    df['crim_sex_offense'] = np.where((df['primary_type'] == 'CRIM SEXUAL ASSAULT')| 
                                (df['primary_type'].isin(['CRIMINAL SEXUAL ABUSE', 'AGG CRIMINAL SEXUAL ABUSE', 'AGG CRIMINAL SEXUAL ABUSE']) == True),
                                      1, 0)
    df['is_agg_assault'] =  np.where((df['primary_type'] == 'ASSAULT') & (df['description'].str.lower().str.contains('agg') == True), 1, 0)

    df['is_violent'] = np.where((df['primary_type'] == 'ROBBERY')|
                               (df['primary_type'] == 'HOMICIDE')|
                               (df['crim_sex_offense'] == 1)|
                                (df['is_agg_assault'] == 1), 1, 0)

    df['is_burglary'] = np.where(df['primary_type'] == 'BURGLARY', 1, 0)

    df['is_homicide'] = np.where(df['primary_type'] == 'HOMICIDE', 1, 0)

    df['is_theft'] = np.where(df['primary_type'] == 'THEFT', 1, 0)
    
    df['is_domestic'] = np.where(df['domestic'] == True, 1, 0)
    
    df['is_robbery'] = np.where(df['primary_type'] == 'ROBBERY', 1, 0)
    
    df['violent_gun'] = np.where((df['is_violent'] == 1) & (df['is_gun'] == 1), 1, 0)
    
    return df



def import_chi_boundaries(boundary_name = "beat"):

    """
    importing chicago boundaries and returns a geopandas dataframe
    boundary_name: the name of the chicago boundary used in the import, beat or community_area
    """
    if boundary_name == "beat":
        #import police beats
        df = gpd.read_file("https://data.cityofchicago.org/api/views/n9it-hstw/rows.geojson")
    elif boundary_name == "community_area":
        #import police beats
        df = gpd.read_file("https://data.cityofchicago.org/api/views/igwz-8jzy/rows.geojson?accessType=DOWNLOAD")
    else : 
        print("file not specified")
    

    
    return df


arr = arr_data_read(full_dataset = True)
print('Arrest data imported.')
inc = inc_data_read(full_dataset = True)
print('Incident data imported.')
inc = offense_features(inc)
inc['date'] = pd.to_datetime(inc['date'])

com = import_chi_boundaries(boundary_name = "community_area")
print('Community boundary data imported.')

sub = ['geometry','area_num_1', 'community']
com1 = com[sub]
com1 = com1.rename(columns={'area_num_1':'community_area', 'geometry':'comm_geom'})
com1['community_area'] = com1['community_area'].astype('float64')
street = street_network_read(full_dataset = True)
print('Street network data imported.')
sub = ['pre_dir','logiclf', 'street_nam','street_typ','trans_id', 'geometry']
street1 = street[sub]

inc = inc[inc.geometry.x != 80803.16843219422]
inc["is_arrest"] = inc["arrest"].astype(int)
sub = ['case_number', 'date','primary_type','arrest', 'domestic', 'beat',
       'district', 'ward', 'community_area', 'year', 'geometry', 'Enforcement Driven Incidents',
       'Domestic Battery', 'Domestic Violence', 'simple-cannabis', 'is_gun', 'gun_possession', 'is_arrest',
       'crim_sex_offense', 'is_agg_assault', 'is_violent', 'is_burglary',
       'is_homicide', 'is_theft', 'is_domestic', 'is_robbery', 'violent_gun']
inc1 = inc[sub]

inc_street_join = inc1.sjoin_nearest(street1, distance_col = "Distances")
print('Spatial join between street network and incident data completed.')
isj_sub = inc_street_join
isj_sub = pd.merge(isj_sub, com1, on='community_area', how = 'left')
isj_sub = isj_sub[isj_sub.community.notnull()]



# Street-Level Aggregates
isj_sub.loc[:, 'gun_arrests'] = isj_sub['is_arrest'] * isj_sub['is_gun']
isj_sub.loc[:, 'gun_poss_arrests'] = isj_sub['is_arrest'] * isj_sub['gun_possession']
isj_sub.loc[:, 'robbery_arrests'] = isj_sub['is_arrest'] * isj_sub['is_robbery']
isj_sub.loc[:, 'violent_arrests'] = isj_sub['is_arrest'] * isj_sub['is_violent']
isj_sub.loc[:, 'homicide_arrests'] = isj_sub['is_arrest'] * isj_sub['is_homicide']
isj_sub.loc[:, 'agg_assault_arrests'] = isj_sub['is_arrest'] * isj_sub['is_agg_assault']
isj_sub.loc[:, 'theft_arrests'] = isj_sub['is_arrest'] * isj_sub['is_theft']

def summarize(isj_sub):
    isj_sub = isj_sub.copy()
    isj_sub['year'] = isj_sub['date'].dt.year
    isj_sub['year-month'] = isj_sub['date'].dt.to_period('M').astype(str)

    seg = isj_sub.groupby('trans_id').agg(
        # crime counts
        gun_count=('is_gun', 'sum'),
        gun_poss_count=('gun_possession', 'sum'),
        robbery_count=('is_robbery', 'sum'),
        violent_count=('is_violent', 'sum'),
        homicide_count=('is_homicide', 'sum'),
        agg_assault_count=('is_agg_assault', 'sum'),
        theft_count=('is_theft', 'sum'),
        viol_gun_count=('violent_gun', 'sum'),
        total_crimes=('case_number', 'count'),  # total crimes on each street
    
        # arrest counts by crime type
        gun_arrests=('gun_arrests', 'sum'),
        gun_poss_arrests=('gun_arrests', 'sum'),
        robbery_arrests=('robbery_arrests', 'sum'),
        violent_arrests=('violent_arrests', 'sum'),
        homicide_arrests=('homicide_arrests', 'sum'),
        agg_assault_arrests=('agg_assault_arrests', 'sum'),
        theft_arrests=('theft_arrests', 'sum'),
        total_arrests=('is_arrest', 'sum'),
    
        
        # spatial-related stuff
        geometry=('geometry', 'first'),
        ward=('ward', 'first'),
        beat=('beat','first'),
        district=('district', 'first'),
        community=('community', 'first'),
        logiclf=('logiclf', 'first'),
        pre_dir=('pre_dir', 'first'),
        street_nam=('street_nam', 'first'),
        street_typ=('street_typ', 'first'),
        case_number=('case_number', 'first')

    ).reset_index()
    seg['gp_ar'] = (seg['gun_poss_arrests'] / seg['gun_poss_count']).round(2)
    seg['vi_ar'] = (seg['violent_arrests'] / seg['violent_count']).round(2)
    seg['total_ar'] = (seg['total_arrests'] / seg['total_crimes']).round(2)
    seg.fillna(0, inplace=True)
    

    # by year
    seg_time = isj_sub.groupby(['trans_id','year-month', 'year']).agg(
        violent_count=('is_violent', 'sum'),
        gun_poss_count=('gun_possession', 'sum'),
        total_crimes=('case_number', 'count'),
        gun_poss_arrests=('gun_arrests', 'sum'),
        violent_arrests=('violent_arrests', 'sum'),
        total_arrests=('is_arrest', 'sum'),
        geometry=('geometry', 'first'),
        ward=('ward', 'first'),
        beat=('beat','first'),
        district=('district', 'first'),
        community=('community', 'first'),
        logiclf=('logiclf', 'first'),
        pre_dir=('pre_dir', 'first'),
        street_nam=('street_nam', 'first'),
        street_typ=('street_typ', 'first'),
        case_number=('case_number', 'first')
    ).reset_index()
    seg_time['total_ar'] = (seg_time['total_arrests'] / seg_time['total_crimes']).round(2)
    seg_time['vi_ar'] = (seg_time['violent_arrests'] / seg_time['violent_count']).round(2)
    seg_time['gp_ar'] = (seg_time['gun_poss_arrests'] / seg_time['gun_poss_count']).round(2)
    seg_time.fillna(0, inplace=True)


    return seg, seg_time

seg, seg_time = summarize(isj_sub)
print("Crime counts by street segment in 'seg' dataframe.")

print("Crime counts by street segment grouped at year-month level in 'seg_time' dataframe.")

# Neighborhood-Level Aggregates

def summarize_neighborhoods(isj_sub):
    # by streets within a community
    street_summary = isj_sub.groupby(["community", "trans_id"]).agg(
        total_incidents=("case_number", "count"),
        violent_incidents=("is_violent", "sum"),
        gun_poss_count=("gun_possession", "sum"),
        total_arrests=("is_arrest", "sum"),
        violent_arrests=("violent_arrests", "sum"),
        gun_poss_arrests=("gun_poss_arrests", "sum"),
        comm_geom=('comm_geom', 'first')
    ).reset_index()

    # arrest rates
    street_summary["total_ar"] = street_summary["total_arrests"] / street_summary["total_incidents"].replace(0, np.nan)
    street_summary["vi_ar"] = street_summary["violent_arrests"] / street_summary["violent_incidents"].replace(0, np.nan)
    street_summary["gp_ar"] = street_summary["gun_poss_arrests"] / street_summary["gun_poss_count"].replace(0, np.nan)
    street_summary.fillna(0, inplace=True)

    # community-level summary
    comm = street_summary.groupby("community").agg(
        total_streets=("trans_id", "nunique"),
        total_incidents=("total_incidents", "sum"),
        violent_incidents=("violent_incidents", "sum"),
        gun_poss_count=("gun_poss_count", "sum"),
        total_arrests=("total_arrests", "sum"),
        violent_arrests=("violent_arrests", "sum"),
        gun_poss_arrests=("gun_poss_arrests", "sum"),
        comm_geom=('comm_geom', 'first')

    ).reset_index()

    # % of streets with specific crime types or arrests
    def pct_streets_with(df, condition_col):
        return df[df[condition_col] > 0].groupby("community")["trans_id"].nunique()

    total_streets = street_summary.groupby("community")["trans_id"].nunique()
    gp_streets = pct_streets_with(street_summary, "gun_poss_count")
    gp_arr_streets = pct_streets_with(street_summary, "gun_poss_arrests")
    vi_streets = pct_streets_with(street_summary, "violent_incidents")
    vi_arr_streets = pct_streets_with(street_summary, "violent_arrests")

    # percentage columns to comm
    comm["pct_streets_with_gun_possession"] = (gp_streets.reindex(comm["community"]).values / comm["total_streets"] * 100).round(2)
    comm["pct_streets_with_gun_possession_arrest"] = (gp_arr_streets.reindex(comm["community"]).values / comm["total_streets"] * 100).round(2)
    comm["pct_streets_with_violent_incident"] = (vi_streets.reindex(comm["community"]).values / comm["total_streets"] * 100).round(2)
    comm["pct_streets_with_violent_arrest"] = (vi_arr_streets.reindex(comm["community"]).values / comm["total_streets"] * 100).round(2)
    comm.fillna(0, inplace=True)


    return comm, street_summary


neighborhood_summary, street_summary = summarize_neighborhoods(isj_sub)
print("Crime counts by each neighborhood in 'neighborhood_summary' dataframe.")

# monthly aggregates for each street
monthly_street_counts = seg_time.groupby(['trans_id', 'year-month', 'year']).agg(
    monthly_violent_count=('violent_count', 'sum'),
    monthly_gun_poss_count=('gun_poss_count', 'sum'),
    monthly_total_crimes=('total_crimes', 'sum'),
    monthly_violent_arrests=('violent_arrests', 'sum'),
    monthly_gun_poss_arrests=('gun_poss_arrests', 'sum'),
    monthly_total_arrests=('total_arrests', 'sum') 
).reset_index()
print('Monthly counts by each street for violent and gun possession crimes and arrests in monthly_street_counts.')

# yearly count for each street
yearly_street_counts = seg_time.groupby(['trans_id', 'year']).agg(
    yearly_violent_count=('violent_count', 'sum'),
    yearly_gun_poss_count=('gun_poss_count', 'sum'),
    yearly_total_crimes=('total_crimes', 'sum'),
    yearly_violent_arrests=('violent_arrests', 'sum'),
    yearly_gun_poss_arrests=('gun_poss_arrests', 'sum'),
    yearly_total_arrests=('total_arrests', 'sum') 
).reset_index()
print('Yearly counts by each street for violent and gun possession crimes and arrests in yearly_street_counts.')



# monthly arrests for ALL streets
monthly_total = seg_time.groupby(['year-month', 'year']).agg(
    total_violent_count=('violent_count', 'sum'),
    total_gun_poss_count=('gun_poss_count', 'sum'),
    total_violent_arrests=('violent_arrests', 'sum'),
    total_gun_poss_arrests=('gun_poss_arrests', 'sum')
).reset_index()
print('Monthly counts and arrests for all streets for violent and gun possession crimes in monthly_total.')

monthly_total['vi_ar'] = round((monthly_total['total_violent_arrests'] / monthly_total['total_violent_count'])*100, 2)
monthly_total['gp_ar'] = round((monthly_total['total_gun_poss_arrests'] / monthly_total['total_gun_poss_count'])*100, 2)
print('Monthly total arrest rates calculated in monthly_total.')


# yearly for ALL streets
yearly_total = seg_time.groupby('year').agg(
    total_violent_count=('violent_count', 'sum'),
    total_gun_poss_count=('gun_poss_count', 'sum'),
    total_violent_arrests=('violent_arrests', 'sum'),
    total_gun_poss_arrests=('gun_poss_arrests', 'sum')
).reset_index()
print('Yearly counts and arrests for all streets for violent and gun possession crimes in yearly_total.')
yearly_total['vi_ar'] = round((yearly_total['total_violent_arrests'] / yearly_total['total_violent_count']) * 100, 2)
yearly_total['gp_ar'] = round((yearly_total['total_gun_poss_arrests'] / yearly_total['total_gun_poss_count']) * 100, 2)
print('Yearly total arrest rates calculated in yearly_total')


# Exporting data

# monthly_street_counts.to_csv('data/monthly_street_counts.csv')
# print('monthly_street_counts exported to CSV file')

# yearly_street_counts.to_csv('data/yearly_street_counts.csv')
# print('yearly_street_counts exported to CSV file')

monthly_total.to_csv('data/monthly_total.csv')
print('monthly_total exported to CSV file')

yearly_total.to_csv('data/yearly_total.csv')
print('yearly_total_counts exported to CSV file')


# API Key
API_KEY = os.getenv("DATAWRAPPER_API_KEY")
if not API_KEY:
    raise ValueError("DATAWRAPPER_API_KEY is not set in the environment")

dw = Datawrapper(access_token=API_KEY)


# YEARLY ARRESTS CHART
yearly_chart_id = 'gjMTR'
print("uploading yearly_total to chart")
dw.add_data(yearly_chart_id, yearly_total)

# update chart description
latest_year = sorted(yearly_total["year"].unique())[-2]
latest = yearly_total[yearly_total["year"] == latest_year]

gp_arrests = int(latest["total_gun_poss_arrests"].iloc[0])
vi_arrests = int(latest["total_violent_arrests"].iloc[0])
gp_vi_diff = gp_arrests - vi_arrests
ratio = round(gp_arrests / vi_arrests, 1)

caption_yearly = (
    f"In {latest_year}, CPD made "
    f"<b style='background-color: rgb(0 174 255); padding: 0 4px; color:white;'>{gp_vi_diff:,}</b> "
    f"more gun possession arrests than violent arrests, or "
    f"<b style='background-color: rgb(0 174 255); padding: 0 4px; color:white;'>{ratio}</b> "
    f"gun possession arrests for every violent arrest."
)

dw.update_description(yearly_chart_id, intro=caption_yearly)
print('Updating yearly chart.')
dw.publish_chart(yearly_chart_id)
print("Yearly chart updated and published.")



# MONTHLY ARRESTS CHART
monthly_chart_id = 'Wcvzf'
print('Uploading monthly_total to Datawrapper')
dw.add_data(monthly_chart_id, monthly_total)


# technically dont use this since there is no dynamic caption here
# keeping it in case we want to update in the future
latest_month_data = monthly_total[monthly_total["year"] == latest_year]
gp_month_total = int(latest_month_data["total_gun_poss_arrests"].sum())
vi_month_total = int(latest_month_data["total_violent_arrests"].sum())
ratio_month_total = round(gp_month_total / vi_month_total, 1)

caption_monthly = (
    f"After 2016, gun possession arrests started to outpace violent arrests each month. "
    f"This difference is most prominent in May 2022, when CPD made four gun possession arrests for every one violent arrest."
    )

print('Updating monthly chart description')
dw.update_description(monthly_chart_id, intro=caption_monthly)
print('Publishing monthly chart')
dw.publish_chart(monthly_chart_id)
print('Monthly chart updated')


# ARREST RATE CHARTS

# MONTHLY VIOLENT ARREST RATE

m_vi_ar_id = 'icuvq'
print("uploading monthly_total to chart")
dw.add_data(m_vi_ar_id, monthly_total)

# update chart description

# 2014 arrest rate
m_start_rate = monthly_total[monthly_total["year-month"] == "2014-01"]["vi_ar"].iloc[0]
m_end_rate = monthly_total[monthly_total["year"] == latest_year]["vi_ar"].iloc[-1] # most recent full year arrest rate
m_drop = round(m_start_rate - m_end_rate, 1)


m_vi_ar_cap = (
    f"Since 2014, the Chicago Police's arrest rate for violent crime dropped from "
    f"<b style='background-color: #ffc800; padding: 0 4px;'>{round(m_start_rate)}%</b> to "
    f"<b style='background-color: #ffc800; padding: 0 4px;'>{round(m_end_rate)}%</b> — a decrease of "
    f"<b style='background-color: #ffc800; padding: 0 4px;'>{m_drop}%</b>."
)

dw.update_description(m_vi_ar_id, intro=m_vi_ar_cap)
dw.publish_chart(m_vi_ar_id)
print("Chart updated and published.")


# YEARLY VIOLENT ARREST RATE

y_vi_ar_id = 't4b8K'
print("uploading yearly_total to chart")
dw.add_data(y_vi_ar_id, yearly_total)

# update chart description

# 2014 violent arrest rate
y_start_rate = yearly_total[yearly_total["year"] == 2014]["vi_ar"].iloc[0]
y_end_rate = latest["vi_ar"].iloc[0]
y_drop = round(y_start_rate - y_end_rate, 1)


y_vi_ar_cap = (
    f"Since 2014, the Chicago Police's arrest rate for violent crime dropped from "
    f"<b style='background-color: #ffc800; padding: 0 4px;'>{round(y_start_rate)}%</b> to "
    f"<b style='background-color: #ffc800; padding: 0 4px;'>{round(y_end_rate)}%</b> — a decrease of "
    f"<b style='background-color: #ffc800; padding: 0 4px;'>{y_drop}%</b>."
)

dw.update_description(y_vi_ar_id, intro=y_vi_ar_cap)
dw.publish_chart(y_vi_ar_id)
print("Chart updated and published.")


### 3 YEAR ROLLING AVERAGES ###

# filter only full years 
latest_full_year = monthly_total["year"].max() - 1
full_years = monthly_total[monthly_total["year"] <= latest_full_year]

# yearly totals
yearly_counts = full_years.groupby("year").agg(
    total_vi_arrs = ("total_violent_arrests", "sum"),
    total_gp_arrs = ("total_gun_poss_arrests", "sum")
). reset_index()

# calculate ratio
yearly_counts["gp_to_vi_ratio"] = (
    yearly_counts["total_gp_arrs"] / yearly_counts["total_vi_arrs"]
).round(2)

# calculate rolling 3year averages
yearly_counts["violent_arrests_3yr_avg"] = (
    yearly_counts["total_vi_arrs"].rolling(window=3).mean()
).round(2)
yearly_counts["gun_arrests_3yr_avg"] = (
    yearly_counts["total_gp_arrs"].rolling(window=3).mean()
).round(2)
yearly_counts["ratio_3yr_avg"] = (
    yearly_counts["gp_to_vi_ratio"].rolling(window=3).mean()
).round(2)

# dropping first two rows since they don't have full 3yr avgs calculated
three_yr_avgs = yearly_counts.iloc[2:] 
print(three_yr_avgs.iloc[:, -3:])
three_yr_avgs.to_csv('data/three_yr_avgs.csv')
print('three_yr_avgs exported to CSV file')

# DATAWRAPPER
roll_id = 'Sfdp9'
print("uploading three_yr_avgs to chart")
dw.add_data(roll_id, three_yr_avgs)
print("three_yr_avgs is added to chart")


### MONTH-TO-MONTH COMPARISONS, YEAR OVER YEAR ###

mtm_ch = monthly_total.copy()
mtm_ch['year-month'] = pd.to_datetime(mtm_ch['year-month'])

mtm_ch['year'] =mtm_ch['year-month'].dt.year
mtm_ch['month'] = mtm_ch['year-month'].dt.month
mtm_ch['month_name'] = mtm_ch['year-month'].dt.strftime('%b') 

vi_co_pivot = mtm_ch.pivot(index='year', columns='month_name', values='total_violent_count')
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
vi_co_pivot = vi_co_pivot[month_order]

gp_co_pivot = mtm_ch.pivot(index='year', columns='month_name', values='total_gun_poss_count')
gp_co_pivot = gp_co_pivot[month_order]


vi_co_pivot.to_csv('data/vi_co_pivot.csv')
print('vi_co_pivot exported to CSV file')
gp_co_pivot.to_csv('data/gp_co_pivot.csv')
print('gp_co_pivots exported to CSV file')
