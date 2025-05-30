import geopandas as gpd
import pandas as pd
import datetime as dt
import pyarrow

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from pathlib import Path

import pandas as pd
import numpy as np
# from loguru import logger
# from tqdm import tqdm

# from ln.config import PROCESSED_DATA_DIR


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
inc = inc_data_read(full_dataset = True)
inc = offense_features(inc)
inc['date'] = pd.to_datetime(inc['date'])

com = import_chi_boundaries(boundary_name = "community_area")

sub = ['geometry','area_num_1', 'community']
com1 = com[sub]
com1 = com1.rename(columns={'area_num_1':'community_area', 'geometry':'comm_geom'})
com1['community_area'] = com1['community_area'].astype('float64')
street = street_network_read(full_dataset = True)
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

# Saving to Files

# seg
seg = gpd.GeoDataFrame(seg, geometry='geometry', crs='EPSG:26916')
seg_final = street.sjoin_nearest(seg, how = 'left')
seg_final = seg_final.to_crs("EPSG:4326")

sub = ['logiclf_right', 'pre_dir_right', 'street_nam_right', 'street_typ_right', 
       'ward', 'beat', 'district', 'community', 'case_number','geometry', 'index_right', 'trans_id_right', 
       'gun_count', 'gun_poss_count', 'robbery_count', 'violent_count', 'homicide_count', 'agg_assault_count',
       'theft_count', 'viol_gun_count', 'total_crimes', 'gun_arrests', 'gun_poss_arrests', 'robbery_arrests', 
       'violent_arrests', 'homicide_arrests', 'agg_assault_arrests', 'theft_arrests',
       'total_arrests', 'gp_ar', 'vi_ar', 'total_ar']
seg_final = seg_final[sub]
seg_final.rename(columns={'logiclf_right':'logiclf', 
                          'pre_dir_right': 'pre_dir', 'street_nam_right': 'street_nam', 
                          'street_typ_right': 'street_typ', 'trans_id_right':'trans_id',
                          'index_right':'index'},
                 inplace = True)
for col in seg_final.select_dtypes(include='object').columns:
    seg_final[col] = seg_final[col].astype(str)

seg_final.to_parquet(r'C:\Users\clohe\Documents\loyola_northwestern\data\processed\seg_summary.parquet')

# seg_time
seg_time = gpd.GeoDataFrame(seg_time, geometry='geometry', crs='EPSG:26916')
street = street.to_crs(seg_time.crs) 
seg_time_final = gpd.sjoin_nearest(seg_time, street, how='left')
seg_time_final = seg_time_final.to_crs("EPSG:4326")

sub = ['ward', 'beat', 'district', 'community', 'logiclf_right', 'pre_dir_right', 'street_nam_right',
       'street_typ_right', 'case_number', 'geometry', 'index_right', 'trans_id_right', 'year-month', 'year',
       'violent_count', 'gun_poss_count', 'total_crimes', 'gun_poss_arrests', 'violent_arrests',
       'total_arrests', 'total_ar', 'vi_ar', 'gp_ar'
]
seg_time_final = seg_time_final[sub]
seg_time_final.rename(columns={'logiclf_right':'logiclf', 
                          'pre_dir_right': 'pre_dir', 'street_nam_right': 'street_nam', 
                          'street_typ_right': 'street_typ', 'trans_id_right':'trans_id',
                          'index_right':'index'},
                 inplace = True)

seg_time_final.to_parquet(r'C:\Users\clohe\Documents\loyola_northwestern\data\processed\seg_time.parquet')

# neighborhood_summary
neighborhood_summary = gpd.GeoDataFrame(neighborhood_summary, geometry='comm_geom', crs='EPSG:26916')
neighborhood_summary.to_parquet(r'C:\Users\clohe\Documents\loyola_northwestern\data\processed\neighborhood_summary.parquet')


