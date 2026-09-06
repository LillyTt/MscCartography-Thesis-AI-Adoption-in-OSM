#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------------------------
# Script to download all indicators into designated folders and save 
# ---------------------------------------------------------------------------

"""
Country Indicators Data Extraction
"""

import logging
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import networkx as nx
import matplotlib.ticker as mticker
import re




logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

FILEPATH_COMMENTS       = r"data\input\piebro\changeset_comments_data\*.parquet"
FILEPATH_NOTES_COMMENTS = r"data\input\piebro\notes_comments_data\*.parquet"
FILEPATH_NOTES          = r"data\input\piebro\notes_data\*.parquet"
FILEPATH_COUNTRIES_GEO  = r"UN_Countries_Simplified\UN_Countries_Simplified.shp"
FILEPATH_GLOBAL_CONTRIBS = r"data\ouput\new_contribs.csv"
FILEPATH_CONTRIBS_XP = r"data\ouput\raw_contributor_experience_uid_ai_total_country_ohsome.csv"

FILEPATH_REPORT_RESULTS = Path(r"\results\country_analysis\v5_indicators")
FILEPATH_CSV_RESULTS = Path(r"data\ouput\country_analysis\v5_indicators")
plain_formatter = FuncFormatter(lambda x, _: f"{int(x):,}")
humanitarian_hashtags = ['hotosm-project-', 'missing_maps', 'missingmaps']
corporate_hashtags = [
    'amap', 'adt', 'bolt', 'DigitalEgypt', 'expedia', 'gojek', 'MSFTOpenMaps', 'grab',
    'Kaart', 'Kontur', 'mbx', 'RocketData', 'disputed_by_claimed_by', 'Snapp', 'stackbox',
    'Telenav', 'Lightcyphers', 'tomtom', 'TIDBO', 'WIGeoGIS-OMV', 'نشان', 'mapbox',
    'Komoot', 'AppLogica']

def build_pattern(words):
        return '|'.join(re.escape(w) for w in words)

def load_data(filepath: str, country: str) -> tuple[duckdb.DuckDBPyConnection, gpd.GeoDataFrame, str, str]:
    con = duckdb.connect()

    # reading in the raw parquet files
    con.execute(f"CREATE VIEW db_raw AS SELECT * FROM read_parquet('{filepath}')")

    # selection where columns are renamed, only edits in the country and data after 2019 is selected 
    con.execute(f"""
        CREATE VIEW db AS
        SELECT
            osm_type, osm_id, osm_version, osm_minor_version,
            osm_edits, osm_last_edit, status, contrib_type,
            valid_from, valid_to,

            "user".id   AS user_id,
            "user".name AS user_name,

            changeset.id         AS changeset_id,
            changeset.created_at AS changeset_created_at,
            changeset.closed_at  AS changeset_closed_at,
            changeset.hashtags   AS hashtags,
            changeset.editor     AS editor,
            changeset.tags       AS changeset_tags,

            geometry, geometry_type, bbox, centroid,
            area, area_delta, length, length_delta,
            tags, tags_before,
            countries, refs, refs_count,
            members, members_count, build_time
        FROM db_raw
        WHERE list_contains(countries, '{country}') AND valid_from >= '2020-01-01'

    """)

    # creating the view of the AI rows - to not rerun it everytime 
    con.execute("""
        CREATE VIEW db_ai AS
        SELECT * FROM db
        WHERE
            REGEXP_EXTRACT(editor, '^([A-Za-z]+)', 1) ILIKE 'Rapid'
            OR element_at(tags, 'source')[1]           ILIKE '%microsoft/BuildingFootprints%'
            OR element_at(tags, 'source')[1]           ILIKE '%esri/Google_Africa_Buildings%'
            OR element_at(tags, 'source')[1]           ILIKE '%esri/Google_Open_Buildings%'
            OR len(list_filter(hashtags, x -> x ILIKE '%mapwithai%')) > 0
            OR element_at(changeset_tags, 'source')[1] ILIKE '%microsoft/BuildingFootprints%'
            OR element_at(changeset_tags, 'source')[1] ILIKE '%esri/Google_Open_Buildings%'
            OR element_at(changeset_tags, 'source')[1] ILIKE '%mapwithai%'
    """)

    # reading in the additional files from the global changeset dataset (piebro) as well as the country file
    con.execute(f"CREATE VIEW cc AS SELECT * FROM read_parquet('{FILEPATH_COMMENTS}')")
    con.execute(f"CREATE VIEW nc AS SELECT * FROM read_parquet('{FILEPATH_NOTES_COMMENTS}')")
    con.execute(f"CREATE VIEW nd AS SELECT * FROM read_parquet('{FILEPATH_NOTES}')")
    #con.execute(f"CREATE VIEW org_teams AS SELECT * FROM read_csv('{FILEPATH_ORGANISED_TEAMS}')")
    countries = gpd.read_file(FILEPATH_COUNTRIES_GEO)
          
    logger.info("Loaded all datasets (main, comments, notes, countries)")
    
    return con, countries



# set up functions to make df creation easier
    
def pivot_table(df, index, columns, values):
    pivot = df.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc='sum'
    ).reset_index().fillna(0)
    return pivot

def build_pivot_set(df, index_col, group_col, value_cols, rename_fn):
    tables = [
        rename_fn(pivot_table(df, index_col, group_col, col), col)
        for col in value_cols
    ]
    result = tables[0]
    for t in tables[1:]:
        result = result.merge(t, on=index_col)
    return result
    
def rename_contrib_type(df, contrib_type):
    df = df.rename(columns={
        'CREATION':f'{contrib_type}_creation',
         'DELETION':f'{contrib_type}_deletion',
         'GEOMETRY':f'{contrib_type}_geometry',
        'TAG':f'{contrib_type}_tag',
        'TAG_GEOMETRY':f'{contrib_type}_tag_geometry'
    })
    return df

    
def rename_feature_type(df, feature_type):
    df = df.rename(columns={
        'building':f'{feature_type}_building',
         'highway':f'{feature_type}_highway',
         'other':f'{feature_type}_other'
    })
    return df


def rename_xp_total(df, xp_type):
    df = df.rename(columns={
        'Casual':f'{xp_type}_casual_total',
         'Inactive':f'{xp_type}_inactive_total',
         'Prolific':f'{xp_type}_prolific_total'
    })
    return df

def rename_xp(df, xp_type):
    df = df.rename(columns={
        'Casual':f'{xp_type}_casual_ai',
         'Inactive':f'{xp_type}_inactive_ai',
         'Prolific':f'{xp_type}_prolific_ai'
    })
    return df
    
# ---------------------------------------------------------------------------
#
# CONTENT INDICATORS
#
# ---------------------------------------------------------------------------

# Yearly Edits, Changesets and Contributors 

def content_yts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    ts = con.sql("""
        SELECT 
            YEAR(valid_from) as year,
            COUNT(*) as Edits,
            COUNT(DISTINCT user_id) as Contributors,
            COUNT(DISTINCT changeset_id) as Changesets,
            SUM(contributors) OVER (ORDER BY year) as "Accumulated Contributors",
            SUM(edits) OVER (ORDER BY year) as "Accumulated Edits",
            SUM(changesets) OVER (ORDER BY year) as "Accumulated Changesets"
        FROM db
        GROUP BY year
        ORDER BY year
    """).df()

    ts_ai = con.sql("""
        SELECT 
            YEAR(valid_from) as year,
            COUNT(DISTINCT user_name) as ContributorsAI,
            COUNT(*) as EditsAI,
            COUNT(DISTINCT changeset_id) as ChangesetsAI
        FROM db_ai
        GROUP BY year
        ORDER BY year
    """).df()

    ts_ai = ts_ai.fillna(0)
    ts = ts.fillna(0)

    df = ts.merge(ts_ai, on= "year", how ="left")
    df['editsPerc'] = (df['EditsAI']/df['Edits'])*100
    df['changesetsPerc'] = (df['ChangesetsAI']/df['Changesets'])*100
    df['contributorsPerc'] = (df['ContributorsAI']/df['Contributors'])*100
    return df    

def new_ycontributors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    dfg = pd.read_csv(FILEPATH_GLOBAL_CONTRIBS).drop(columns={"Unnamed: 0"})
    dfg = dfg.loc[(dfg["year"]>2019) &  (dfg["months"]<"2026-03")]

    dfn = con.sql("""
        WITH user_first_changeset AS (
            SELECT
                valid_from,
                user_name,
                changeset_id AS changeset_id,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY valid_from, changeset_id) as rn
            FROM db
        )
        SELECT
            user_name,
            changeset_id,
           YEAR(valid_from) as first_edit,
        FROM user_first_changeset
        WHERE rn = 1 
        ORDER BY first_edit, user_name
    """).df()

    df = dfn.merge(dfg, on=["user_name", "changeset_id"], how="left")
    df['type'] = df['year'].isna().map({True: 'local', False: 'global'})

    new_contribs = (
        df
        .groupby(['first_edit', 'type'])
        .size()
        .reset_index(name='new_contributors')
    )
    new_contribs_clean = new_contribs.pivot_table(
        index='first_edit',
        columns='type',
        values='new_contributors',
        aggfunc='sum'
    ).reset_index()

    new_contribs_clean.columns.name = None
    new_contribs_clean = new_contribs_clean.rename(columns={
        'global': 'global_new_contrib',
        'local': 'local_new_contrib'
    })

    return new_contribs_clean

def yearly_stats(con) -> pd.DataFrame:
    ndf = new_ycontributors(con) #new contributors df
    ts = content_yts(con)

    df = ts.merge(ndf, left_on="year", right_on="first_edit", how="left")
    return df

# Monthly Stats: Changesets, Edits, Contributors, New Contributors (global, local), Share in AI

def content_ts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    ts = con.sql("""
        SELECT 
            YEAR(valid_from) as year,
            STRFTIME(valid_from, '%Y-%m') as months,
            COUNT(*) as Edits,
            COUNT(DISTINCT user_id) as Contributors,
            COUNT(DISTINCT changeset_id) as Changesets,
            SUM(contributors) OVER (ORDER BY year) as "Accumulated Contributors",
            SUM(edits) OVER (ORDER BY year) as "Accumulated Edits",
            SUM(changesets) OVER (ORDER BY year) as "Accumulated Changesets"
        FROM db
        GROUP BY year, months
        ORDER BY months
    """).df()

    ts_ai = con.sql("""
        SELECT 
            STRFTIME(valid_from, '%Y-%m') as months,
            COUNT(DISTINCT user_name) as ContributorsAI,
            COUNT(*) as EditsAI,
            COUNT(DISTINCT changeset_id) as ChangesetsAI
        FROM db_ai
        GROUP BY months
        ORDER BY months
    """).df()

    ts_ai = ts_ai.fillna(0)
    ts = ts.fillna(0)

    df = ts.merge(ts_ai, on= "months", how ="left")
    df = df.fillna(0)
    df['editsPerc'] = (df['EditsAI']/df['Edits'])*100
    df['changesetsPerc'] = (df['ChangesetsAI']/df['Changesets'])*100
    df['contributorsPerc'] = (df['ContributorsAI']/df['Contributors'])*100
    return df    

def new_contributors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    dfg = pd.read_csv(FILEPATH_GLOBAL_CONTRIBS).drop(columns={"Unnamed: 0"})
    dfg = dfg.loc[dfg["months"]<"2026-03"]

    dfn = con.sql("""
        WITH user_first_changeset AS (
            SELECT
                valid_from,
                user_name,
                tags,
                editor,
                hashtags,
                changeset_tags,
                changeset_id AS changeset_id,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY valid_from, changeset_id) as rn
            FROM db
        )
        SELECT
            user_name,
            changeset_id,
           STRFTIME(valid_from, '%Y-%m') as first_edit,
           (
                REGEXP_EXTRACT(editor, '^([A-Za-z]+)', 1) ILIKE 'Rapid'
                OR element_at(tags, 'source')[1] ILIKE '%microsoft/BuildingFootprints%'
                OR element_at(tags, 'source')[1] ILIKE '%esri/Google_Africa_Buildings%'
                OR element_at(tags, 'source')[1] ILIKE '%esri/Google_Open_Buildings%'
                OR len(list_filter(hashtags, x -> x ILIKE '%mapwithai%')) > 0
                OR element_at(changeset_tags, 'source')[1] ILIKE '%microsoft/BuildingFootprints%'
                OR element_at(changeset_tags, 'source')[1] ILIKE '%esri/Google_Open_Buildings%'
                OR element_at(changeset_tags, 'source')[1] ILIKE '%mapwithai%'
            ) AS is_AI
        FROM user_first_changeset
        WHERE rn = 1 
        ORDER BY first_edit, user_name
    """).df()

    df = dfn.merge(dfg, on=["user_name", "changeset_id"], how="left")
    df['type'] = df['year'].isna().map({True: 'local', False: 'global'})

    new_contribs = (
        df
        .groupby(['first_edit', 'type', 'is_AI'])
        .size()
        .reset_index(name='new_contributors')
    )
    new_contribs_clean = new_contribs.pivot_table(
        index='first_edit',
        columns='type',
        values='new_contributors',
        aggfunc='sum'
    ).reset_index()

    new_contribs_clean.columns.name = None
    new_contribs_clean = new_contribs_clean.rename(columns={
        'global': 'global_new_contrib',
        'local': 'local_new_contrib'
    })
    new_contribs_clean = new_contribs_clean.fillna(0)
    new_contribs_clean_ai = new_contribs.pivot_table(
        index='first_edit',
        columns='is_AI',
        values='new_contributors',
        aggfunc='sum'
    ).reset_index()
    
    new_contribs_clean_ai.columns.name = None
    new_contribs_clean_ai = new_contribs_clean_ai.rename(columns={
        True: 'new_contrib_ai',
        False: 'new_contrib_non_ai'
    })
    new_contribs_clean_ai = new_contribs_clean_ai.fillna(0)
    new_contribs_clean = new_contribs_clean.merge(new_contribs_clean_ai, on="first_edit", how ="left")
    return new_contribs_clean


def monthly_stats(con) -> pd.DataFrame:
    ndf = new_contributors(con) #new contributors df
    ts = content_ts(con)

    df = ts.merge(ndf, left_on="months", right_on="first_edit", how="left")
    df=df.loc[df["months"]<="2026-03-01"]
    return df



def plot_overview(df: pd.DataFrame, country: str) -> plt.Figure:
    df = df.copy() #monthly_stats(con)

    df['months'] = pd.to_datetime(df['months'], format="%Y-%m")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(2,2, figsize=(10, 6))
    fig.suptitle(f"{country} Overview 2020-01 - 2026-03", fontsize=13, fontweight="bold", y=0.95)

    ax0 = ax[0,0]
    ax0.set_title("Total Changesets", fontsize=9)
    ax0.plot(df["months"], df["Changesets"], lw=1, color="#6A8D73")
    ax0.fill_between(df["months"], 0, df["Changesets"], alpha=.3, color="#6A8D73")
    ax0.yaxis.set_major_formatter(plain_formatter)
    ax0.set_xlim(np.datetime64('2020-01-01'), np.datetime64('2026-03-01'))
    ax0.set_ylim(0,)
    ax0.grid(linewidth=0.4)
    ax0.tick_params(axis='both', which='major', labelsize=8)

    ax1 = ax[0,1]
    ax1.set_title("Total Edits", fontsize=9)
    ax1.plot(df["months"], df["Edits"], lw=1,  color="#A9CDEF")
    ax1.fill_between(df["months"], 0, df["Edits"], alpha=.3, color="#A9CDEF")
    ax1.yaxis.set_major_formatter(plain_formatter)
    ax1.set_xlim(np.datetime64('2020-01-01'), np.datetime64('2026-03-01'))
    ax1.set_ylim(0,)
    ax1.grid(linewidth=0.4)
    ax1.tick_params(axis='both', which='major', labelsize=8)

    ax2 = ax[1,0]
    ax2.set_title("Monthly Contributors", fontsize=9)
    sns.lineplot(x="months", y="Contributors", data=df,
            label="Total Contributors", color="#F3A916", linewidth=0.7, ax = ax2)
    ax2.fill_between(df["months"], 0, df["Contributors"], alpha=.3, color="#F3A916")
    sns.lineplot(x="months", y="local_new_contrib", data=df,
                label="New Contributors Local", color="#AE7709", linewidth=0.7, ax = ax2)
    sns.lineplot(x="months", y="global_new_contrib", data=df,
                label="New Contributors Global", color="#4E3504", linewidth=0.7, ax = ax2)

    ax2.legend(ncol=1, loc="upper left", frameon=True, facecolor="w")
    ax2.grid(linewidth=0.3)
    ax2.set_xlim(np.datetime64('2020-01-01'), np.datetime64('2026-03-01'))
    ax2.set_ylim(0,)
    ax2.legend(fontsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.set_xlabel('Years', fontsize=9)
    ax2.set_ylabel("Contributors", fontsize=9)

    ax3 = ax[1,1]
    ax3.set_title(f"Share of AI-assisted Mapping in {country}", fontsize=9)
    sns.lineplot(x = 'months',
                 y='editsPerc',
                 data=df,
                 linewidth=1,
                 color = "#A9CDEF",
                 label = 'Edits',
                 ax = ax3
            )
    sns.lineplot(x = 'months',
                 y='changesetsPerc',
                 data=df,
                 linewidth=1,
                 color = "#6A8D73",
                 label = "Changesets",
                 ax = ax3
            )
    sns.lineplot(x = 'months',
                 y='contributorsPerc',
                 data=df,
                 linewidth=1,
                 color = "#F3A916",
                 label = "Contributors",
                 ax = ax3
            )
    ax3.set_xlabel('Years')
    ax3.set_ylabel("Percentage (%)")
    ax3.grid(linewidth=0.3)
    ax3.set_xlim(np.datetime64('2020-01-01'), np.datetime64('2026-03-01'))
    ax3.set_ylim(0,)
    ax3.legend(fontsize=9)
    ax3.tick_params(axis='both', which='major', labelsize=8)

    fig.tight_layout()
    return fig



# Contribution and Feature Type 
def contrib_type(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    ai_contrib = con.sql("""
        SELECT 
            STRFTIME(valid_from, '%Y-%m') as months,
            COUNT(DISTINCT user_name) as contributors_ai,
            COUNT(*) as edits_ai,
            COUNT(DISTINCT changeset_id) as changesets_ai,
            contrib_type
        FROM db_ai
        WHERE contrib_type != '' 
        GROUP BY months, contrib_type
        ORDER BY months DESC
    """).df()

    total_contrib = con.sql("""
        SELECT 
            YEAR(valid_from) as year,
            STRFTIME(valid_from, '%Y-%m') as months,
            contrib_type,
            COUNT(DISTINCT user_name) as contributors,
            COUNT(*) as edits,
            COUNT(DISTINCT changeset_id) as changesets
        FROM db
        WHERE contrib_type != ''
        GROUP BY months, year, contrib_type
        ORDER BY months DESC
    """).df()

    combi = total_contrib.merge(ai_contrib, on=("months","contrib_type"), how = "left")
    combi['share_edits'] = (combi['edits_ai'] / combi['edits'])*100
    combi['share_contributors'] = (combi['contributors_ai'] / combi['contributors'])*100
    combi['share_changesets'] = (combi['changesets_ai'] / combi['changesets'])*100
    combi = combi.fillna(0)

    contrib_cols = ['contributors', 'contributors_ai', 'changesets', 'changesets_ai', 'edits', 'edits_ai']
    all_contributions = build_pivot_set(combi, 'months', 'contrib_type', contrib_cols, rename_contrib_type)


    return combi, all_contributions


def feature_type(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    type_total = con.sql("""
        SELECT YEAR(valid_from) AS year,
        STRFTIME(valid_from, '%Y-%m') as months,
        CASE 
            WHEN element_at(tags, 'building') != [] THEN 'building'
            WHEN element_at(tags, 'highway') != [] THEN 'highway'
            ELSE 'other'
        END AS tag_type,
        COUNT(DISTINCT user_name) as contributors,
        COUNT(*) as edits,
        COUNT(DISTINCT changeset_id) as changesets
        FROM db
        GROUP BY months, year, tag_type
        ORDER BY months DESC
    """).df()

    type_ai = con.sql("""
        SELECT STRFTIME(valid_from, '%Y-%m') as months,
        CASE 
            WHEN element_at(tags, 'building') != [] THEN 'building'
            WHEN element_at(tags, 'highway') != [] THEN 'highway'
            ELSE 'other'
        END AS tag_type,
        COUNT(DISTINCT user_name) as contributors_ai,
        COUNT(*) as edits_ai,
        COUNT(DISTINCT changeset_id) as changesets_ai
        FROM db_ai
        GROUP BY months, tag_type
        ORDER BY months DESC
    """).df()

    type_combi = type_total.merge(type_ai, on=("months","tag_type"), how="left")
    type_combi= type_combi.fillna(0)
    type_combi['share_edits'] = (type_combi['edits_ai'] / type_combi['edits'])*100
    type_combi['share_contributors'] = (type_combi['contributors_ai'] / type_combi['contributors'])*100
    type_combi['share_changesets'] = (type_combi['changesets_ai'] / type_combi['changesets'])*100

    feature_specs = ['contributors', 'contributors_ai', 'share_contributors', 'edits','edits_ai', 'share_edits', 'changesets', 'changesets_ai', 'share_changesets']
    all_features = build_pivot_set(type_combi, 'months', 'tag_type', feature_specs, rename_feature_type)


    return type_combi, all_features


def plot_contrib_feature(df1: pd.DataFrame, df2: pd.DataFrame, country: str) -> plt.Figure:
    contrib_df = df1.copy() #contrib_type(con)
    feature_df = df2.copy() #feature_type(con)
    contrib_df['months'] = pd.to_datetime(contrib_df['months'], format="%Y-%m")
    feature_df['months'] = pd.to_datetime(feature_df['months'], format="%Y-%m")
   
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    sns.set_theme(style="whitegrid")

    ax1 = ax[0]
    ax1.set_title(f"Contribution Type of AI Assisted Mapping in {country}", fontsize=10, fontweight="bold")
    sns.lineplot(x = 'months',
                 y='share_changesets',
                 data=contrib_df,
                 hue="contrib_type",
                 linewidth=1,
                 ax = ax1
                )
    ax1.grid(linewidth=0.4)
    ax1.set_xlim(2020, 2026)
    ax1.set_ylim(0,)
    ax1.legend(fontsize=9)
    ax1.set_ylabel("Changesets (%)")
    ax1.set_xlabel("Months")

    ax2 = ax[1]
    ax2.set_title(f"Feature Type of AI Assisted Mapping in {country}", fontsize=10, fontweight="bold")
    sns.lineplot(x = 'months',
                 y='share_changesets',
                 data=feature_df,
                 hue="tag_type",
                 linewidth=1,
                 ax = ax2
                )
    ax2.grid(linewidth=0.4)
    ax2.set_xlim(2020, 2026)
    ax2.set_ylim(0,)
    ax2.set_ylabel("Changesets (%)")
    ax2.set_xlabel("Months")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    return fig


    
# ---------------------------------------------------------------------------
#
# User Indicators
#
# ---------------------------------------------------------------------------

#helper fuciton to transofrm the raw data into monthly aggregates
def create_monthly_pivot_xp(df, value, aggf):
    result = df.pivot_table(
        index='month',
        columns='contributor_type',
        values=value,
        aggfunc=aggf
    ).reset_index()
    return result

def ohsome_contrib_xp(country: str):
    wrld = pd.read_csv(FILEPATH_CONTRIBS_XP)
    if country =="NZ1": 
        df = wrld.loc[wrld["country"]=="NZL"]
    elif country == "B35":
        df = wrld.loc[wrld["country"]=="GEO"]
    elif country == "US1":
        df = wrld.loc[wrld["country"]=="USA"]
    else: df = wrld.loc[wrld["country"]==country]
    df_ai = df.loc[df["n_changesets_AI"]>0]
    
    
    monthly_ctypes = create_monthly_pivot_xp(df, 'user_id', 'count').rename(columns={"Casual": "contributors_casual_total", "Inactive": "contributors_inactive_total", "Prolific":"contributors_prolific_total"})
    monthly_ctypes_ai = create_monthly_pivot_xp(df_ai, 'user_id', 'count').rename(columns={"Casual": "contributors_casual_ai", "Inactive": "contributors_inactive_ai", "Prolific":"contributors_prolific_ai"})
    monthly_edits = create_monthly_pivot_xp(df, 'n_edits', 'sum').rename(columns={"Casual": "edits_casual_total", "Inactive": "edits_inactive_total", "Prolific":"edits_prolifiv_total"})
    monthly_changesets = create_monthly_pivot_xp(df, 'n_changesets', 'sum').rename(columns={"Casual": "changesets_casual_total", "Inactive": "changesets_inactive_total", "Prolific":"changesets_prolific_total"})
    monthly_edits_ai = create_monthly_pivot_xp(df, 'n_edits_AI', 'sum').rename(columns={"Casual": "edits_casual_ai", "Inactive": "edits_inactive_ai", "Prolific":"edits_prolific_ai"})
    monthly_changesets_ai = create_monthly_pivot_xp(df, 'n_changesets_AI', 'sum').rename(columns={"Casual": "changesets_casual_ai", "Inactive": "changesets_inactive_ai", "Prolific":"changesets_prolific_ai"})

    monthly = monthly_ctypes.merge(monthly_edits, on="month")
    monthly = monthly.merge(monthly_ctypes_ai, on="month")
    monthly = monthly.merge(monthly_changesets, on="month")
    monthly = monthly.merge(monthly_edits_ai, on="month")
    monthly = monthly.merge(monthly_changesets_ai, on="month")
    
    monthly['month'] = monthly['month'].str[:7].astype("string")
    #monthly["month"] = pd.to_datetime(monthly["month"], format="%Y-%m")
    monthly = monthly.rename(columns={"month": "months"}).fillna(0)
    
    return monthly

def plot_ohsome_contrib_xp(df: pd.DataFrame, country: str) -> plt.Figure:   
    monthly = df
    monthly["months"] = pd.to_datetime(monthly["months"], format="%Y-%m")
    
    monthly_changesets_ai = monthly[["months", "changesets_casual_ai", "changesets_inactive_ai", "changesets_prolific_ai"]]
    monthly_ctypes = monthly[["months", "contributors_casual_total", "contributors_inactive_total", "contributors_prolific_total"]]
    
    categories = ["contributors_casual_total", "contributors_inactive_total", "contributors_prolific_total",
                  "changesets_casual_ai", "changesets_inactive_ai", "changesets_prolific_ai"]
    colors = {"contributors_casual_total": "blue", "contributors_inactive_total": "green", "contributors_prolific_total": "red",
                  "changesets_casual_ai" :"lightblue", "changesets_inactive_ai":"lightgreen", "changesets_prolific_ai":"salmon"}

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax1 = ax[0]
    ax1.set_title(f"Monthly Number of Contributor types (total) in {country}", fontsize=10, fontweight="bold")
    monthly_ctypes.plot(
        x ="months",
        ax=ax1,
        linewidth=1,
        color=colors
    )
    ax1.set_ylabel("Number Contributors")
    ax1.set_xlabel("Months")
    ax1.legend()
    ax1.grid(linewidth=0.2, color = "gray")
    ax1.set_ylim(0,)
    ax1.set_xlim('2020-01', '2026-06')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    
    ax2 = ax[1]
    ax1.set_title(f"Monthly Skill Level from Contributors using AI in {country}", fontsize=10, fontweight="bold")
    monthly_changesets_ai.plot(
        x ="months",
        ax=ax2,
        linewidth=1,
        color=colors
    )
    ax2.set_ylabel("Number Changesets")
    ax2.set_xlabel("Months")
    #ax2.set_title("Monthly SKill Level from Contributors for AI Changesets", fontsize=10, fontweight="bold")
    ax2.legend()
    ax2.grid(linewidth=0.2, color = "gray")
    ax2.set_ylim(0,)
    ax2.set_xlim('2020-01', '2026-06')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))   
    fig.tight_layout()
    return fig
    

#  Affiliation of Edits
def affiliation_all(con: duckdb.DuckDBPyConnection, corporate_pattern: str, humanitarian_pattern: str) -> pd.DataFrame:
    affiliation_all = con.sql("""
        WITH flagged AS (
        SELECT
            valid_from,
            user_id,
            changeset_id,
           (regexp_matches(array_to_string(hashtags, ' '), $corporate, 'i') AND NOT regexp_matches(array_to_string(hashtags, ' '), $humanitarian, 'i')) AS is_corporate,
            regexp_matches(array_to_string(hashtags, ' '), $humanitarian, 'i') AS is_humanitarian,
            (
                REGEXP_EXTRACT(editor, '^([A-Za-z]+)', 1) ILIKE 'Rapid'
                OR element_at(tags, 'source')[1] ILIKE '%microsoft/BuildingFootprints%'
                OR element_at(tags, 'source')[1] ILIKE '%esri/Google_Africa_Buildings%'
                OR element_at(tags, 'source')[1] ILIKE '%esri/Google_Open_Buildings%'
                OR len(list_filter(hashtags, x -> x ILIKE '%mapwithai%')) > 0
                OR element_at(changeset_tags, 'source')[1] ILIKE '%microsoft/BuildingFootprints%'
                OR element_at(changeset_tags, 'source')[1] ILIKE '%esri/Google_Open_Buildings%'
                OR element_at(changeset_tags, 'source')[1] ILIKE '%mapwithai%'
            ) AS is_AI
        FROM db
    )
    SELECT
        STRFTIME(valid_from, '%Y-%m') as months,
        YEAR(valid_from) AS year,

        COUNT(*) AS edits,
        COUNT(DISTINCT user_id) AS contributors,
        COUNT(DISTINCT changeset_id) AS changesets,

        COUNT(CASE WHEN is_corporate THEN 1 END) AS edits_corporate,
        COUNT(DISTINCT CASE WHEN is_corporate THEN user_id END) AS contributors_corporate,
        COUNT(DISTINCT CASE WHEN is_corporate THEN changeset_id END) AS changesets_corporate,

        COUNT(CASE WHEN is_humanitarian THEN 1 END) AS edits_humanitarian,
        COUNT(DISTINCT CASE WHEN is_humanitarian THEN user_id END) AS contributors_humanitarian,
        COUNT(DISTINCT CASE WHEN is_humanitarian THEN changeset_id END) AS changesets_humanitarian,

        COUNT(CASE WHEN is_AI THEN 1 END) AS edits_AI,
        COUNT(DISTINCT CASE WHEN is_AI THEN user_id END) AS contributors_AI,
        COUNT(DISTINCT CASE WHEN is_AI THEN changeset_id END) AS changesets_AI

    FROM flagged
    GROUP BY months, year
    ORDER BY months
    """, params={"corporate": corporate_pattern, "humanitarian": humanitarian_pattern}).df()

    return affiliation_all

def affiliation_hot(con: duckdb.DuckDBPyConnection, corporate_pattern: str, humanitarian_pattern: str) -> pd.DataFrame:
    affiliation_humanitarian = con.sql(f"""
        WITH flagged AS (
            SELECT
                valid_from,
                user_id,
                changeset_id,
               (regexp_matches(array_to_string(hashtags, ' '), $corporate, 'i') AND NOT regexp_matches(array_to_string(hashtags, ' '), $humanitarian, 'i')) AS is_corporate,
                (
                    REGEXP_EXTRACT(editor, '^([A-Za-z]+)', 1) ILIKE 'Rapid'
                    OR element_at(tags, 'source')[1] ILIKE '%microsoft/BuildingFootprints%'
                    OR element_at(tags, 'source')[1] ILIKE '%esri/Google_Africa_Buildings%'
                    OR element_at(tags, 'source')[1] ILIKE '%esri/Google_Open_Buildings%'
                    OR len(list_filter(hashtags, x -> x ILIKE '%mapwithai%')) > 0
                    OR element_at(changeset_tags, 'source')[1] ILIKE '%microsoft/BuildingFootprints%'
                    OR element_at(changeset_tags, 'source')[1] ILIKE '%esri/Google_Open_Buildings%'
                    OR element_at(changeset_tags, 'source')[1] ILIKE '%mapwithai%'
                ) AS is_AI
            FROM db
            WHERE regexp_matches(array_to_string(hashtags, ' '), $humanitarian, 'i') 
            )
            SELECT
                STRFTIME(valid_from, '%Y-%m') as months,
                YEAR(valid_from) AS year,
        
                COUNT(*) AS edits,
                COUNT(DISTINCT user_id) AS contributors,
                COUNT(DISTINCT changeset_id) AS changesets,
        
                COUNT(CASE WHEN is_corporate THEN 1 END) AS edits_corporate,
                COUNT(DISTINCT CASE WHEN is_corporate THEN user_id END) AS contributors_corporate,
                COUNT(DISTINCT CASE WHEN is_corporate THEN changeset_id END) AS changesets_corporate,
        
                COUNT(CASE WHEN is_AI THEN 1 END) AS edits_AI,
                COUNT(DISTINCT CASE WHEN is_AI THEN user_id END) AS contributors_AI,
                COUNT(DISTINCT CASE WHEN is_AI THEN changeset_id END) AS changesets_AI
    
        FROM flagged
        GROUP BY months, year
        ORDER BY months
        """, params={"corporate": corporate_pattern, "humanitarian": humanitarian_pattern}).df()
    
    return affiliation_humanitarian

def affiliation_ai(con: duckdb.DuckDBPyConnection, corporate_pattern: str, humanitarian_pattern: str) -> pd.DataFrame:
    affiliation_ai = con.sql(f"""
    WITH flagged AS (
        SELECT
            valid_from,
            user_id,
            changeset_id,
           (regexp_matches(array_to_string(hashtags, ' '), $corporate, 'i') AND NOT regexp_matches(array_to_string(hashtags, ' '), $humanitarian, 'i')) AS is_corporate,
            regexp_matches(array_to_string(hashtags, ' '), $humanitarian, 'i') AS is_humanitarian,
        FROM db_ai
            )
    SELECT
        STRFTIME(valid_from, '%Y-%m') as months,
        YEAR(valid_from) AS year,

        COUNT(*) AS edits,
        COUNT(DISTINCT user_id) AS contributors,
        COUNT(DISTINCT changeset_id) AS changesets,

        COUNT(CASE WHEN is_corporate THEN 1 END) AS edits_corporate,
        COUNT(DISTINCT CASE WHEN is_corporate THEN user_id END) AS contributors_corporate,
        COUNT(DISTINCT CASE WHEN is_corporate THEN changeset_id END) AS changesets_corporate,

        COUNT(CASE WHEN is_humanitarian THEN 1 END) AS edits_humanitarian,
        COUNT(DISTINCT CASE WHEN is_humanitarian THEN user_id END) AS contributors_humanitarian,
        COUNT(DISTINCT CASE WHEN is_humanitarian THEN changeset_id END) AS changesets_humanitarian,

    FROM flagged
    GROUP BY months, year
    ORDER BY months
    """, params={"corporate": corporate_pattern, "humanitarian": humanitarian_pattern}).df()

    return affiliation_ai


def affiliation_all_plot(df: pd.DataFrame, country: str) -> plt.Figure:
    #df = affiliation_all(con)
    df =df.copy()
    #df['months'] = df['months'].dt.strftime('%Y-%m')
    
    fig, ax = plt.subplots(figsize=(10, 5))

    df.set_index('months')['changesets'].plot(
        kind='bar',
        color='#9CC4B2',
        width=1.0,
        edgecolor='none',
        zorder=1, 
        ax = ax
    )   
    df.set_index('months')[["changesets_corporate", "changesets_humanitarian", "changesets_AI"]].plot(
        kind='bar',
        stacked=True,
        color=['#304C89', '#9893DA', "#82735C"],
        width=1.0,
        edgecolor='none',
        zorder=2, 
        ax = ax    ) 
       
    ax.set_ylabel('Changeset')
    ax.set_xlabel('Months')
    ax.legend(['Changesets (total)', 'Corporate', 'Humanitarian', 'AI']) 

    ticks = ax.get_xticks()
    labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
    ax.set_xticks(ticks[::4])
    ax.set_xticklabels(labels[::4], rotation=90)

    ax.set_title(f"Affiliation of all Changesets in {country}", fontsize=10, fontweight="bold")    
    ax.grid(linewidth=0.2, color = "gray")

    ax.set_ylim(0,)

    fig.tight_layout()
    return fig


# Retention

def retention(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    monthly_contribs = con.sql("""
        SELECT
        distinct user_name,
        STRFTIME(valid_from, '%Y-%m') as months,
        YEAR(valid_from) as year,
        FROM db
        """).df()

    monthly_contribs['months'] = pd.to_datetime(monthly_contribs['months']).dt.to_period('M')    
    active = monthly_contribs[['user_name', 'months']].drop_duplicates()
    user_month_counts = active.groupby('user_name')['months'].count()
    one_time_users = user_month_counts[user_month_counts == 1].index #getting users that only contributed once
    active['is_one_time'] = active['user_name'].isin(one_time_users)
    next_month = active[['user_name', 'months']].copy() 
    next_month['months'] = next_month['months'] - 1 
    retained = active.merge( #merging with the next_month table to see if the users stays from one month to the next
        next_month,
        on=['user_name', 'months'],
        how='left',
        indicator=True
    )
    retained['returned'] = retained['_merge'] == 'both'    
    monthly = (
        retained
        .groupby('months')
        .agg(
            active_users=('user_name', 'count'),
            returning_users=('returned', 'sum'),
            one_time_users=('is_one_time', 'sum')  
        )
        .reset_index() )    
    monthly['retention_rate'] = monthly['returning_users'] / monthly['active_users']


    #add retention of contributors using AI
    monthly_contribs_ai = con.sql("""
        SELECT
        distinct user_name,
        STRFTIME(valid_from, '%Y-%m') as months,
        YEAR(valid_from) as year,
        FROM db_ai
        """).df()
    
    monthly_contribs_ai['months'] = pd.to_datetime(monthly_contribs_ai['months']).dt.to_period('M')
    active_ai = monthly_contribs_ai[['user_name', 'months']].drop_duplicates()
    user_month_counts_ai = active_ai.groupby('user_name')['months'].count()
    one_time_users_ai = user_month_counts_ai[user_month_counts_ai == 1].index #getting users that only contributed once
    active_ai['is_one_time'] = active_ai['user_name'].isin(one_time_users_ai)
    next_month_ai = active_ai[['user_name', 'months']].copy() 
    next_month_ai['months'] = next_month_ai['months'] - 1 
    retained_ai = active_ai.merge( #merging with the next_month table to see if the users stays from one month to the next
        next_month_ai,
        on=['user_name', 'months'],
        how='left',
        indicator=True
    )
    retained_ai['returned'] = retained_ai['_merge'] == 'both'  
    monthly_ai = (
        retained_ai
        .groupby('months')
        .agg(
            active_users_ai=('user_name', 'count'),
            returning_users_ai=('returned', 'sum'),
            one_time_users_ai=('is_one_time', 'sum')  
        )
        .reset_index())
    monthly_ai['retention_rate_ai'] = monthly_ai['returning_users_ai'] / monthly_ai['active_users_ai']

    combi = monthly.merge(monthly_ai, on="months", how="left").fillna(0)
    combi["ratio_active_users_ai"]=combi["active_users_ai"] / combi["active_users"]
    combi['months'] = combi['months'].astype(str)

    return combi

def retention_plot(df: pd.DataFrame, country: str) -> plt.Figure:
    df = df.copy() #retention(con)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax1 = ax[0]
    # total active_users bar (background)
    df.set_index('months')['active_users'].plot(
        kind='bar',
        color='#9CC4B2',
        width=1.0,
        edgecolor='none',
        zorder=1, 
        ax = ax1
    )   
    # returning + one_time stacked on top (the breakdown of active_users)
    df.set_index('months')[['one_time_users', 'returning_users']].plot(
        kind='bar',
        stacked=True,
        color=['#304C89', '#9893DA'],
        width=1.0,
        edgecolor='none',
        zorder=2, 
        ax = ax1    ) 
    ax1.set_ylabel('Contributor')
    ax1.set_xlabel('Months')
    ax1.legend(['active users (total)', 'one-time users', 'returning users']) 
    ticks = ax1.get_xticks()
    labels = [lbl.get_text() for lbl in ax1.get_xticklabels()]
    ax1.set_xticks(ticks[::4])
    ax1.set_xticklabels(labels[::4], rotation=90)
    ax1.grid(linewidth=0.2, color = "gray")
    ax1.set_title(f"Retention of Contributors in {country}", fontsize=10, fontweight="bold")  

    ax2 = ax[1]
    # total active_users bar (background)
    df.set_index('months')['active_users_ai'].plot(
        kind='bar',
        color='#9CC4B2',
        width=1.0,
        edgecolor='none',
        zorder=1, 
        ax = ax2)   
    # returning + one_time stacked on top (the breakdown of active_users)
    df.set_index('months')[['one_time_users_ai', 'returning_users_ai']].plot(
        kind='bar',
        stacked=True,
        color=['#304C89', '#9893DA'],
        width=1.0,
        edgecolor='none',
        zorder=2, 
        ax = ax2) 
    ax2.sharey(ax1)
    ax2.set_ylabel('Contributor')
    ax2.set_xlabel('Months')
    ax2.legend(['active users (total)', 'one-time users', 'returning users'])   
    ticks = ax2.get_xticks()
    labels = [lbl.get_text() for lbl in ax2.get_xticklabels()]
    ax2.set_xticks(ticks[::4])
    ax2.set_xticklabels(labels[::4], rotation=90)
    ax2.grid(linewidth=0.2, color = "gray")
    ax2.set_title(f"Retention of Contributors using AI in {country}", fontsize=10, fontweight="bold")  
    
    return fig


#  Number of (days) between first and last Edit -> Survival Period

def survival(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df_first_edit = con.sql("""
        WITH user_first_changeset AS (
            SELECT
                valid_from,
                user_name,
                user_id,
                changeset_id,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY valid_from, changeset_id) as rn
            FROM db
        )
        SELECT
            user_id, user_name,
            changeset_id as changeset_first,
            DATE_TRUNC('month', valid_from) as first_edit,
        FROM user_first_changeset
        WHERE rn = 1 
        ORDER BY first_edit, user_id
    """).df()

    df_last_edit = con.sql("""
        WITH user_last_changeset AS (
            SELECT
                valid_from,
                user_id, user_name,
                changeset_id,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY valid_from DESC) as rn
            FROM (
            SELECT DISTINCT user_id,user_name, valid_from, changeset_id,
            FROM db
            )
        )
        SELECT
            user_id, user_name,
            changeset_id as changeset_last,
            DATE_TRUNC('month', valid_from) as last_edit,
        FROM user_last_changeset
        WHERE rn = 1 
        ORDER BY last_edit, user_id
    """).df()

    #list of contributors and if they have or have not used ai
    uses_ai_df = con.sql("""
        SELECT DISTINCT user_id, user_name,
        CASE
            WHEN (REGEXP_EXTRACT(editor, '^([A-Za-z]+)', 1) ILIKE 'Rapid'
            OR element_at(tags, 'source')[1] ILIKE '%microsoft/BuildingFootprints%'
            OR element_at(tags, 'source')[1] ILIKE '%esri/Google_Africa_Buildings%'
            OR element_at(tags, 'source')[1] ILIKE '%esri/Google_Open_Buildings%'
            OR list_contains(hashtags, 'mapwithai')
            OR element_at(changeset_tags, 'source')[1] ILIKE '%microsoft/BuildingFootprints%'
            OR element_at(changeset_tags, 'source')[1] ILIKE '%esri/Google_Open_Buildings%'
            OR element_at(changeset_tags, 'source')[1] ILIKE '%mapwithai%'
            )THEN 'ai_use'
            ELSE 'no_ai_use'
            END AS ai_usage,
        FROM db  
    """).df()

    first_last = df_first_edit.merge(df_last_edit, on =("user_id", "user_name"), how="left")
    first_last = first_last.merge(uses_ai_df, on =("user_id", "user_name"), how="left")
    first_last['days_active'] = ((first_last["last_edit"] -first_last['first_edit'])/np.timedelta64(1, 'D'))
    first_last.loc[first_last['days_active'] == 0, 'days_active'] = 1
    average_days_per_month = 30.44
    first_last = first_last.fillna(0)
    first_last['months_active'] = (first_last['days_active'] / average_days_per_month).astype(int)

    return first_last
    

def survival_plot(df: pd.DataFrame, country: str) -> plt.Figure:
    df = df.copy() #survival(con)
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data = df,
        x = "months_active",
        hue= 'ai_usage',
        multiple="stack",  
        palette=['#6A8D73', '#304C89'],
        edgecolor='white',
        linewidth=0.3,
        ax=ax,
    )
    ax.set_yscale('log')
    ax.set_xlim(0,)
    ax.set_xlabel('Active Duration (months)', fontsize=8)
    ax.set_ylabel('Count (log scale)', fontsize=8)
    ax.legend(fontsize=9, labels = ["AI-use", "No AI-use"])
    ax.set_title(f'OSM Contributors Months active until 2026 in {country}', fontsize=11, fontweight='bold', pad=14)

    fig.tight_layout()
    return fig


    
# ---------------------------------------------------------------------------
#
# Community Indicators
#    
# ---------------------------------------------------------------------------

#  Interaction

#monthly Interaction network 
def get_monthly_edgelist(con, month: str) -> pd.DataFrame:
    df = con.sql(f"""
        WITH edits AS (
            SELECT DISTINCT
                user_id,
                osm_type || '/' || CAST(osm_id AS VARCHAR) AS element_id
            FROM db
            WHERE user_id IS NOT NULL
              AND STRFTIME(valid_from, '%Y-%m') = '{month}'
        )
        SELECT
            a.user_id AS user_a,
            b.user_id AS user_b,
            COUNT(*)  AS weight
        FROM edits a
        JOIN edits b
          ON  a.element_id = b.element_id
          AND a.user_id < b.user_id
        GROUP BY a.user_id, b.user_id
    """).df()
    if not df.empty:
        assert (df["weight"] > 0).all(), f"{month}: non-positive weights found"
        assert (df["user_a"] != df["user_b"]).all(), f"{month}: self-loops found"
    return df


def get_monthly_degree(edge_df: pd.DataFrame, month: str) -> pd.DataFrame:
    dupes = edge_df.duplicated(subset=["user_a", "user_b"]).sum()
    if dupes:
        raise ValueError(f"{month}: {dupes} duplicate user pairs in edge_df")

    G = nx.from_pandas_edgelist(edge_df, source="user_a", target="user_b")
    degree_df = pd.DataFrame({
        "months": month,
        "user_id": list(G.nodes()),
        "degree": [d for _, d in G.degree(weight="weight")],
    })
    # every user in the edgelist should show up as a node
    expected_users = set(edge_df["user_a"]) | set(edge_df["user_b"])
    assert set(degree_df["user_id"]) == expected_users, f"{month}: node/user mismatch"
    return degree_df

def monthly_interaction_network(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    # Check table exists and has expected columns
    required_cols = {"user_id", "osm_type", "osm_id", "valid_from"}
    actual_cols = set(con.sql("SELECT * FROM db LIMIT 0").df().columns)
    missing = required_cols - actual_cols
    if missing:
        raise ValueError(f"db table missing required columns: {missing}")

    # Sanity check the date range you're about to loop over actually matches the data
    date_bounds = con.sql("SELECT MIN(valid_from), MAX(valid_from) FROM db").fetchone()
    print(f"Data spans {date_bounds[0]} to {date_bounds[1]}")
    
    months = pd.period_range("2020-01", "2026-05", freq="M").strftime("%Y-%m")
    all_metrics = []
    skipped_months = []

    for month in months:
        try:
            edge_df = get_monthly_edgelist(con, month)
        except Exception as e:
            print(f"ERROR processing {month}: {e}")
            skipped_months.append(month)
            continue

        if edge_df.empty:
            skipped_months.append(month)
            continue

        metrics = get_monthly_degree(edge_df, month)
        all_metrics.append(metrics)

    if not all_metrics:
        raise RuntimeError("No months produced data — check date range/table contents")

    if skipped_months:
        print(f"Skipped {len(skipped_months)} months (no data or error): {skipped_months}")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary = (
        metrics_df
        .groupby("months")["degree"]
        .agg(
            n_contributors="count",
            mean_degree="mean",
            median_degree="median",
            max_degree="max",
        )
        .reset_index()
        .round(2)
    )
    #print(summary)
    return metrics_df, summary

# Comments and Notes Size 

def comments_notes(con: duckdb.DuckDBPyConnection, countries: gpd.GeoDataFrame,
                    country: str) -> pd.DataFrame:
    comments = con.sql("""
        SELECT 
            STRFTIME(valid_from, '%Y-%m') as months,
            YEAR(valid_from) as year,
            db.user_name,
            db.changeset_id,
            c.text,
            LENGTH(c.text) as changeset_comments_total,
            CASE
                WHEN (
                REGEXP_EXTRACT(editor, '^([A-Za-z]+)', 1) ILIKE 'Rapid'
                OR element_at(tags, 'source')[1] ILIKE '%microsoft/BuildingFootprints%'
                OR element_at(tags, 'source')[1] ILIKE '%esri/Google_Africa_Buildings%'
                OR element_at(tags, 'source')[1] ILIKE '%esri/Google_Open_Buildings%'
                OR len(list_filter(hashtags, x -> x ILIKE '%mapwithai%')) > 0
                OR element_at(changeset_tags, 'source')[1] ILIKE '%microsoft/BuildingFootprints%'
                OR element_at(changeset_tags, 'source')[1] ILIKE '%esri/Google_Open_Buildings%'
                OR element_at(changeset_tags, 'source')[1] ILIKE '%mapwithai%') THEN 'changeset_comments_ai'
                ELSE 'changeset_comments_nonai'
            END AS changesets_AI_use
        FROM db
        LEFT JOIN cc c ON db.user_name = c.user_name AND db.changeset_id = c.changeset_id
        ORDER BY valid_from DESC
    """).df()
    #comments['months'] = pd.to_datetime(comments['months']).dt.to_period('M')
    comments[["changeset_comments_total"]] = comments[["changeset_comments_total"]].fillna(0)
    monthly_comment_avg = comments[["months", "changeset_comments_total"]].groupby("months").mean()
    monthly_comment_avg_ai = comments[["months", "changeset_comments_total", "changesets_AI_use"]].groupby(["months","changesets_AI_use"]).mean().reset_index()
    combi = monthly_comment_avg_ai.pivot_table(
        index='months',
        columns='changesets_AI_use',
        values='changeset_comments_total',
        aggfunc='sum'
    ).reset_index()
    monthly_comment_avg_combi = combi.merge(monthly_comment_avg, on="months", how = "right")
    monthly_comment_avg_combi = monthly_comment_avg_combi.rename(columns={"comment_length": "total"}).fillna(0)

    notes_loc = con.sql("""
        SELECT 
            STRFTIME(created_at, '%Y-%m') as months,
            nd.note_id,
            lat,
            lon,
            closed_at,
            nc.user_name,
            nc.text,
            LENGTH(nc.text) as notes_length
        FROM nd
        LEFT JOIN nc ON nd.note_id = nc.note_id
        WHERE created_at >= '2020-01-01'
    """).df()
    notes_loc_gdf = gpd.GeoDataFrame(notes_loc, geometry=gpd.points_from_xy(notes_loc.lon, notes_loc.lat),crs="EPSG:4326")
    
    if country == "US1":
        selected = countries[["iso3cd", "geometry"]].loc[countries["iso3cd"]=="USA"]
    elif country =="NZ1":
        selected = countries[["iso3cd", "geometry"]].loc[countries["iso3cd"]=="NZL"]
    elif country =="B35":
        selected = countries[["iso3cd", "geometry"]].loc[countries["iso3cd"]=="GEO"]
    else: selected = countries[["iso3cd", "geometry"]].loc[countries["iso3cd"]==f"{country}"]

    selected_notes = gpd.sjoin(notes_loc_gdf, selected, how="inner", predicate="within")   
    #selected_notes['months'] = pd.to_datetime(selected_notes['created_at']).dt.to_period('M')
    selected_notes[["notes_length"]] = selected_notes[["notes_length"]].fillna(0)
    monthly_notes_avg = selected_notes[["months", "notes_length"]].groupby("months").mean()

    average_length = monthly_comment_avg_combi.merge(monthly_notes_avg, on = "months").fillna(0)

    #print(average_length.mean(axis=0))
    return average_length


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------

def save_report(figures: list, output_path: str) -> None:
    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    logger.info(f"PDF report saved to {output_path}")


# ---------------------------------------------------------------------------
# Prompts 
# ---------------------------------------------------------------------------

def prompt_path(label: str, must_exist: bool = True) -> str:
    while True:
        p = input(f"{label}: ").strip()
        if not p:
            print("  ✗ Path cannot be empty, try again.")
            continue
        if must_exist and not Path(p).exists():
            print(f"  ✗ File not found: {p}")
            continue
        return p

def prompt_inputs() -> dict:
    print("\n=== Country Indicators Data Extraction ===\n")
    country     = input("Country code (e.g. ALB): ").strip()
    data        = prompt_path("Path to main .parquet files")
    return dict(country=country, data=data)



def main():
    inputs = prompt_inputs()
    country = inputs["country"]

    # 1. load
    con, countries = load_data(inputs["data"], country)

    corporate_pattern = build_pattern(corporate_hashtags)
    humanitarian_pattern = build_pattern(humanitarian_hashtags)

     # 3. save CSVs — save og after each calculation
    country_csv_dir = FILEPATH_CSV_RESULTS / country
    country_csv_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. compute 
    logger.info("Computing monthly stats")
    monthly_df = monthly_stats(con)
    if not monthly_df.empty:
        monthly_df.to_csv(country_csv_dir / f"{country}_monthly_stats.csv", index=False)
        logger.info("Saved monthly_df csv")
    else:
        logger.warning(f"{country}: no data")
        
        

    logger.info("Computing contribution type and feature type")
    contrib_type_df, contrib_type_df_long = contrib_type(con)
    feature_type_df, feature_type_df_long = feature_type(con)
    contrib_type_df.to_csv(country_csv_dir / f"{country}_contrib_type.csv", index=False)
    feature_type_df.to_csv(country_csv_dir / f"{country}_feature_type.csv", index=False)
    logger.info("Saved contrib_type_df and feature_type_df csv")

    logger.info("Computing Contributor Experience Time Series")
    xp_monthly = ohsome_contrib_xp(country)   
    xp_monthly.to_csv(country_csv_dir / f"{country}_updated_monthly_experience_ohsome.csv", index=False)
    logger.info("Saved xp_monthly csv")

    logger.info("Computing Affiliation of Edits")
    affiliation_all_df = affiliation_all(con, corporate_pattern, humanitarian_pattern)
    affiliation_hot_df = affiliation_hot(con, corporate_pattern, humanitarian_pattern)
    affiliation_ai_df = affiliation_ai(con, corporate_pattern, humanitarian_pattern)
    affiliation_all_df.to_csv(country_csv_dir / f"{country}_affiliation_all.csv", index=False)
    affiliation_hot_df.to_csv(country_csv_dir / f"{country}_affiliation_hot.csv", index=False)
    affiliation_ai_df.to_csv(country_csv_dir / f"{country}_affiliation_ai.csv", index=False)
    logger.info("Saved affiliation_ai_df, affiliation_hot_df and affiliation_all_df csv")

    logger.info("Computing Contributor Retention and Survival")
    retention_df = retention(con)  
    survival_df = survival(con)
    retention_df.to_csv(country_csv_dir / f"{country}_retention.csv", index=False)
    survival_df.to_csv(country_csv_dir / f"{country}_survival.csv", index=False)
    logger.info("Saved retention_df, survival_df csv")
    
    logger.info("Computing Interaction Network")
    #adding some safeguards to the interaction network
    try:
        monthly_interaction_network_df_short, monthly_interaction_network_df = monthly_interaction_network(con)
    except RuntimeError as e:
        logger.warning(f"Skipping interaction network for {country}: {e}")
        monthly_interaction_network_df_short = pd.DataFrame()
        monthly_interaction_network_df = pd.DataFrame()

    if not monthly_interaction_network_df.empty:
        monthly_interaction_network_df.to_csv(country_csv_dir / f"{country}_monthly_interaction_network.csv", index=False)
    else:
        logger.warning(f"{country}: interaction network CSV not saved — no data")

    logger.info("Saved monthly_interaction_network_df csv")

    logger.info("Computing Notes Length")
    comments_notes_df = comments_notes(con, countries, country)
    comments_notes_df.to_csv(country_csv_dir / f"{country}_comments_notes_total.csv", index=False)
    logger.info("Saved comments_notes_df csv")

    #creating one big csv with all monthly indicators
    big_monthly_df = affiliation_all_df.merge(monthly_df[["months", "editsPerc", "changesetsPerc", "contributorsPerc", "global_new_contrib", "local_new_contrib", "new_contrib_ai", "new_contrib_non_ai"]], on ="months")
    big_monthly_df = big_monthly_df.merge(xp_monthly, on = "months")
    big_monthly_df = big_monthly_df.merge(comments_notes_df, on = "months")
    big_monthly_df = big_monthly_df.merge(contrib_type_df_long, on = "months")
    big_monthly_df = big_monthly_df.merge(feature_type_df_long, on = "months")
    big_monthly_df = big_monthly_df.merge(retention_df, on = "months")
    big_monthly_df = big_monthly_df.merge(monthly_interaction_network_df[["months", "mean_degree"]], on = "months")
    #calculating massing ratios / shares
    big_monthly_df["changesets_corporate_share"] =big_monthly_df["changesets_corporate"] / big_monthly_df["changesets"]
    big_monthly_df["contributors_corporate_share"] =big_monthly_df["contributors_corporate"] / big_monthly_df["contributors"]
    
    big_monthly_df["changesets_humanitarian_share"] =big_monthly_df["changesets_humanitarian"] / big_monthly_df["changesets"]
    big_monthly_df["contributors_humanitarian_share"] =big_monthly_df["contributors_humanitarian"] / big_monthly_df["contributors"]
    
    big_monthly_df["changesets_ai_share"]= big_monthly_df["changesetsPerc"]/100
    big_monthly_df["contributors_ai_share"]= big_monthly_df["contributorsPerc"]/100
    big_monthly_df["new_contributors_ai_share"]= big_monthly_df["new_contrib_ai"] / (big_monthly_df["global_new_contrib"] + big_monthly_df["local_new_contrib"])
    
    big_monthly_df["contributors_prolific_share_ai_total"]= big_monthly_df["contributors_prolific_ai"]/big_monthly_df["contributors"]
    big_monthly_df["contributors_casual_share_ai_total"]= big_monthly_df["contributors_casual_ai"]/big_monthly_df["contributors"]
    big_monthly_df["contributors_prolific_share_total"]= big_monthly_df["contributors_prolific_total"]/big_monthly_df["contributors"]
    big_monthly_df["contributors_casual_share_total"]= big_monthly_df["contributors_casual_total"]/big_monthly_df["contributors"]
    
    big_monthly_df["contributors_inactive_share_total"]= big_monthly_df["contributors_inactive_total"]/big_monthly_df["contributors"]
    big_monthly_df["contributors_inactive_share_ai_total"]= big_monthly_df["contributors_inactive_ai"]/big_monthly_df["contributors"]

    big_monthly_df["global_new_contrib_share"]= big_monthly_df["global_new_contrib"]/big_monthly_df["contributors"]
    big_monthly_df["building_changesets_share"] = big_monthly_df["changesets_building"] / big_monthly_df["changesets"]
    big_monthly_df["highway_changesets_share"] = big_monthly_df["changesets_highway"] / big_monthly_df["changesets"]
    big_monthly_df["building_changesets_ai_share"] = big_monthly_df["changesets_ai_building"] / big_monthly_df["changesets"]
    big_monthly_df["highway_changesets_ai_share"] = big_monthly_df["changesets_ai_highway"] / big_monthly_df["changesets"]

    #added from Spearman 
    big_monthly_df["total_new_contrib"] = big_monthly_df["global_new_contrib"] + big_monthly_df["local_new_contrib"]
    big_monthly_df["edits_corporate_share"] =big_monthly_df["edits_corporate"] / big_monthly_df["edits"]
    big_monthly_df["edits_humanitarian_share"] =big_monthly_df["edits_humanitarian"] / big_monthly_df["edits"]
    big_monthly_df["edits_ai_share"]= big_monthly_df["editsPerc"]/100
    big_monthly_df["edits_building_share"]= big_monthly_df["share_edits_building"]/100
    big_monthly_df["edits_other_share"]= big_monthly_df["share_edits_other"]/100
    big_monthly_df["edits_highway_share"]= big_monthly_df["share_edits_highway"]/100
    
    #new columns
    big_monthly_df["contributors_inactive_share_ai"]= big_monthly_df["contributors_inactive_ai"]/big_monthly_df["contributors_AI"]
    big_monthly_df["contributors_prolific_share_ai"]= big_monthly_df["contributors_prolific_ai"]/big_monthly_df["contributors_AI"]
    big_monthly_df["contributors_casual_share_ai"]= big_monthly_df["contributors_casual_ai"]/big_monthly_df["contributors_AI"]
    
    indicators_df = big_monthly_df[['months', 'contributors', 'changesets', 'edits','mean_degree', 'active_users', 'total_new_contrib' , 
        'changeset_comments_ai', 'ratio_active_users_ai',
        'changesets_corporate_share', 'contributors_corporate_share', 'changesets_humanitarian_share', 'contributors_humanitarian_share', 
        'edits_corporate_share', 'edits_humanitarian_share', 'edits_ai_share', 
        'changesets_ai_share', 'contributors_ai_share', 'contributors_prolific_share_ai', 'contributors_casual_share_ai','contributors_inactive_share_ai',
        'building_changesets_share', 'highway_changesets_share', 'building_changesets_ai_share', 'highway_changesets_ai_share',
        'new_contributors_ai_share', 'notes_length',
        'contributors_prolific_share_ai_total', 'contributors_casual_share_ai_total', 'contributors_inactive_share_ai_total',
        ]]
    
    
    indicators_df.to_csv(country_csv_dir / f"{country}_prepped_indicators_stat.csv", index=False)
    big_monthly_df.to_csv(country_csv_dir / f"{country}_all_monthly_metrics.csv", index=False)
    
    logger.info("indicators_df and big_monthly_df successfully saved")

    # 4. plot — one append per figure, add new plot_* calls here
    figures = {
        "overview":        plot_overview(monthly_df, country),
        "contrib_feature":  plot_contrib_feature(contrib_type_df, feature_type_df, country),
        "contrib_xp":       plot_ohsome_contrib_xp(xp_monthly, country),
        "affiliation_all":  affiliation_all_plot(affiliation_all_df, country),
        "retention":        retention_plot(retention_df, country),
        "survival":         survival_plot(survival_df, country),
    }

        
    #6. save PDF
    pdf_path = FILEPATH_REPORT_RESULTS/f"{country}_plots_overview.pdf"
    save_report(list(figures.values()), str(pdf_path))
    logger.info(f"Pdf report saved to {pdf_path}")

    # 6. close
    con.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()