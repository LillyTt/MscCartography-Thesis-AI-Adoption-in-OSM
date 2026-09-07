# MscCartography-Thesis-AI-Adoption-in-OSM
Repository for Code and Figures for the Thesis "Measuring the Impact of AI Adoption on the Sustainability of OpenStreetMap" 

# Data Sources

## Global Analysis
For the global level analysis, the prepared changeset data files (parquet) from https://github.com/piebro/openstreetmap-statistics were used. 
Downloaded on 11.05.2025 under ODbL liscense. 

A dataset of all OSM contributors, created and provided from HeiGIT’s OSHDB data infrastructure was used as well. Extraction query in code folder

## Country Analysis

**Country level parquet files**

The country level parquet files were created using the ohsome planet tools (https://github.com/GIScience/ohsome-planet) and enriched with the OSM changesetdataset (following the tutorial on ohsome-planet github)
source pbf files from Geofabrik (https://download.geofabrik.de/index.html) including full metadata 
global changeset dataset from: https://planet.openstreetmap.org/ from 25.05.2026


*Disclaimer: Claude was used to support coding and debugging 
