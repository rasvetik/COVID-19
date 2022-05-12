"""
Created on September 2021
Sveta Raboy & Doron Bar

"""
# ####################################################################
# Use setup.py file to install the all necessary requirements.
# ####################################################################


# 1 World status
######################################################################################################################
# Run world_status.py file to download the covid-19 database, create the csv clean files and process the data.
# The results are saved to a folder with the current date and consist of *.html and *.png files.
# FYI: from May 2021 the Recovered values in all countries are 0
try:
    exec(open('world_status.py').read())
except:
    exec(open('world_status.py').read())
######################################################################################################################

# 2 Insufficiency status
######################################################################################################################
try:
    exec(open('insufficiency_status.py').read())
except:
    exec(open('insufficiency_status.py').read())
######################################################################################################################

# 3 Subplots: 1,2) n-days-estimators, 3) infected & confirmed, 4) recovered & death
######################################################################################################################
try:
    exec(open('countriesPlots2article2.py').read())
except:
    exec(open('countriesPlots2article2.py').read())
######################################################################################################################

# 4 Israel status
######################################################################################################################
try:
    exec(open('israel_status.py', encoding='utf-8').read())
except:
    exec(open('israel_status.py', encoding='utf-8').read())
######################################################################################################################

# 5 R-factor vs nDay Estimator
######################################################################################################################
try:
    exec(open('israel_COVID19_R_Factor_vs_nDayEstimator.py').read())
except:
    exec(open('israel_COVID19_R_Factor_vs_nDayEstimator.py').read())
######################################################################################################################
