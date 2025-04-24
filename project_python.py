# import libraries
import pandas as pd
import lightgbm as lgbm
#from sklearn import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# Read in CSV files of prior and post periods
pre_dat = pd.read_csv("nh_archive_01_2018/ProviderInfo_Download.csv", encoding='\latin-1')
post_dat = pd.read_csv("nursing_homes_including_rehab_services_03_2025/NH_ProviderInfo_Mar2025.csv")

pre_dat.drop([
              "RN_staffing_rating"
              ,"rn_staffing_rating_fn"
              ,"exp_aide"
              ,"exp_lpn"
              ,"exp_rn"
              ,"exp_total"
              ],axis=1,inplace=True)

# Encode the float64 footnotes in the new dataset as strings
# List of column names to convert
float_columns = [c for c in post_dat.columns if "FOOTNOTE" in c.upper()]
# Convert specified columns to string
post_dat[float_columns] = post_dat[float_columns].astype(str)

# Create a dictionary mapping old column names to equivalent new ones
rename_dict_ProviderInfo = {
    "provnum":"CMS Certification Number (CCN)"
    ,"PROVNAME":"Provider Name"
    ,"ADDRESS":"Provider Address"
    ,"CITY":"City/Town"
    ,"STATE":"State"
    ,"ZIP":"ZIP Code"
    ,"PHONE":"Telephone Number"
    ,"COUNTY_SSA":"Provider SSA County Code"
    ,"COUNTY_NAME":"County/Parish"
    ,"OWNERSHIP":"Ownership Type"
    ,"BEDCERT":"Number of Certified Beds"
    ,"RESTOT":"Average Number of Residents per Day" ########
    ,"CERTIFICATION":"Provider Type"
    ,"INHOSP":"Provider Resides in Hospital" #########
    ,"LBN":"Legal Business Name"
    ,"PARTICIPATION_DATE":"Date First Approved to Provide Medicare and Medicaid Services"
    ,"CCRC_FACIL":"Continuing Care Retirement Community"
    ,"SFF":"Special Focus Status"
    ,"OLDSURVEY":"Most Recent Health Inspection More Than 2 Years Ago"
    ,"CHOW_LAST_12MOS":"Provider Changed Ownership in Last 12 Months"
    ,"resfamcouncil":"With a Resident and Family Council"
    ,"sprinkler_status":"Automatic Sprinkler Systems in All Required Areas"
    ,"overall_rating":"Overall Rating"
    ,"overall_rating_fn":"Overall Rating Footnote"
    ,"survey_rating":"Health Inspection Rating"
    ,"survey_rating_fn":"Health Inspection Rating Footnote"
    ,"quality_rating": "QM Rating"
    ,"quality_rating_fn": "QM Rating Footnote"
    ,"staffing_rating":"Staffing Rating"
    ,"staffing_rating_fn":"Staffing Rating Footnote"
    #,"RN_staffing_rating":
    #,"rn_staffing_rating_fn"
    ,"STAFFING_FLAG":"Reported Staffing Footnote"
    ,"PT_STAFFING_FLAG":"Physical Therapist Staffing Footnote"
    ,"AIDHRD":"Reported Nurse Aide Staffing Hours per Resident per Day"
    ,"VOCHRD":"Reported LPN Staffing Hours per Resident per Day"
    ,"RNHRD":"Reported RN Staffing Hours per Resident per Day"
    ,"TOTLICHRD":"Reported Licensed Staffing Hours per Resident per Day"
    ,"TOTHRD":"Reported Total Nurse Staffing Hours per Resident per Day"
    ,"PTHRD":"Reported Physical Therapist Staffing Hours per Resident Per Day"
    #,"exp_aide"
    #,"exp_lpn"
    #,"exp_rn"
    #,"exp_total"
    ,"adj_aide":"Adjusted Nurse Aide Staffing Hours per Resident per Day"
    ,"adj_lpn":"Adjusted LPN Staffing Hours per Resident per Day"
    ,"adj_rn":"Adjusted RN Staffing Hours per Resident per Day"
    ,"adj_total":"Adjusted Total Nurse Staffing Hours per Resident per Day"
    
    ,"cycle_1_defs":"Rating Cycle 1 Total Number of Health Deficiencies"
    ,"cycle_1_nfromdefs":"Rating Cycle 1 Number of Standard Health Deficiencies"
    ,"cycle_1_nfromcomp":"Rating Cycle 1 Number of Complaint Health Deficiencies"
    ,"cycle_1_defs_score":"Rating Cycle 1 Health Deficiency Score"
    ,"CYCLE_1_SURVEY_DATE":"Rating Cycle 1 Standard Survey Health Date"
    ,"CYCLE_1_NUMREVIS":"Rating Cycle 1 Number of Health Revisits"
    ,"CYCLE_1_REVISIT_SCORE":"Rating Cycle 1 Health Revisit Score"
    ,"CYCLE_1_TOTAL_SCORE":"Rating Cycle 1 Total Health Score"
    
    ,"cycle_2_defs":"Rating Cycle 2 Total Number of Health Deficiencies"
    ,"cycle_2_nfromdefs":"Rating Cycle 2 Number of Standard Health Deficiencies"
    ,"cycle_2_nfromcomp":"Rating Cycle 2 Number of Complaint Health Deficiencies"
    ,"cycle_2_defs_score":"Rating Cycle 2 Health Deficiency Score"
    ,"CYCLE_2_SURVEY_DATE":"Rating Cycle 2 Standard Health Survey Date"
    ,"CYCLE_2_NUMREVIS":"Rating Cycle 2 Number of Health Revisits"
    ,"CYCLE_2_REVISIT_SCORE":"Rating Cycle 2 Health Revisit Score"
    ,"CYCLE_2_TOTAL_SCORE":"Rating Cycle 2 Total Health Score"
    
    ,"cycle_3_defs":"Rating Cycle 3 Total Number of Health Deficiencies"
    ,"cycle_3_nfromdefs":"Rating Cycle 3 Number of Standard Health Deficiencies"
    ,"cycle_3_nfromcomp":"Rating Cycle 3 Number of Complaint Health Deficiencies"
    ,"cycle_3_defs_score":"Rating Cycle 3 Health Deficiency Score"
    ,"CYCLE_3_SURVEY_DATE":"Rating Cycle 3 Standard Health Survey Date"
    ,"CYCLE_3_NUMREVIS":"Rating Cycle 3 Number of Health Revisits"
    ,"CYCLE_3_REVISIT_SCORE":"Rating Cycle 3 Health Revisit Score"
    ,"CYCLE_3_TOTAL_SCORE":"Rating Cycle 3 Total Health Score"
    
    ,"WEIGHTED_ALL_CYCLES_SCORE":"Total Weighted Health Survey Score"
    ,"incident_cnt":"Number of Facility Reported Incidents"
    ,"cmplnt_cnt":"Number of Substantiated Complaints"
    ,"FINE_CNT":"Number of Fines"
    ,"FINE_TOT":"Total Amount of Fines in Dollars"
    ,"PAYDEN_CNT":"Number of Payment Denials"
    ,"TOT_PENLTY_CNT":"Total Number of Penalties"
    ,"FILEDATE":"Processing Date"
    }


# Rename old columns to the new column names
pre_dat = pre_dat.rename(columns=rename_dict_ProviderInfo)

all_dat = pd.merge(pre_dat,post_dat[pre_dat.columns.values])








