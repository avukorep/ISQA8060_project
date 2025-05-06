# import libraries
import pandas as pd
import lightgbm as lgbm
from lightgbm import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import shap
import csv


# Read in CSV files of prior and post periods
pre_dat = pd.read_csv("nh_archive_01_2018/ProviderInfo_Download.csv", encoding='\latin-1')
post_dat = pd.read_csv("nursing_homes_including_rehab_services_03_2025/NH_ProviderInfo_Mar2025.csv")
post_health_citations = pd.read_csv("nursing_homes_including_rehab_services_03_2025/NH_HealthCitations_Mar2025.csv"
                                    ,dtype={"CMS Certification Number (CCN)":"string"}
                                    )
post_mds_quality_measures = pd.read_csv("nursing_homes_including_rehab_services_03_2025/NH_QualityMsr_MDS_Mar2025.csv"
                                    ,dtype={"CMS Certification Number (CCN)":"string"}
                                    )
post_claims_quality_measures = pd.read_csv("nursing_homes_including_rehab_services_03_2025/NH_QualityMsr_Claims_Mar2025.csv"
                                    ,dtype={"CMS Certification Number (CCN)":"string"}
                                    )
post_ownership = pd.read_csv("nursing_homes_including_rehab_services_03_2025/NH_Ownership_Mar2025.csv"
                                    ,dtype={"CMS Certification Number (CCN)":"string"}
                                    )

pre_dat.drop([
              "RN_staffing_rating"
              ,"rn_staffing_rating_fn"
              ,"exp_aide"
              ,"exp_lpn"
              ,"exp_rn"
              ,"exp_total"
              ],axis=1,inplace=True)

# Set primary key var name
rowkey = "CMS Certification Number (CCN)"

# Set target variable name
response_var = "Lower_Staffing"

# Set staffing target variable
original_var = "Adjusted Total Nurse Staffing Hours per Resident per Day"

# Set lower staffing cutoff:
cutoff = -1.0 # try three levels? lowered, stayed the same, went up?

# Encode the float64 footnotes in the new dataset as strings
# List of column names to convert
float_columns = [c for c in post_dat.columns if "FOOTNOTE" in c.upper()]
# Convert specified columns to string
post_dat[float_columns] = post_dat[float_columns].astype(str)

# Encode the provider ID as text
pre_dat['provnum'] = pre_dat['provnum'].astype(str)
post_dat[rowkey] = post_dat[rowkey].astype(str)
post_health_citations[rowkey] = post_health_citations[rowkey].astype(str)
post_health_citations['Deficiency Tag Number'] = post_health_citations['Deficiency Tag Number'].astype(str)

# Clean special characters out of certain categories
post_health_citations['Deficiency Category'] = post_health_citations['Deficiency Category'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

# Add any prefixes
post_health_citations['Deficiency Tag Number'] = post_health_citations['Deficiency Tag Number'].astype(str).apply(lambda x: 'deficiency_tag_' + x)
post_health_citations['Scope Severity Code'] = post_health_citations['Scope Severity Code'].astype(str).apply(lambda x: 'severity_code_' + x)
post_mds_quality_measures['Measure Code'] = post_mds_quality_measures['Measure Code'].astype(str).apply(lambda x: 'mds_quality_code_' + x)
post_claims_quality_measures['Measure Code'] = post_claims_quality_measures['Measure Code'].astype(str).apply(lambda x: 'claims_quality_code_' + x)

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
pre_dat = pre_dat.rename(columns=lambda x: f"PREV {x}")


# Aggregate health citation data to the facility level
# Function to aggregate a column by the provider code
def agg_column(df, aggvar, rowkey, common_limit=0, value=None, agg='mean', fill_val=0):
    
    # Get counts per category
    val_counts = df[aggvar].value_counts()
    above_limit_vals = val_counts[val_counts > common_limit].index
    
    # Use pivot_table to aggregate values and pivot the categorical column
    if value is None:
        result = pd.pivot_table(df.loc[df[aggvar].isin(above_limit_vals)]
                                , index=rowkey
                                , columns=aggvar
                                , aggfunc='size'
                                , fill_value=fill_val
                                )
    else:
        result = pd.pivot_table(df.loc[df[aggvar].isin(above_limit_vals)]
                                , values=value
                                , index=rowkey
                                , columns=aggvar
                                , aggfunc=agg
                                , fill_value=fill_val
                                )
        
    return result

# Add all pre_dat columns as new columns to the post_dat dataframe
all_dat = pd.merge(post_dat, pre_dat, left_on=rowkey, right_on="PREV "+rowkey, how="inner")
all_dat = pd.merge(all_dat, agg_column(post_health_citations, "Scope Severity Code", rowkey, 0), on=rowkey, how="left")
all_dat = pd.merge(all_dat, agg_column(post_health_citations, "Deficiency Category", rowkey, 0), on=rowkey, how="left")
all_dat = pd.merge(all_dat, agg_column(post_health_citations, "Deficiency Tag Number", rowkey, 1500), on=rowkey, how="left")
all_dat = pd.merge(all_dat, agg_column(post_mds_quality_measures, "Measure Code", rowkey, 0, value="Four Quarter Average Score", agg='mean', fill_val=None), on=rowkey, how="left")
all_dat = pd.merge(all_dat, agg_column(post_claims_quality_measures, "Measure Code", rowkey, 0, value="Adjusted Score", agg='mean', fill_val=None), on=rowkey, how='left')
all_dat = all_dat.reset_index(drop=True)

# Fill NAs with 0
all_dat[[c for c in all_dat.columns if c in post_health_citations['Scope Severity Code'].values]] = all_dat[[c for c in all_dat.columns if c in post_health_citations['Scope Severity Code'].values]].fillna(0)
all_dat[[c for c in all_dat.columns if c in post_health_citations['Deficiency Category'].values]] = all_dat[[c for c in all_dat.columns if c in post_health_citations['Deficiency Category'].values]].fillna(0)
all_dat[[c for c in all_dat.columns if c in post_health_citations['Deficiency Tag Number'].values]] = all_dat[[c for c in all_dat.columns if c in post_health_citations['Deficiency Tag Number'].values]].fillna(0)

# Compare staffing levels
print("Previous staffing level:")
print(np.mean(all_dat['PREV '+original_var]))
print("Current staffing level:")
print(np.mean(all_dat[original_var]))

# Calculate target variable
all_dat[response_var + "_CALC"] = all_dat[original_var] - all_dat["PREV "+original_var]

# Drop nans
all_dat = all_dat.dropna(subset=[response_var+"_CALC"])
all_dat = all_dat.reset_index(drop=True)

# Choose a cutoff to create a 0/1 binary response variable
all_dat[response_var] = np.where(all_dat[f"{response_var}_CALC"] < cutoff, 1, 0)

# Look at imbalance
print("% of facilities with lower staffing:")
print(sum(all_dat[response_var])/len(all_dat) * 100)


# Create variables for model training
def generate_features(df):
    # Ownership Type
    df['ownership_for_profit_corporation'] = np.where(df['Ownership Type']=="For profit - Corporation",1,0)
    df['ownership_for_profit_individual'] = np.where(df['Ownership Type']=="For profit - Individual",1,0)
    df['ownership_for_profit_llc'] = np.where(df['Ownership Type']=="For profit - Limited Liability company",1,0)
    df['ownership_for_profit_partnership'] = np.where(df['Ownership Type']=="For profit - Partnership",1,0)
    df['ownership_non_profit_corporation'] = np.where(df['Ownership Type']=="Non profit - Corporation",1,0)
    df['ownership_non_profit_other'] = np.where(df['Ownership Type']=="Non profit - Other",1,0)
    df['ownership_non_profit_church'] = np.where(df['Ownership Type']=="Non profit - Church related",1,0)
    df['ownership_government_county'] = np.where(df['Ownership Type']=="Government - County",1,0)
    df['ownership_government_city_county'] = np.where(df['Ownership Type']=="Government - City/county",1,0)
    df['ownership_government_federal'] = np.where(df['Ownership Type']=="Government - Federal",1,0)
    df['ownership_government_state'] = np.where(df['Ownership Type']=="Government - State",1,0)
    df['ownership_government_hospital_district'] = np.where(df['Ownership Type']=="Government - Hospital district",1,0)
    df['ownership_government_city'] = np.where(df['Ownership Type']=="Government - City",1,0)
    
    # Provider Type
    df['provider_medicare_and_medicaid'] = np.where(df['Provider Type']=="Medicare and Medicaid",1,0)
    df['provider_medicare'] = np.where(df['Provider Type']=="Medicare",1,0)
    df['provider_medicaid'] = np.where(df['Provider Type']=="Medicaid",1,0)

    # Provider Resides in Hospital
    df['provider_resides_in_hospital'] = np.where(df['Provider Resides in Hospital']=='Y',1,0)
    
    # Months Since Approval
    df['months_since_approval'] = pd.to_datetime(df['Date First Approved to Provide Medicare and Medicaid Services']).apply(lambda x: (datetime(2025,3,31).year - x.year) * 12 + (datetime(2025,3,31).month - x.month))          
    df['months_since_approval_prev'] = pd.to_datetime(df['PREV Date First Approved to Provide Medicare and Medicaid Services']).apply(lambda x: (datetime(2018,1,31).year - x.year) * 12 + (datetime(2018,1,31).month - x.month))          

    # Continuing Care Retirement Community
    df['continuing_care_retirement_community'] = np.where(df['Continuing Care Retirement Community']=='Y',1,0)
    
    # Special Focus Status
    df['special_focus_status_sff_candidate'] = np.where(df['Special Focus Status']=='SFF Candidate',1,0)
    df['special_focus_status_sff'] = np.where(df['Special Focus Status']=='SFF',1,0)
    
    # Abuse Icon
    df['abuse_icon'] = np.where(df['Abuse Icon']=='Y',1,0)
    
    # Most Recent Health Inspection More Than 2 Years Ago
    df['most_recent_health_inspection_2plus_years_ago'] = np.where(df['Most Recent Health Inspection More Than 2 Years Ago']=='Y',1,0)
    
    # Provider Changed Ownership in Last 12 Months
    df['provider_changed_ownership_last_12_months'] = np.where(df['Provider Changed Ownership in Last 12 Months']=='Y',1,0)
    
    # With a Resident and Family Council
    df['council_resident'] = np.where(df['With a Resident and Family Council']=='Resident',1,0)
    df['council_family'] = np.where(df['With a Resident and Family Council']=='Family',1,0)
    df['council_resident_and_family'] = np.where(df['With a Resident and Family Council']=='Both',1,0)
    
    # Ratio of all beds to average residents per day (is the facility always full?) ##### obvious
    df['avg_daily_residents_per_bed_ratio'] = np.where(df['Number of Certified Beds']!=0, df['Average Number of Residents per Day']/df['Number of Certified Beds'], 1)
    
    # Automatic Sprinkler Systems in All Required Areas
    df['sprinklers_in_required_areas'] = np.where(df['Automatic Sprinkler Systems in All Required Areas']=='Yes',1,0)
    df['sprinklers_in_required_areas_partial'] = np.where(df['Automatic Sprinkler Systems in All Required Areas']=='Partial',1,0)

    # 5 star overall rating
    df['five_star_rating_1'] = np.where(df['Overall Rating']==1,1,0)
    df['five_star_rating_2'] = np.where(df['Overall Rating']==2,1,0)
    df['five_star_rating_3'] = np.where(df['Overall Rating']==3,1,0)
    df['five_star_rating_4'] = np.where(df['Overall Rating']==4,1,0)
    df['five_star_rating_5'] = np.where(df['Overall Rating']==5,1,0)
    
    # 5 star health inspection rating
    df['health_inspection_rating_1'] = np.where(df['Health Inspection Rating']==1,1,0)
    df['health_inspection_rating_2'] = np.where(df['Health Inspection Rating']==2,1,0)
    df['health_inspection_rating_3'] = np.where(df['Health Inspection Rating']==3,1,0)
    df['health_inspection_rating_4'] = np.where(df['Health Inspection Rating']==4,1,0)
    df['health_inspection_rating_5'] = np.where(df['Health Inspection Rating']==5,1,0)
    
    # 5 star health inspection rating
    df['health_inspection_rating_1'] = np.where(df['Health Inspection Rating']==1,1,0)
    df['health_inspection_rating_2'] = np.where(df['Health Inspection Rating']==2,1,0)
    df['health_inspection_rating_3'] = np.where(df['Health Inspection Rating']==3,1,0)
    df['health_inspection_rating_4'] = np.where(df['Health Inspection Rating']==4,1,0)
    df['health_inspection_rating_5'] = np.where(df['Health Inspection Rating']==5,1,0)
    
    # 5 star quality rating
    df['quality_rating_1'] = np.where(df['QM Rating']==1,1,0)
    df['quality_rating_2'] = np.where(df['QM Rating']==2,1,0)
    df['quality_rating_3'] = np.where(df['QM Rating']==3,1,0)
    df['quality_rating_4'] = np.where(df['QM Rating']==4,1,0)
    df['quality_rating_5'] = np.where(df['QM Rating']==5,1,0)
    
    # 5 star long-stay quality rating
    df['long_stay_quality_rating_1'] = np.where(df['Long-Stay QM Rating']==1,1,0)
    df['long_stay_quality_rating_2'] = np.where(df['Long-Stay QM Rating']==2,1,0)
    df['long_stay_quality_rating_3'] = np.where(df['Long-Stay QM Rating']==3,1,0)
    df['long_stay_quality_rating_4'] = np.where(df['Long-Stay QM Rating']==4,1,0)
    df['long_stay_quality_rating_5'] = np.where(df['Long-Stay QM Rating']==5,1,0)
    
    # 5 star short-stay quality rating
    df['short_stay_quality_rating_1'] = np.where(df['Short-Stay QM Rating']==1,1,0)
    df['short_stay_quality_rating_2'] = np.where(df['Short-Stay QM Rating']==2,1,0)
    df['short_stay_quality_rating_3'] = np.where(df['Short-Stay QM Rating']==3,1,0)
    df['short_stay_quality_rating_4'] = np.where(df['Short-Stay QM Rating']==4,1,0)
    df['short_stay_quality_rating_5'] = np.where(df['Short-Stay QM Rating']==5,1,0)
    
    # 5 star staffing rating ############ obvious - remove later
    df['staffing_rating_1'] = np.where(df['Staffing Rating']==1,1,0)
    df['staffing_rating_2'] = np.where(df['Staffing Rating']==2,1,0)
    df['staffing_rating_3'] = np.where(df['Staffing Rating']==3,1,0)
    df['staffing_rating_4'] = np.where(df['Staffing Rating']==4,1,0)
    df['staffing_rating_5'] = np.where(df['Staffing Rating']==5,1,0)
    
    # Average dollar amount per fine (severity of fines)
    df['dollars_per_fine_count'] = np.where(df['Number of Fines']!=0, df['Total Amount of Fines in Dollars']/df['Number of Fines'], 0)

    return df

# Call the generate features function
all_dat = generate_features(all_dat)

# Drop young facilities
#all_dat = all_dat.loc[all_dat['months_since_approval'] >= 12]
#all_dat.reset_index(drop=True)

# Select variables for model training
model_vars = set([
    #"Number of Certified Beds"
    #,"Average Number of Residents per Day" #### highly correlated with beds
    'avg_daily_residents_per_bed_ratio'
    #,'Total nursing staff turnover'
    #,'Registered Nurse turnover'
    ,'Number of administrators who have left the nursing home'
    
    ,'Rating Cycle 1 Total Number of Health Deficiencies'
    ,'Rating Cycle 1 Number of Standard Health Deficiencies'
    ,'Rating Cycle 1 Number of Complaint Health Deficiencies'
    ,'Rating Cycle 1 Health Deficiency Score'
    ,'Rating Cycle 1 Number of Health Revisits'
    ,'Rating Cycle 1 Health Revisit Score'
    ,'Rating Cycle 1 Total Health Score'
    
    ,'Rating Cycle 2 Total Number of Health Deficiencies'
    ,'Rating Cycle 2 Number of Standard Health Deficiencies'
    ,'Rating Cycle 2 Number of Complaint Health Deficiencies'
    ,'Rating Cycle 2 Health Deficiency Score'
    ,'Rating Cycle 2 Number of Health Revisits'
    ,'Rating Cycle 2 Health Revisit Score'
    ,'Rating Cycle 2 Total Health Score'
    
    ,'Rating Cycle 3 Total Number of Health Deficiencies'
    ,'Rating Cycle 3 Number of Standard Health Deficiencies'
    ,'Rating Cycle 3 Number of Complaint Health Deficiencies'
    ,'Rating Cycle 3 Health Deficiency Score'
    ,'Rating Cycle 3 Number of Health Revisits'
    ,'Rating Cycle 3 Health Revisit Score'
    ,'Rating Cycle 3 Total Health Score'
    
    ,'Number of Facility Reported Incidents'
    ,'Number of Substantiated Complaints'
    ,'Number of Citations from Infection Control Inspections'
    ,'Number of Fines'
    ,'Total Amount of Fines in Dollars'
    ,'dollars_per_fine_count'
    ,'Number of Payment Denials'
    ,'Total Number of Penalties'
    ] + \
[c for c in all_dat if "OWNERSHIP_" in c.upper()] + \
[c for c in all_dat if "PROVIDER_" in c.upper()] + \
[c for c in all_dat if "MONTHS_SINCE_APPROVAL" in c.upper()] + \
[c for c in all_dat if "CONTINUING_CARE_RETIREMENT_COMMUNITY" in c.upper()] + \
[c for c in all_dat if "SPECIAL_FOCUS" in c.upper()] + \
[c for c in all_dat if "ABUSE_ICON" in c.upper()] + \
[c for c in all_dat if "MOST_RECENT_HEALTH_INSPECTION" in c.upper()] + \
[c for c in all_dat if "COUNCIL_" in c.upper()] + \
[c for c in all_dat if "SPRINKLERS_" in c.upper()] + \
#[c for c in all_dat if "FIVE_STAR_" in c.upper()] + \
[c for c in all_dat if "HEALTH_INSPECTION_RATING_" in c.upper()] + \
#[c for c in all_dat if "QUALITY_RATING_" in c.upper()] + \
#[c for c in all_dat if "STAFFING_RATING_" in c.upper()] + \
[c for c in all_dat if c in post_health_citations['Scope Severity Code'].unique()] + \
#[c for c in all_dat if c in post_health_citations['Deficiency Category'].unique()] + \
[c for c in all_dat if c in post_health_citations['Deficiency Tag Number'].unique()] + \
[c for c in all_dat if c in post_mds_quality_measures['Measure Code'].unique()] + \
[c for c in all_dat if c in post_claims_quality_measures['Measure Code'].unique()]
)
    
# Pop any variables not helping prediction
model_vars.remove('months_since_approval_prev')
#model_vars.remove('mds_quality_code_408')

# Drop facilities that were new in 2018
#all_dat = all_dat.loc[all_dat['months_since_approval_prev'] >= 36]

# Drop facilities that had very little staffing change
all_dat = all_dat.loc[(all_dat[response_var + "_CALC"] <= cutoff) | (all_dat[response_var + "_CALC"] >= (cutoff*-1))]
all_dat = all_dat.reset_index(drop=True)

print("Number of facilities in dataset:")
print(len(all_dat))

# Create a histogram of the target variable
hist_target = px.histogram(all_dat
                           ,response_var + "_CALC"
                           ,color="Lower_Staffing"
                           ,color_discrete_sequence=px.colors.qualitative.Set1
                           )
hist_target.write_html("hist_target_images/hist_target.html")

# Train test split
# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(all_dat[list(model_vars)]
                                                    , all_dat[response_var]
                                                    , test_size=0.33
                                                    , random_state=42
                                                    )

# Create the LightGBM classifier
lgb_model = lgbm.LGBMClassifier(boosting_type='gbdt'
                                , num_threads=4
                                , class_weight='balanced'
                                , learning_rate=0.001
                                , num_leaves=31
                                #, lambda_l2=1.0
                                )
#, num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100)

# Train the model
lgb_model.fit(X_train, y_train)

# Make predictions
y_pred = lgb_model.predict(X_test)
y_fit = lgb_model.predict(X_train)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.2f}")

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")

f1 = f1_score(y_test, y_pred)
print(f"F1: {f1:.2f}")

roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC_AUC: {roc_auc:.2f}")

################################################ Confusion Matrix
# Create a confusion matrix
cm_count_fit = confusion_matrix(y_train, y_fit)
# Visualize the confusion matrix
disp_count_fit = ConfusionMatrixDisplay(confusion_matrix=cm_count_fit, display_labels=lgb_model.classes_)
disp_count_fit.plot(cmap='Blues')
plt.title('Training Fit Count')
plt.show()

# Create a confusion matrix
cm_norm_fit = confusion_matrix(y_train, y_fit, normalize='true')
# Visualize the confusion matrix
disp_norm_fit = ConfusionMatrixDisplay(confusion_matrix=cm_norm_fit, display_labels=lgb_model.classes_)
disp_norm_fit.plot(cmap='Blues')
plt.title('Training Fit Perc')
plt.show()

# Create a confusion matrix
cm_count = confusion_matrix(y_test, y_pred)
# Visualize the confusion matrix
disp_count = ConfusionMatrixDisplay(confusion_matrix=cm_count, display_labels=lgb_model.classes_)
disp_count.plot(cmap='Blues')
plt.title("Test Count")
plt.show()

# Create a confusion matrix
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
# Visualize the confusion matrix
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=lgb_model.classes_)
disp_norm.plot(cmap='Blues')
plt.title("Test Perc")
plt.show()

########################################### Correlation map
# Compute correlation matrix
corr = all_dat[list(model_vars)].corr()

# Create a heatmap
plt.figure(figsize=(16, 12))  # Adjust the size as needed
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.7)

# Add title
plt.title("Correlation Matrix Heatmap")

# Display the plot
plt.show()


##################### Correlated pairs

# Extract upper triangle of correlation matrix (excluding the diagonal)
correlated_pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        col1 = corr.columns[i]
        col2 = corr.columns[j]
        correlation = corr.iloc[i, j]
        correlated_pairs.append((col1, col2, correlation))

# Sort by absolute value of correlation strength
correlated_pairs = sorted(correlated_pairs, key=lambda x: abs(x[2]), reverse=True)

# Create a file with the sorted pairs
with open('correlated_pairs.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
        
    # Write header
    writer.writerow(['Feature 1', 'Feature 2', 'Correlation'])
        
    # Write each pair
    for pair in correlated_pairs:
        writer.writerow([pair[0], pair[1], f"{pair[2]:.2f}"])

    
    








###################### GBM important features
# Plot feature importance
plt.figure(figsize=(16, 12))  # Adjust plot size
importance = plot_importance(lgb_model, max_num_features=len(model_vars), importance_type='gain', figsize=(16, 12))
plt.title("Feature Importance - LightGBM")
plt.show(importance)

importance = pd.DataFrame({"Feature": list(model_vars), "Importance": lgb_model.feature_importances_})
importance = importance.sort_values(by="Importance", ascending=False)

top_features = importance["Feature"].head(20).tolist()
top_features.append(response_var)
corr_matrix = all_dat[top_features].corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()


####################### Shapley values
#shapsize = 100
#num_features = 25

# Generate SHAP values
explainer = shap.TreeExplainer(lgb_model)  # Use SHAP's TreeExplainer for tree-based models
shap_values = explainer.shap_values(all_dat[list(model_vars)])

# Visualize SHAP values
shap.summary_plot(shap_values, all_dat[list(model_vars)])  # Summary plot for overall feature importance








X_test['prediction'] = y_pred
X_test[response_var] = y_test

######################## False positives
fp = X_test.loc[(X_test['prediction']==1) & (X_test[response_var]==0)]

######################## False negatives
fn = X_test.loc[(X_test['prediction']==0) & (X_test[response_var]==1)]




px.scatter(all_dat
           , 'mds_quality_code_408'
           , response_var+"_CALC"
           , color=response_var
           ).write_html("qual_code_408_scatter.html")

px.scatter(all_dat
           , 'mds_quality_code_401'
           , response_var+"_CALC"
           , color=response_var
           ).write_html("qual_code_401_scatter.html")
px.scatter(all_dat
           , 'mds_quality_code_480'
           , response_var+"_CALC"
           , color=response_var
           ).write_html("qual_code_480_scatter.html")

px.scatter(all_dat
           , 'claims_quality_code_521'
           , response_var+"_CALC"
           , color=response_var
           ).write_html("claims_code_521_scatter.html")
px.scatter(all_dat
           , 'claims_quality_code_522'
           , response_var+"_CALC"
           , color=response_var
           ).write_html("claims_code_522_scatter.html")
px.scatter(all_dat
           , 'avg_daily_residents_per_bed_ratio'
           , response_var+"_CALC"
           , color=response_var
           ).write_html("residents_per_bed_scatter.html")


#px.box(all_dat, response_var, 'mds_quality_code_408').write_html("qual_code_408_box.html")


px.box(all_dat, "ownership_for_profit_llc"
       , response_var + "_CALC"
       ).write_html("llc_box.html")
px.box(all_dat
       , response_var
       , "mds_quality_code_408"
       , color="Lower_Staffing"
       ).write_html("mds_quality_code_408_box.html")
px.box(all_dat, response_var
       , "mds_quality_code_401"
       ).write_html("mds_quality_code_401_box.html")
px.box(all_dat, response_var
       , "mds_quality_code_480"
       ).write_html("mds_quality_code_480_box.html")
px.box(all_dat, response_var
       , "mds_quality_code_479"
       ).write_html("mds_quality_code_479_box.html")
px.box(all_dat, response_var
       , "claims_quality_code_521"
       ).write_html("claims_quality_code_521_box.html")









