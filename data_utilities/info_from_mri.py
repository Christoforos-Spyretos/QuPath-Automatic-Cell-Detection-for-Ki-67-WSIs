'''
Script to extract the tumor descriptor from the MRI data 
and assign it to the KI67 data.

The tumor descriptor is the MRI session that is closest to 
the age at histological diagnosis of the patient.
'''

# %% IMPORTS
import pandas as pd

# %% LOAD CSV FILES
df_KI67 = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/CBTN_KI67.csv')
df_MRI = pd.read_excel('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/MRI_summary_extended.xlsx')

# rename column name in df_MRI from 'subjetID' to 'subjectID'  
df_MRI.rename(columns={'subjetID': 'subjectID'}, inplace=True)

# %% FILTER SUBJECTS AND RENAME MRI DATA BASED ON TUMOUR TYPES
# get common subjects
df_MRI['subjectID'] = 'C' + df_MRI['subjectID'].astype(str)

subjects_KI67 = df_KI67['case_id'].unique()
subjects_MRI = df_MRI['subjectID'].unique()

common_subjects = list(set(subjects_KI67).intersection(subjects_MRI))

df_KI67 = df_KI67[df_KI67['case_id'].isin(common_subjects)]
df_MRI = df_MRI[df_MRI['subjectID'].isin(common_subjects)]

# brain tumor types interested in
tumor_types = [
    'Low-grade glioma/astrocytoma (WHO grade I/II)',
    'High-grade glioma/astrocytoma (WHO grade III/IV)', 
    'Medulloblastoma',
    'Ependymoma',
    'Brainstem glioma- Diffuse intrinsic pontine glioma',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)',
    'Meningioma',
    'Craniopharyngioma',
    'Dysembryoplastic neuroepithelial tumor (DNET)',
    'Ganglioglioma'
    ]

df_MRI = df_MRI[df_MRI['diagnosis'].isin(tumor_types)]

# rename tumor types
df_MRI['diagnosis'] = df_MRI['diagnosis'].replace({
    'Low-grade glioma/astrocytoma (WHO grade I/II)': 'ASTR_LGG',
    'High-grade glioma/astrocytoma (WHO grade III/IV)': 'ASTR_HGG', 
    'Medulloblastoma': 'MED',
    'Ependymoma': 'EP',
    'Brainstem glioma- Diffuse intrinsic pontine glioma': 'DIPG',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)': 'ATRT',
    'Meningioma': 'MEN',
    'Craniopharyngioma': 'CRAN',
    'Dysembryoplastic neuroepithelial tumor (DNET)': 'DNET',
    'Ganglioglioma': 'GANG'
})

# %% EXTRACT THE DAY OF THE SESSION
df_MRI['day_from_session_name'] = df_MRI['session_name']

for index, row in df_MRI.iterrows():
    session_name = row['day_from_session_name']
    day = session_name.split("d")[0]
    df_MRI.at[index, 'day_from_session_name'] = day

# %% GET THE TUMOR DESCRIPTOR
df_KI67['tumor_descriptor'] = None

for ki67_index, ki67_row in df_KI67.iterrows():
    subject_id = ki67_row['case_id']
    label = ki67_row['label']
    age_at_diagnosis = ki67_row['age_at_diagnosis_(days)']

    mri_rows = df_MRI[(df_MRI['subjectID'] == subject_id) & (df_MRI['diagnosis'] == label)]
    
    if not mri_rows.empty:
        # calculate the absolute difference between ages
        mri_rows['age_difference'] = abs(mri_rows['day_from_session_name'].astype(int) - age_at_diagnosis)
        
        # find the row with the smallest age difference
        closest_row = mri_rows.loc[mri_rows['age_difference'].idxmin()]
        
        # assign the tumor descriptor to df_KI67
        df_KI67.at[ki67_index, 'tumor_descriptor'] = closest_row['tumor_descriptor']

# %% SAVE CSV FILE
df_KI67.to_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/CBTN_KI67.csv', index=False)

# %%
