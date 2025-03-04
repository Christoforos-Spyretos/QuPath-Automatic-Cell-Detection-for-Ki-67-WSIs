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
df_KI67.to_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/CBTN_KI67_aligned_with_MRI.csv', index=False)

# %% FEEL MISSING INFO MANALLY
df_KI67.loc[(df_KI67['case_id'] == 'C22017') & (df_KI67['slide_id'] == 'C22017___7316-70___Ki-67'), ['label', 'tumor_descriptor']] = ['GANG', 'Initial CNS Tumor']

df_KI67.loc[(df_KI67['case_id'] == 'C53013') & (df_KI67['slide_id'] == 'C53013___7316-958___Ki-67'), ['tumor_descriptor']] = ['Progressive']
df_KI67.loc[(df_KI67['case_id'] == 'C53013') & (df_KI67['slide_id'] == 'C53013___7316-3022___Ki-67_A1'), ['tumor_descriptor']] = ['Progressive']

df_KI67.loc[(df_KI67['case_id'] == 'C245754') & (df_KI67['slide_id'] == 'C245754___7316-901___Ki-67'), ['tumor_descriptor']] = ['Second Malignacy']

df_KI67.loc[(df_KI67['case_id'] == 'C19926') & (df_KI67['slide_id'] == 'C19926___7316-7502___KI-67'), ['tumor_descriptor']] = ['Second Malignacy']

# save csv file
df_KI67.to_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/CBTN_KI67_aligned_with_MRI.csv', index=False)

# %%
# subjects with 2 or more diagnoses 
print('Subjects with 2 or more diagnoses:')
i = 0
for subject in df_KI67['case_id'].unique():
    if len(df_KI67[df_KI67['case_id'] == subject]) > 1:
        i += 1
        print(subject)
        # flag them in a new column
        df_KI67.loc[df_KI67['case_id'] == subject, '2_or_more_diagnoses'] = True
    else:
        df_KI67.loc[df_KI67['case_id'] == subject, '2_or_more_diagnoses'] = False

print(f'Total number of subjects with 2 or more diagnoses: {i}')

# %%
print('Subjects with 2 or more tumor descriptors:')
i = 0
# subjects with 2 or more tumor descriptors 
for subject in df_KI67['case_id'].unique():
    if df_KI67[df_KI67['case_id'] == subject]['tumor_descriptor'].nunique() > 1:
        i += 1
        print(subject)
        # flag them in a new column
        df_KI67.loc[df_KI67['case_id'] == subject, '2_or_more_tumor_descriptors'] = True
    else:
        df_KI67.loc[df_KI67['case_id'] == subject, '2_or_more_tumor_descriptors'] = False
    
print(f'Total number of subjects with 2 or more tumor descriptors: {i}')

# %%
print('Subjects with initial CNS tumor:')
i = 0
# subjects with initial CNS tumor
for subject in df_KI67['case_id'].unique():
    if df_KI67.loc[df_KI67['case_id'] == subject, 'tumor_descriptor'].eq('Initial CNS Tumor').any():
        i += 1
        print(subject)
        # flag them in a new column
        df_KI67.loc[df_KI67['case_id'] == subject, 'initial_CNS_tumor'] = True
    else:
        df_KI67.loc[df_KI67['case_id'] == subject, 'initial_CNS_tumor'] = False
    
print(f'Total number of subjects with initial CNS tumor: {i}')

# %%
print('Subjects with second malignacy:')
i = 0
# subjects with second malignancy
for subject in df_KI67['case_id'].unique():
    if df_KI67.loc[df_KI67['case_id'] == subject, 'tumor_descriptor'].eq('Second Malignancy').any():
        i += 1
        print(subject)
        # flag them in a new column
        df_KI67.loc[df_KI67['case_id'] == subject, 'second_malignancy'] = True
    else:
        df_KI67.loc[df_KI67['case_id'] == subject, 'second_malignancy'] = False
    
print(f'Total number of subjects with second malignacy: {i}')

# %%
print('Subjects with progressive:')
i = 0
# subjects with progressive
for subject in df_KI67['case_id'].unique():
    if df_KI67.loc[df_KI67['case_id'] == subject, 'tumor_descriptor'].eq('Progressive').any():
        i += 1
        print(subject)
        # flag them in a new column
        df_KI67.loc[df_KI67['case_id'] == subject, 'progressive'] = True
    else:
        df_KI67.loc[df_KI67['case_id'] == subject, 'progressive'] = False
    
print(f'Total number of subjects with progressive: {i}')

# %%
print('Subjects with recurrence:')
i = 0
# subjects with recurrence
for subject in df_KI67['case_id'].unique():
    if df_KI67.loc[df_KI67['case_id'] == subject, 'tumor_descriptor'].eq('Recurrence').any():
        i += 1
        print(subject)
        # flag them in a new column
        df_KI67.loc[df_KI67['case_id'] == subject, 'recurrence'] = True
    else:
        df_KI67.loc[df_KI67['case_id'] == subject, 'recurrence'] = False
    
print(f'Total number of subjects with recurrence: {i}')

# %% SAVE CSV FILE
df_KI67.to_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/CBTN_KI67_aligned_with_MRI.csv', index=False)

# %%