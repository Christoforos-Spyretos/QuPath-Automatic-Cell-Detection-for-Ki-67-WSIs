'''
Script to curate the data in the CBTN KI-67 dataset.

Loads the clinical data xlsx file and the images directory 
to filter the data and extract the case_id, slide_id and
label information.
'''

# %% IMPORTS
import pandas as pd
import os

# %% LOAD CSV FILES & DIRECTORIES
clinical_data = pd.read_excel('/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/CSV_FILES/CBTN_clinical_data_from_portal.xlsx', sheet_name=3)
path_to_imgs = '/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/HISTOLOGY/SUBJECTS'

# %% FILTER CLINICAL DATA 
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

# Filter clinical data based on tumour types
clinical_data = clinical_data[clinical_data['Histological Diagnosis (Source Text)'].isin(tumor_types)]

# %% GET IDS
external_ids = clinical_data['External Id']
external_sample_ids = clinical_data['External Sample Id']
subject_ids = os.listdir(path_to_imgs)

# match external ids with subject ids
matched_subjects = []

for external_id in external_ids:
    for subject_id in subject_ids:
        if external_id in subject_id:
            matched_subjects.append(subject_id)

matched_subjects = set(matched_subjects)

# %% FIND SLIDES WITH KI-67 STAIN
KI67_names = ["KI-67", "ki-67", "Ki-67", "KI67", "Ki067_A2", "Ki-62", "Ki-57", "Ki67", "Ki-67_B1", "H_and_E_Ki-67", "KI-67_A1", "KI-67_A2",
         "KI-67_C1", "KI-67_B", "KI-67_B2", "1_A_Ki67", "1_B_Ki67", "KI-67_B3", "ki-67_C2", "ki-67_D1", "KI-67_A3", "KI-67_C", "KI-67_",
         "ki-67_A4", "KI-67_A", "ki-67_(B)", "Ki-67_C3", "1_D_Ki67", "Ki-67_D", "Ki-67_2", "Ki-67_A5", "Ki-67_3", "Ki67_A1", "Ki067_A2",
         "Ki67_C2", "Ki-67_D2", "1_C_Ki67", "Ki-67_FSA", "Ki-67_1", "Ki-67_FS", "Ki-67_C4", "Ki-57", "2_A_Ki67", "Ki-67_(B2)", "Ki67_(D5)",
         "Ki-67_E1", "Ki-67__C2", "Ki-67_(2)", "Ki67_B1", "Ki-62", "1_F_Ki67", "Ki67_B3", "__Ki-67_D1", "9980_1B_KI67", "KI-67_BLOCK_C",
         "Ki-67_BLOCK_D", "4492-Ki67", "Ki-67_(C2)", "KI-67_B4", "Ki-67_(A2)", "Ki-67_(C3)", "Ki-67_A10", "Ki-67_S-05-6044", "Ki-67_B2",
         "666-ki67-001", "666-ki67", "KI67_BLOCK_A3", "KI67-_BLOCK_B1", "Ki-67_E", "2_B_Ki67", "4992-ki67", "4992-ki67-001", "4992-ki67-002",
         "KI67_C1", "Ki-67,_Ki-67", "Ki-67_B9", "5432-Ki67", "Ki-67_(S-08-2219)", "_Ki-67_B1", "Ki-67_(3)", "Ki67_B2", "Ki-67_(FS)", "Ki-67_B4",
         "Ki-67_B6", "3477-ki67", "3477-ki67-001", "Ki-67-A", "1_E_Ki67", "956_1A_KI67", "1184_-_2A_Ki67_MIB-1", "Ki-67_(S-14-904)", "Ki-67_FSB",
         "Ki-67_(D)", "Ki-67_(A)", "KI-67_BLOCK_2", "KI-67_BLOCK_1", "Ki-67_C9", "Ki-67_(A1)", "10412_1B_KI67"],

# Flatten the KI67_names list
KI67_names = [item for sublist in KI67_names for item in sublist]

slide_ids = []

for subject_id in matched_subjects:
    path_to_session = os.path.join(path_to_imgs, subject_id, 'SESSIONS')  
    if os.path.exists(path_to_session):
        session_ids = os.listdir(path_to_session)

        for session_id in session_ids:
            path_to_slides = os.path.join(path_to_session, session_id, "ACQUISITIONS", "Files", "FILES")

            if os.path.exists(path_to_slides):
                files = os.listdir(path_to_slides)
                svs_files = [file for file in files if file.endswith('.svs')]

                for svs_file in svs_files:
                    svs_file_name = os.path.splitext(svs_file)[0]
                    for KI67_name in KI67_names:
                        if KI67_name in svs_file_name:
                            slide_id = subject_id + '___' + session_id + '___' + svs_file_name
                            slide_ids.append(slide_id)

# %% GET LABELS
clinical_data_dict = clinical_data.set_index(['External Id', 'External Sample Id'])['Histological Diagnosis (Source Text)'].to_dict()

data = []

# extract label from clinical data
for slide_id in slide_ids:
    subject_id = slide_id.split('___')[0]
    session_id = slide_id.split('___')[1]

    if (subject_id, session_id) in clinical_data_dict:
        label = clinical_data_dict[(subject_id, session_id)]
        data.append({'case_id': subject_id, 'slide_id': slide_id, 'label': label})

# remove duplicates based on slide_id
data = [dict(t) for t in {tuple(d.items()) for d in data}]

df_KI67 = pd.DataFrame(data)

# %% GET AGE AT DIAGNOSIS (DAYS)
# get age at diagnosis (days) from clinical data
clinical_data_dict = clinical_data.set_index(['External Id', 'External Sample Id'])['Age at Diagnosis (Days)'].to_dict()

data = []

for slide_id in slide_ids:
    subject_id = slide_id.split('___')[0]
    session_id = slide_id.split('___')[1]

    if (subject_id, session_id) in clinical_data_dict:
        age = clinical_data_dict[(subject_id, session_id)]
        data.append({'case_id': subject_id, 'slide_id': slide_id, 'age_at_diagnosis_(days)': age})

# remove duplicates based on slide_id
data = [dict(t) for t in {tuple(d.items()) for d in data}]

df_KI67 = pd.merge(df_KI67, pd.DataFrame(data), on=['case_id', 'slide_id'], how='left')

# %% RENAME LABELS  
df_KI67['label'] = df_KI67['label'].replace({
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

# %% SAVE CSV FILE
df_KI67.to_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/CBTN_KI67.csv', index=False)

# %%