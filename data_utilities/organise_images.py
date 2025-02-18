'''

'''

# %% IMPORTS
import os
import pandas as pd
import shutil

# %% LOAD CSV FILES & DIRECTORIES
df_KI67 = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/CBTN_KI67.csv')
path_to_imgs = '/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/HISTOLOGY/SUBJECTS'
save_path = '/run/media/chrsp39/CBNT_v2/KI67/WSI'

if not os.path.exists(save_path):
        os.makedirs(save_path)

# %% KI-67 NAMES
KI67_names = ["KI-67", "ki-67", "Ki-67", "KI67", "Ki067_A2", "Ki-62", "Ki-57", "Ki67", "Ki-67_B1", "H_and_E_Ki-67", "KI-67_A1", "KI-67_A2",
              "KI-67_C1", "KI-67_B", "KI-67_B2", "1_A_Ki67", "1_B_Ki67", "KI-67_B3", "ki-67_C2", "ki-67_D1", "KI-67_A3", "KI-67_C", "KI-67_",
              "ki-67_A4", "KI-67_A", "ki-67_(B)", "Ki-67_C3", "1_D_Ki67", "Ki-67_D", "Ki-67_2", "Ki-67_A5", "Ki-67_3", "Ki67_A1", "Ki067_A2",
              "Ki67_C2", "Ki-67_D2", "1_C_Ki67", "Ki-67_FSA", "Ki-67_1", "Ki-67_FS", "Ki-67_C4", "Ki-57", "2_A_Ki67", "Ki-67_(B2)", "Ki67_(D5)",
              "Ki-67_E1", "Ki-67__C2", "Ki-67_(2)", "Ki67_B1", "Ki-62", "1_F_Ki67", "Ki67_B3", "__Ki-67_D1", "9980_1B_KI67", "KI-67_BLOCK_C",
              "Ki-67_BLOCK_D", "4492-Ki67", "Ki-67_(C2)", "KI-67_B4", "Ki-67_(A2)", "Ki-67_(C3)", "Ki-67_A10", "Ki-67_S-05-6044", "Ki-67_B2",
              "666-ki67-001", "666-ki67", "KI67_BLOCK_A3", "KI67-_BLOCK_B1", "Ki-67_E", "2_B_Ki67", "4992-ki67", "4992-ki67-001", "4992-ki67-002",
              "KI67_C1", "Ki-67,_Ki-67", "Ki-67_B9", "5432-Ki67", "Ki-67_(S-08-2219)", "_Ki-67_B1", "Ki-67_(3)", "Ki67_B2", "Ki-67_(FS)", "Ki-67_B4",
              "Ki-67_B6", "3477-ki67", "3477-ki67-001", "Ki-67-A", "1_E_Ki67", "956_1A_KI67", "1184_-_2A_Ki67_MIB-1", "Ki-67_(S-14-904)", "Ki-67_FSB",
              "Ki-67_(D)", "Ki-67_(A)", "KI-67_BLOCK_2", "KI-67_BLOCK_1", "Ki-67_C9", "Ki-67_(A1)", "10412_1B_KI67"]

# %% COPY SVS FILES
wsi_files = os.listdir(save_path)

subject_ids = df_KI67['case_id'].values

i = 0
for subject_id in df_KI67['case_id']:
    i += 1
    path_to_session = os.path.join(path_to_imgs, subject_id, 'SESSIONS')  
    if os.path.exists(path_to_session):
        sessions_ids = os.listdir(path_to_session)        
        
        for session_id in sessions_ids:

            # if folder exists skip
            if os.path.exists(os.path.join(save_path, subject_id, session_id)):
                continue

            path_to_slides = os.path.join(path_to_session, session_id, "ACQUISITIONS", "Files", "FILES")

            if os.path.exists(path_to_slides):
                files = os.listdir(path_to_slides)
                svs_files = [file for file in files if file.endswith('.svs')]

                for svs_file in svs_files:
                    svs_file_name = os.path.splitext(svs_file)[0]
                    for KI67_name in KI67_names:
                        if KI67_name in svs_file_name:
                            slide_id = subject_id + '___' + session_id + '___' + svs_file_name + '.svs'

                            if slide_id in wsi_files:
                                print(f'{slide_id} already exists')
                            elif slide_id not in wsi_files:
                                os.makedirs(os.path.join(save_path, subject_id, session_id), exist_ok=True)
                                shutil.copy(os.path.join(path_to_slides, svs_file), os.path.join(save_path, subject_id, session_id, svs_file))
                                wsi_files.append(slide_id)

    print('Subjects copied: {} / {}'.format(i, len(subject_ids)), end='\r')

# %% RENAME SVS FILES
wsi_folders = os.listdir(save_path)

# select only folders and not .svs files
wsi_folders = [folder for folder in wsi_folders if not folder.endswith('.svs')]

i = 0
for subject_id in wsi_folders:
    i += 1
    subject_path = os.path.join(save_path, subject_id)

    if os.path.exists(subject_path):
        sessions = os.listdir(subject_path)

        for session in sessions:
            session_path = os.path.join(subject_path, session)

            if os.path.exists(session_path):
                for file in os.listdir(session_path):
                    if file.endswith(".svs"):
                        os.rename(os.path.join(session_path, file), os.path.join(session_path, subject_id + "___" + session + "___" + file))   

    print('Subjects renamed: {} / {}'.format(i, len(wsi_folders)), end='\r')
          
# %% MOVE SVS FILES TO MAIN DIRECTORY
wsi_folders = os.listdir(save_path)

# select only folders and not .svs files
wsi_folders = [folder for folder in wsi_folders if not folder.endswith('.svs')]

i = 0
for subject_id in wsi_folders:
    i += 1
    subject_path = os.path.join(save_path, subject_id)

    if os.path.exists(subject_path):
        sessions = os.listdir(subject_path)

        for session in sessions:
            session_path = os.path.join(subject_path, session)

            if os.path.exists(session_path):
                for file in os.listdir(session_path):
                    if file.endswith(".svs"):
                        shutil.copy(session_path + "/" + file, save_path)

    shutil.rmtree(subject_path)

    print('Subjects transferred: {} / {}'.format(i, len(wsi_folders)), end='\r')
    
# %%
