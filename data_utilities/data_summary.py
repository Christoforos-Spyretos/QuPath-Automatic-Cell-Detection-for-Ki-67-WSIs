'''
Script to summarise the data in the CBTN KI-67 dataset.

Loads the CSV file and prints a summary table and barplot of 
the number of subjects and slides with KI-67 stain per tumour 
family/type.
'''

# %% IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% LOAD CSV FILES
df_KI67 = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/CBTN_KI67.csv')
df_KI67_aligned_with_MRI = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/CBTN_KI67_aligned_with_MRI.csv')

# %% DATA SUMMARY OF KI-67
subjects_per_tumour_KI67 = df_KI67.groupby(['case_id'])['label'].min().reset_index()
sorted_subjects_per_tumour_KI67 = subjects_per_tumour_KI67.iloc[:, 1].value_counts().sort_values(ascending=False)
subjects_per_tumour_KI67 = pd.DataFrame(sorted_subjects_per_tumour_KI67).reset_index()

sorted_images_per_tumour_K67= df_KI67.iloc[:, 2].value_counts().sort_values(ascending=False)
images_per_tumour_KI67 = pd.DataFrame(sorted_images_per_tumour_K67).reset_index()

merged_df_KI67 = pd.merge(subjects_per_tumour_KI67, images_per_tumour_KI67, on='label', suffixes=('_subjects', '_images'))
merged_df_KI67 = merged_df_KI67.sort_values(by='label')
merged_df_KI67.columns = ['Label', 'Number of Subjects', 'Number of Images']
merged_df_KI67 = merged_df_KI67.sort_values(by='Number of Subjects',ascending=False)
print('KI-67 information')
print(f'Total number of subjects: {merged_df_KI67["Number of Subjects"].sum()}')
print(f'Total number of slides: {merged_df_KI67["Number of Images"].sum()}')
# print(merged_df_KI67)

# table of contents
markdown_table = merged_df_KI67.to_markdown(index=False)
print(markdown_table)

# bar plot
bar_width = 0.4
x = np.arange(len(merged_df_KI67['Label']))

fig = plt.figure(figsize=(12, 8))
plt.bar(x - bar_width/2, merged_df_KI67['Number of Subjects'], color='gray', width=0.4, label='Number of subjects')
for i, count in enumerate(merged_df_KI67['Number of Subjects']):
    plt.text(i - bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x + bar_width/2, merged_df_KI67['Number of Images'], color='darkcyan', width=0.4, label='Number of slides')
for i, count in enumerate(merged_df_KI67['Number of Images']):
    plt.text(i + bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.xlabel("Tumour family/type")
plt.ylabel("Count")
plt.title("Number of subjects and slides with KI-67 stain per tumour family/type")
plt.xticks(x, merged_df_KI67['Label'])
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/Barplot_of_Number_of_subjects_and_slides_with_KI-67_stain_per_tumour_family_type.png')

# %% DATA SUMMARY OF KI-67 ALIGNED WITH MRI
subjects_per_tumour_KI67_aligned_with_MRI = df_KI67_aligned_with_MRI.groupby(['case_id'])['label'].min().reset_index()
sorted_subjects_per_tumour_KI67_aligned_with_MRI = subjects_per_tumour_KI67_aligned_with_MRI.iloc[:, 1].value_counts().sort_values(ascending=False)
subjects_per_tumour_KI67_aligned_with_MRI = pd.DataFrame(sorted_subjects_per_tumour_KI67_aligned_with_MRI).reset_index()

sorted_images_per_tumour_K67_aligned_with_MRI = df_KI67_aligned_with_MRI.iloc[:, 2].value_counts().sort_values(ascending=False)
images_per_tumour_KI67_aligned_with_MRI = pd.DataFrame(sorted_images_per_tumour_K67_aligned_with_MRI).reset_index()

merged_df_KI67_aligned_with_MRI = pd.merge(subjects_per_tumour_KI67_aligned_with_MRI, images_per_tumour_K67_aligned_with_MRI, on='label', suffixes=('_subjects', '_images'))
merged_df_KI67_aligned_with_MRI = merged_df_KI67_aligned_with_MRI.sort_values(by='label')
merged_df_KI67_aligned_with_MRI.columns = ['Label', 'Number of Subjects', 'Number of Images']
merged_df_KI67_aligned_with_MRI = merged_df_KI67_aligned_with_MRI.sort_values(by='Number of Subjects',ascending=False)
print('KI-67 aligned with MRI information')
print(f'Total number of subjects: {merged_df_KI67_aligned_with_MRI["Number of Subjects"].sum()}')
print(f'Total number of slides: {merged_df_KI67_aligned_with_MRI["Number of Images"].sum()}')
# print(merged_df_KI67_aligned_with_MRI)

# table of contents
markdown_table = merged_df_KI67_aligned_with_MRI.to_markdown(index=False)
print(markdown_table)

# bar plot
bar_width = 0.4
x = np.arange(len(merged_df_KI67_aligned_with_MRI['Label']))

fig = plt.figure(figsize=(12, 8))
plt.bar(x - bar_width/2, merged_df_KI67_aligned_with_MRI['Number of Subjects'], color='gray', width=0.4, label='Number of subjects')
for i, count in enumerate(merged_df_KI67_aligned_with_MRI['Number of Subjects']):
    plt.text(i - bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x + bar_width/2, merged_df_KI67_aligned_with_MRI['Number of Images'], color='darkcyan', width=0.4, label='Number of slides')
for i, count in enumerate(merged_df_KI67_aligned_with_MRI['Number of Images']):
    plt.text(i + bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.xlabel("Tumour family/type")
plt.ylabel("Count")
plt.title("Number of subjects and slides with KI-67 stain aligned with MRI per tumour family/type")
plt.xticks(x, merged_df_KI67_aligned_with_MRI['Label'])
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/Barplot_of_Number_of_subjects_and_slides_with_KI-67_stain_aligned_with_MRI_per_tumour_family_type.png')

# %% 
# subjects with 2 or more diagnoses 
print('Subjects with 2 or more diagnoses:')
i = 0
for subject in df_KI67['case_id'].unique():
    if len(df_KI67[df_KI67['case_id'] == subject]) > 1:
        i += 1
        print(subject)

print(f'Total number of subjects with 2 or more diagnoses: {i}')

# %%
print('Subjects with 2 or more tumor descriptors:')
i = 0
# subjects with 2 or more tumor descriptors 
for subject in df_KI67['case_id'].unique():
    if df_KI67[df_KI67['case_id'] == subject]['tumor_descriptor'].nunique() > 1:
        i += 1
        print(subject)
    
print(f'Total number of subjects with 2 or more tumor descriptors: {i}')

# %%
print('Subjects with initial CNS tumor:')
i = 0
# subjects with initial CNS tumor
for subject in df_KI67['case_id'].unique():
    if df_KI67.loc[df_KI67['case_id'] == subject, 'tumor_descriptor'].eq('Initial CNS Tumor').any():
        i += 1
        print(subject)
    
print(f'Total number of subjects with initial CNS tumor: {i}')

# %%
print('Subjects with second malignacy:')
i = 0
# subjects with second malignancy
for subject in df_KI67['case_id'].unique():
    if df_KI67.loc[df_KI67['case_id'] == subject, 'tumor_descriptor'].eq('Second Malignancy').any():
        i += 1
        print(subject)
    
print(f'Total number of subjects with second malignacy: {i}')

# %%
print('Subjects with progressive:')
i = 0
# subjects with progressive
for subject in df_KI67['case_id'].unique():
    if df_KI67.loc[df_KI67['case_id'] == subject, 'tumor_descriptor'].eq('Progressive').any():
        i += 1
        print(subject)
    
print(f'Total number of subjects with progressive: {i}')

# %%
print('Subjects with recurrence:')
i = 0
# subjects with recurrence
for subject in df_KI67['case_id'].unique():
    if df_KI67.loc[df_KI67['case_id'] == subject, 'tumor_descriptor'].eq('Recurrence').any():
        i += 1
        print(subject)
    
print(f'Total number of subjects with recurrence: {i}')

# %%