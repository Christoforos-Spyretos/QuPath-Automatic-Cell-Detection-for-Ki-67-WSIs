# %% IMPORT
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import os

# %% LOAD DATA
# read xlsx file icluding the column names
df_sum_up = pd.read_excel('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/KI67_sum_up.xlsx')    
df_full_data = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/CBTN_KI67.csv')
df_alinged_with_MRI = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/CBTN_KI67_aligned_with_MRI.csv')

# %% 
df_sum_up['label'] = df_sum_up['label'].replace({'ASTR_LGG': 'LGG', 'ASTR_HGG': 'HGG'})
df_full_data['label'] = df_full_data['label'].replace({'ASTR_LGG': 'LGG', 'ASTR_HGG': 'HGG'})
df_alinged_with_MRI['label'] = df_alinged_with_MRI['label'].replace({'ASTR_LGG': 'LGG', 'ASTR_HGG': 'HGG'})

# %% FILTERING 
# number of excluded cases in full dataset
df_sum_up_1 = df_sum_up[df_sum_up['slide_id'].isin(df_full_data['slide_id'])]
na_count = df_sum_up_1['KI67_LI_2'].isna().sum()
print(f"Number of excluded cases in full dataset': {na_count}")
# drop the na values from the df_sum_up_1
df_sum_up_1 = df_sum_up_1.dropna(subset=['KI67_LI_2'])

# number of excluded cases in dataset with aligned MRI
df_sum_up_2 = df_sum_up[df_sum_up['slide_id'].isin(df_alinged_with_MRI['slide_id'])]
na_count = df_sum_up_2['KI67_LI_2'].isna().sum()
print(f"Number of excluded cases in dataset with aligned MRI: {na_count}")
# drop the na values from the df_sum_up_2
df_sum_up_2 = df_sum_up_2.dropna(subset=['KI67_LI_2'])


# %% BOXPLOT
sns.set_style("whitegrid", {'axes.grid' : False})
plt.figure(figsize=(10, 6))  
plt.gca().set_facecolor('white')  
flierprops = dict(marker='D', markerfacecolor='darkgrey', markersize=5, linestyle='none')
order = ['LGG', 'GANG', 'HGG', 'MED', 'EP', 'MEN', 'DIPG', 'ATRT']
boxplot = sns.boxplot(x='label', y='KI67_LI_2', data=df_sum_up_2, flierprops=flierprops, order=order)

plt.title('Ki-67 label index across all diagnoses') 
plt.xlabel('Diagnosis')
plt.ylabel('Ki-67 label index')
plt.tight_layout()  
plt.show()

# %% BOXPLOT of initial_CNS_tumor
initial_CNS_tumor = df_sum_up[df_sum_up['initial_CNS_tumor'] == 1]

sns.set_style("whitegrid", {'axes.grid' : False})
plt.figure(figsize=(10, 6))  
plt.gca().set_facecolor('white')  
flierprops = dict(marker='D', markerfacecolor='darkgrey', markersize=5, linestyle='none')
order = ['LGG', 'GANG', 'HGG', 'MED', 'EP', 'MEN', 'DIPG', 'ATRT']
boxplot = sns.boxplot(x='label', y='KI67_LI_2', data=initial_CNS_tumor, flierprops=flierprops, order=order)
plt.title('Diagnosis with an initial brain tumor descriptor')
plt.xlabel('Diagnosis')
plt.ylabel('Ki-67 label index')
plt.tight_layout()  
plt.show()

# %% BOXPLOT of second_malignancy
second_malignancy = df_sum_up[df_sum_up['second_malignancy'] == 1]

plt.figure(figsize=(10, 6))  
plt.gca().set_facecolor('white')  
flierprops = dict(marker='D', markerfacecolor='darkgrey', markersize=5, linestyle='none')
boxplot = sns.boxplot(x='label', y='KI67_LI_2', data=second_malignancy, flierprops=flierprops)
plt.title('Diagnosis with a second malignacy tumor descriptor')
plt.xlabel('Diagnosis')
plt.ylabel('Ki-67 label index')
plt.tight_layout()
plt.show()

# %% BOXPLOT of progressive
progressive = df_sum_up[df_sum_up['progressive'] == 1]

plt.figure(figsize=(10, 6))  
plt.gca().set_facecolor('white')
flierprops = dict(marker='D', markerfacecolor='darkgrey', markersize=5, linestyle='none')
boxplot = sns.boxplot(x='label', y='KI67_LI_2', data=progressive, flierprops=flierprops)
plt.title('Diagnosis with a progressive tumor descriptor')
plt.xlabel('Diagnosis')
plt.ylabel('Ki-67 label index')
plt.tight_layout() 
plt.show()

# %% BOXPLOT of recurrence
recurrence = df_sum_up[df_sum_up['recurrence'] == 1]

plt.figure(figsize=(10, 6))  
plt.gca().set_facecolor('white')  
flierprops = dict(marker='D', markerfacecolor='darkgrey', markersize=5, linestyle='none')
boxplot = sns.boxplot(x='label', y='KI67_LI_2', data=recurrence, flierprops=flierprops)
plt.title('Diagnosis with a recurrence tumor descriptor')
plt.xlabel('Diagnosis')
plt.ylabel('Ki-67 label index')
plt.tight_layout()  
plt.show()

# %% PRINT TABLE WITH 2_or_more_tumor_descriptors
df_sum_up_2_tumor_descriptors = df_sum_up[df_sum_up['2_or_more_tumor_descriptors'] == 1]
df_sum_up_2_tumor_descriptors = df_sum_up_2_tumor_descriptors[['case_id', 'slide_id', 'label', 'KI67_LI_2', 'age_at_diagnosis_(days)', 'tumor_descriptor']]
df_sum_up_2_tumor_descriptors = df_sum_up_2_tumor_descriptors.sort_values(by=['case_id', 'age_at_diagnosis_(days)'])
# remove the case_id that has one entry 
df_sum_up_2_tumor_descriptors = df_sum_up_2_tumor_descriptors[df_sum_up_2_tumor_descriptors.duplicated(subset=['case_id'], keep=False)]
markdown_table = df_sum_up_2_tumor_descriptors.to_markdown(index=False)
print(markdown_table)

# %% ANALYZE KI67_LI_2 CHANGES
df_sum_up_2_tumor_descriptors_analysis = df_sum_up_2_tumor_descriptors

# Calculate the mean KI67_LI_2 based on the same slide_id
mean_ki67_li_2 = df_sum_up_2_tumor_descriptors_analysis.groupby('slide_id')['KI67_LI_2'].mean().reset_index()
mean_ki67_li_2.columns = ['slide_id', 'mean_KI67_LI_2']

# Merge additional columns based on slide_id
additional_columns = df_sum_up_2_tumor_descriptors_analysis[['slide_id', 'case_id', 'label', 'age_at_diagnosis_(days)', 'tumor_descriptor']].drop_duplicates()
mean_ki67_li_2 = mean_ki67_li_2.merge(additional_columns, on='slide_id', how='left')

mean_ki67_li_2 = mean_ki67_li_2[['case_id', 'slide_id', 'label', 'mean_KI67_LI_2', 'age_at_diagnosis_(days)', 'tumor_descriptor']]
mean_ki67_li_2 = mean_ki67_li_2.sort_values(by=['case_id', 'age_at_diagnosis_(days)'])

def check_increase_decrease(df):
    df = df.sort_values(by=['case_id', 'age_at_diagnosis_(days)'])
    df['Change'] = df.groupby('case_id')['mean_KI67_LI_2'].diff().fillna(0)
    df['Trend'] = df['Change'].apply(lambda x: 'Increased' if x > 0 else 'Decreased')
    return df

mean_ki67_li_2 = check_increase_decrease(mean_ki67_li_2)

mean_ki67_li_2 = mean_ki67_li_2[['case_id', 'label', 'mean_KI67_LI_2', 'age_at_diagnosis_(days)', 'tumor_descriptor', 'Trend']]


# %% PRINT TIF IMAGES
imgs_path = '/local/data2/chrsp39/CBTN_v2/new_KI67/WSI'

tif_images = [f for f in os.listdir(imgs_path) if f.endswith('.tif')]

valid_slide_ids = df_sum_up[~df_sum_up['KI67_LI_2'].isna()]['slide_id'].tolist()
matching_tif_images = [img for img in tif_images if any(slide_id in img for slide_id in valid_slide_ids)]

print(f"Total TIF images: {len(tif_images)}")
print(f"Matching TIF images: {len(matching_tif_images)}")

# %%





