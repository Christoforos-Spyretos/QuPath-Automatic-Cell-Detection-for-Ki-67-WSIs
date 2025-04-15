'''
Script to summarise the data in the CBTN KI-67 dataset.
'''

# %% IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% LOAD CSV FILES
df_KI67 = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/CBTN_KI67.csv')
df_KI67_aligned_with_MRI = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/CBTN_KI67_aligned_with_MRI.csv')
final_results = pd.read_excel('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/QuPath_Ki-67_summary_analysis.xlsx')

# %% DATA SUMMARY OF KI-67
print('SUMMARY OF THE FULL KI-67 DATASET')
print('Total number of subjects:')
print(df_KI67['case_id'].nunique())
print('Total number of images:')
print(df_KI67['slide_id'].nunique())

print()

unique_comb_case_id_label= df_KI67
i = 0
double_diagnosis_subjects = []
for subject in unique_comb_case_id_label['case_id'].unique():
    if unique_comb_case_id_label[unique_comb_case_id_label['case_id'] == subject]['label'].nunique() > 1:
        i += 1
        double_diagnosis_subjects.append(subject)

print(f'Total number of subjects with 2 or more unique labels: {i}')
print('Subjects with 2 or more unique labels:')
for subject in double_diagnosis_subjects:
    print(subject)

print()

print('Unique combinations of subjects and labels:')
print(df_KI67[['case_id', 'label']].drop_duplicates().shape[0])
summary_table = df_KI67.groupby('label').agg(
    num_subjects=('case_id', 'nunique'),
    num_images=('slide_id', 'nunique')
).reset_index()

summary_table = summary_table.sort_values(by='num_subjects', ascending=False)

print('| Label | Number of Subjects | Number of Images |')
print('|-------|--------------------|------------------|')
for _, row in summary_table.iterrows():
    print(f"| {row['label']} | {row['num_subjects']} | {row['num_images']} |")
print(f"| Total | {summary_table['num_subjects'].sum()} | {summary_table['num_images'].sum()} |")

print()

print('Sex distribution:')
# filter by unique combinations of case_id and label
sex_df = df_KI67.drop_duplicates(subset=['case_id', 'label'])
sex_counts = sex_df['sex'].value_counts()
sex_percentages = (sex_counts / sex_counts.sum()) * 100
sex_counts_with_percentages = sex_counts.astype(str) + " (" + sex_percentages.round(1).astype(str) + "%)"
# print(f'Total number of subjects: {sex_counts.sum()}')
print(sex_counts_with_percentages)
print()

sex_df = df_KI67
# ASTR_LGG sex distribution
print('Sex distribution for ASTR_LGG:')
sex_df_ASTR_LGG = sex_df[sex_df['label'] == 'ASTR_LGG']
sex_df_ASTR_LGG = sex_df_ASTR_LGG.drop_duplicates(subset=['case_id'])
sex_df_ASTR_LGG = sex_df_ASTR_LGG['sex'].value_counts()
print(sex_df_ASTR_LGG)
print()
# ASTR_HGG sex distribution
print('Sex distribution for ASTR_HGG:')
sex_df_ASTR_HGG = sex_df[sex_df['label'] == 'ASTR_HGG']
sex_df_ASTR_HGG = sex_df_ASTR_HGG.drop_duplicates(subset=['case_id'])
sex_counts_ASTR_HGG = sex_df_ASTR_HGG['sex'].value_counts()
print(sex_counts_ASTR_HGG)
print()
# MED sex distribution
print('Sex distribution for MED:')
sex_df_MED = sex_df[sex_df['label'] == 'MED']
sex_df_MED = sex_df_MED.drop_duplicates(subset=['case_id'])
sex_counts_MED = sex_df_MED['sex'].value_counts()
print(sex_counts_MED)
print()
# EP sex distribution
print('Sex distribution for EP:')
sex_df_EP = sex_df[sex_df['label'] == 'EP']
sex_df_EP = sex_df_EP.drop_duplicates(subset=['case_id'])
sex_counts_EP = sex_df_EP['sex'].value_counts()
print(sex_counts_EP)
print()
# GANG sex distribution
print('Sex distribution for GANG:')
sex_df_GANG = sex_df[sex_df['label'] == 'GANG']
sex_df_GANG = sex_df_GANG.drop_duplicates(subset=['case_id'])
sex_counts_GANG = sex_df_GANG['sex'].value_counts()
print(sex_counts_GANG)
print()
# MEN sex distribution
print('Sex distribution for MEN:')
sex_df_MEN = sex_df[sex_df['label'] == 'MEN']
sex_df_MEN = sex_df_MEN.drop_duplicates(subset=['case_id'])
sex_counts_MEN = sex_df_MEN['sex'].value_counts()
print(sex_counts_MEN)
print()
# ATRT sex distribution
print('Sex distribution for ATRT:')
sex_df_ATRT = sex_df[sex_df['label'] == 'ATRT']
sex_df_ATRT = sex_df_ATRT.drop_duplicates(subset=['case_id'])
sex_counts_ATRT = sex_df_ATRT['sex'].value_counts()
print(sex_counts_ATRT)
print()
# DNET sex distribution
print('Sex distribution for DNET:')
sex_df_DNET = sex_df[sex_df['label'] == 'DNET']
sex_df_DNET = sex_df_DNET.drop_duplicates(subset=['case_id'])
sex_counts_DNET = sex_df_DNET['sex'].value_counts()
print(sex_counts_DNET)
print()
# DIPG sex distribution
print('Sex distribution for DIPG:')
sex_df_DIPG = sex_df[sex_df['label'] == 'DIPG']
sex_df_DIPG = sex_df_DIPG.drop_duplicates(subset=['case_id'])
sex_counts_DIPG = sex_df_DIPG['sex'].value_counts()
print(sex_counts_DIPG)
print()
print('Age at diagnosis (days):')
mean_age = df_KI67['age_at_diagnosis_(days)'].mean()
std_age = df_KI67['age_at_diagnosis_(days)'].std()
print(f'Mean age: {mean_age:.2f} ± {std_age:.2f}')
print()

# AGE INSIGHTS
print('Age at diagnosis (years):')
df_KI67['age_at_diagnosis_(years)'] = df_KI67['age_at_diagnosis_(days)'] / 365.25
mean_age_years = df_KI67['age_at_diagnosis_(years)'].mean()
std_age_years = df_KI67['age_at_diagnosis_(years)'].std()
print(f'Mean age: {mean_age_years:.2f} ± {std_age_years:.2f}')

# %% DATA SUMMARY OF KI-67 ALIGNED WITH MRI
print('SUMMARY OF THE KI-67 DATASET ALIGNED WITH MRI')
print('Total number of subjects:')
print(df_KI67_aligned_with_MRI['case_id'].nunique())
print('Total number of images:')
print(df_KI67_aligned_with_MRI['slide_id'].nunique())

print()

unique_comb_case_id_label= df_KI67_aligned_with_MRI
i = 0
double_diagnosis_subjects = []
for subject in unique_comb_case_id_label['case_id'].unique():
    if unique_comb_case_id_label[unique_comb_case_id_label['case_id'] == subject]['label'].nunique() > 1:
        i += 1
        double_diagnosis_subjects.append(subject)

print(f'Total number of subjects with 2 or more unique labels: {i}')
print('Subjects with 2 or more unique labels:')
for subject in double_diagnosis_subjects:
    print(subject)

print()

print('Unique combinations of subjects and labels:')
print(df_KI67_aligned_with_MRI[['case_id', 'label']].drop_duplicates().shape[0])

print()

summary_table = df_KI67_aligned_with_MRI.groupby('label').agg(
    num_subjects=('case_id', 'nunique'),
    num_images=('slide_id', 'nunique')
).reset_index()

summary_table = summary_table.sort_values(by='num_subjects', ascending=False)

print('| Label | Number of Subjects | Number of Images |')
print('|-------|--------------------|------------------|')
for _, row in summary_table.iterrows():
    print(f"| {row['label']} | {row['num_subjects']} | {row['num_images']} |")
print(f"| Total | {summary_table['num_subjects'].sum()} | {summary_table['num_images'].sum()} |")

print()

print('Sex distribution:')
# filter by unique combinations of case_id and label
sex_df = df_KI67_aligned_with_MRI.drop_duplicates(subset=['case_id', 'label'])
sex_counts = sex_df['sex'].value_counts()
sex_percentages = (sex_counts / sex_counts.sum()) * 100
sex_counts_with_percentages = sex_counts.astype(str) + " (" + sex_percentages.round(1).astype(str) + "%)"
# print(f'Total number of subjects: {sex_counts.sum()}')
print(sex_counts_with_percentages)
print()

sex_df = df_KI67_aligned_with_MRI
# ASTR_LGG sex distribution
print('Sex distribution for ASTR_LGG:')
sex_df_ASTR_LGG = sex_df[sex_df['label'] == 'ASTR_LGG']
sex_df_ASTR_LGG = sex_df_ASTR_LGG.drop_duplicates(subset=['case_id'])
sex_df_ASTR_LGG = sex_df_ASTR_LGG['sex'].value_counts()
print(sex_df_ASTR_LGG)
print()
# ASTR_HGG sex distribution
print('Sex distribution for ASTR_HGG:')
sex_df_ASTR_HGG = sex_df[sex_df['label'] == 'ASTR_HGG']
sex_df_ASTR_HGG = sex_df_ASTR_HGG.drop_duplicates(subset=['case_id'])
sex_counts_ASTR_HGG = sex_df_ASTR_HGG['sex'].value_counts()
print(sex_counts_ASTR_HGG)
print()
# MED sex distribution
print('Sex distribution for MED:')
sex_df_MED = sex_df[sex_df['label'] == 'MED']
sex_df_MED = sex_df_MED.drop_duplicates(subset=['case_id'])
sex_counts_MED = sex_df_MED['sex'].value_counts()
print(sex_counts_MED)
print()
# EP sex distribution
print('Sex distribution for EP:')
sex_df_EP = sex_df[sex_df['label'] == 'EP']
sex_df_EP = sex_df_EP.drop_duplicates(subset=['case_id'])
sex_counts_EP = sex_df_EP['sex'].value_counts()
print(sex_counts_EP)
print()
# GANG sex distribution
print('Sex distribution for GANG:')
sex_df_GANG = sex_df[sex_df['label'] == 'GANG']
sex_df_GANG = sex_df_GANG.drop_duplicates(subset=['case_id'])
sex_counts_GANG = sex_df_GANG['sex'].value_counts()
print(sex_counts_GANG)
print()
# MEN sex distribution
print('Sex distribution for MEN:')
sex_df_MEN = sex_df[sex_df['label'] == 'MEN']
sex_df_MEN = sex_df_MEN.drop_duplicates(subset=['case_id'])
sex_counts_MEN = sex_df_MEN['sex'].value_counts()
print(sex_counts_MEN)
print()
# ATRT sex distribution
print('Sex distribution for ATRT:')
sex_df_ATRT = sex_df[sex_df['label'] == 'ATRT']
sex_df_ATRT = sex_df_ATRT.drop_duplicates(subset=['case_id'])
sex_counts_ATRT = sex_df_ATRT['sex'].value_counts()
print(sex_counts_ATRT)
print()
# DNET sex distribution
print('Sex distribution for DNET:')
sex_df_DNET = sex_df[sex_df['label'] == 'DNET']
sex_df_DNET = sex_df_DNET.drop_duplicates(subset=['case_id'])
sex_counts_DNET = sex_df_DNET['sex'].value_counts()
print(sex_counts_DNET)
print()
# DIPG sex distribution
print('Sex distribution for DIPG:')
sex_df_DIPG = sex_df[sex_df['label'] == 'DIPG']
sex_df_DIPG = sex_df_DIPG.drop_duplicates(subset=['case_id'])
sex_counts_DIPG = sex_df_DIPG['sex'].value_counts()
print(sex_counts_DIPG)

print()
print('Age at diagnosis (days):')
mean_age = df_KI67['age_at_diagnosis_(days)'].mean()
std_age = df_KI67['age_at_diagnosis_(days)'].std()
print(f'Mean age: {mean_age:.2f} ± {std_age:.2f}')
print()

# AGE INSIGHTS
print('Age at diagnosis (years):')
df_KI67_aligned_with_MRI['age_at_diagnosis_(years)'] = df_KI67_aligned_with_MRI['age_at_diagnosis_(days)'] / 365.25
mean_age_years = df_KI67_aligned_with_MRI['age_at_diagnosis_(years)'].mean()
std_age_years = df_KI67_aligned_with_MRI['age_at_diagnosis_(years)'].std()
print(f'Mean age: {mean_age_years:.2f} ± {std_age_years:.2f}')

# %% DATA SUMMARY OF KI-67 WITH EXCLUSIONS
print('SUMMARY OF THE KI-67 RESULTS (EXCLUSIONS ARE NOT INCLUDED)')
final_results = final_results[final_results['Quality'] != 'Exclude']

print('Total number of subjects:')
print(final_results['case_id'].nunique())

print()

print('Total number of images:')
print(final_results['slide_id'].nunique())

print()

unique_comb_case_id_label= final_results
i = 0
double_diagnosis_subjects = []
for subject in unique_comb_case_id_label['case_id'].unique():
    if unique_comb_case_id_label[unique_comb_case_id_label['case_id'] == subject]['label'].nunique() > 1:
        i += 1
        double_diagnosis_subjects.append(subject)

print(f'Total number of subjects with 2 or more unique labels: {i}')
print('Subjects with 2 or more unique labels:')
for subject in double_diagnosis_subjects:
    print(subject)

print()

print('Unique combinations of subjects and labels:')
print(final_results[['case_id', 'label']].drop_duplicates().shape[0])

print()

summary_table = final_results.groupby('label').agg(
    num_subjects=('case_id', 'nunique'),
    num_images=('slide_id', 'nunique')
).reset_index()

summary_table = summary_table.sort_values(by='num_subjects', ascending=False)

print('| Label | Number of Subjects | Number of Images |')
print('|-------|--------------------|------------------|')
for _, row in summary_table.iterrows():
    print(f"| {row['label']} | {row['num_subjects']} | {row['num_images']} |")
print(f"| Total | {summary_table['num_subjects'].sum()} | {summary_table['num_images'].sum()} |")

print()

print('Sex distribution:')
# filter by unique combinations of case_id and label
sex_df = final_results.drop_duplicates(subset=['case_id', 'label'])
sex_counts = sex_df['sex'].value_counts()
sex_percentages = (sex_counts / sex_counts.sum()) * 100
sex_counts_with_percentages = sex_counts.astype(str) + " (" + sex_percentages.round(1).astype(str) + "%)"
# print(f'Total number of subjects: {sex_counts.sum()}')
print(sex_counts_with_percentages)
print()

sex_df = final_results
# ASTR_LGG sex distribution
print('Sex distribution for ASTR_LGG:')
sex_df_ASTR_LGG = sex_df[sex_df['label'] == 'ASTR_LGG']
sex_df_ASTR_LGG = sex_df_ASTR_LGG.drop_duplicates(subset=['case_id'])
sex_df_ASTR_LGG = sex_df_ASTR_LGG['sex'].value_counts()
print(sex_df_ASTR_LGG)
print()
# ASTR_HGG sex distribution
print('Sex distribution for ASTR_HGG:')
sex_df_ASTR_HGG = sex_df[sex_df['label'] == 'ASTR_HGG']
sex_df_ASTR_HGG = sex_df_ASTR_HGG.drop_duplicates(subset=['case_id'])
sex_counts_ASTR_HGG = sex_df_ASTR_HGG['sex'].value_counts()
print(sex_counts_ASTR_HGG)
print()
# MED sex distribution
print('Sex distribution for MED:')
sex_df_MED = sex_df[sex_df['label'] == 'MED']
sex_df_MED = sex_df_MED.drop_duplicates(subset=['case_id'])
sex_counts_MED = sex_df_MED['sex'].value_counts()
print(sex_counts_MED)
print()
# EP sex distribution
print('Sex distribution for EP:')
sex_df_EP = sex_df[sex_df['label'] == 'EP']
sex_df_EP = sex_df_EP.drop_duplicates(subset=['case_id'])
sex_counts_EP = sex_df_EP['sex'].value_counts()
print(sex_counts_EP)
print()
# GANG sex distribution
print('Sex distribution for GANG:')
sex_df_GANG = sex_df[sex_df['label'] == 'GANG']
sex_df_GANG = sex_df_GANG.drop_duplicates(subset=['case_id'])
sex_counts_GANG = sex_df_GANG['sex'].value_counts()
print(sex_counts_GANG)
print()
# MEN sex distribution
print('Sex distribution for MEN:')
sex_df_MEN = sex_df[sex_df['label'] == 'MEN']
sex_df_MEN = sex_df_MEN.drop_duplicates(subset=['case_id'])
sex_counts_MEN = sex_df_MEN['sex'].value_counts()
print(sex_counts_MEN)
print()
# ATRT sex distribution
print('Sex distribution for ATRT:')
sex_df_ATRT = sex_df[sex_df['label'] == 'ATRT']
sex_df_ATRT = sex_df_ATRT.drop_duplicates(subset=['case_id'])
sex_counts_ATRT = sex_df_ATRT['sex'].value_counts()
print(sex_counts_ATRT)
print()
# DNET sex distribution
print('Sex distribution for DNET:')
sex_df_DNET = sex_df[sex_df['label'] == 'DNET']
sex_df_DNET = sex_df_DNET.drop_duplicates(subset=['case_id'])
sex_counts_DNET = sex_df_DNET['sex'].value_counts()
print(sex_counts_DNET)
print()
# DIPG sex distribution
print('Sex distribution for DIPG:')
sex_df_DIPG = sex_df[sex_df['label'] == 'DIPG']
sex_df_DIPG = sex_df_DIPG.drop_duplicates(subset=(['case_id']))
sex_counts_DIPG = sex_df_DIPG['sex'].value_counts()
print(sex_counts_DIPG)

print()
print('Age at diagnosis (days):')
mean_age = final_results['age_at_diagnosis_(days)'].mean()
std_age = final_results['age_at_diagnosis_(days)'].std()
print(f'Mean age: {mean_age:.2f} ± {std_age:.2f}')
print()

# AGE INSIGHTS
print('Age at diagnosis (years):')
final_results['age_at_diagnosis_(years)'] = final_results['age_at_diagnosis_(days)'] / 365.25
mean_age_years = final_results['age_at_diagnosis_(years)'].mean()
std_age_years = final_results['age_at_diagnosis_(years)'].std()
print(f'Mean age: {mean_age_years:.2f} ± {std_age_years:.2f}')

# %% AGE DISTRIBUTION
df_KI67 = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/CBTN_KI67.csv')
df_KI67 = df_KI67[['case_id', 'label', 'age_at_diagnosis_(days)', 'sex']].drop_duplicates()
df_KI67['age_at_diagnosis_(years)'] = df_KI67['age_at_diagnosis_(days)'] / 365.25

# density estimates of age_at_diagnosis_(years) male and female side by side
fig, axes = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
sns.kdeplot(
    data=df_KI67, 
    x='age_at_diagnosis_(years)', 
    color='darkgrey', 
    fill=True, 
    alpha=0.6, 
    linewidth=0, 
    ax=axes[0]
)
sns.kdeplot(
    data=df_KI67[df_KI67['sex'] == 'Male'], 
    x='age_at_diagnosis_(years)', 
    fill=True, 
    alpha=0.5, 
    color='cornflowerblue', 
    linewidth=0, 
    ax=axes[0]
)
axes[0].set_title('')
axes[0].set_xlabel('')
axes[0].set_ylabel('')
axes[0].set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
axes[0].set_xticklabels(['0', '5', '10', '15', '20', '25', '30', '35', '40'], fontsize=20)
axes[0].set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
axes[0].set_yticklabels(['0', '10', '20', '30', '40', '50'], fontsize=20)
axes[0].set_xlim(left=0, right=40)
axes[0].grid(axis='y', linestyle='-', alpha=0.8)
axes[0].set_facecolor('white')
for spine in axes[0].spines.values():
    spine.set_visible(False)

# Female plot
sns.kdeplot(
    data=df_KI67, 
    x='age_at_diagnosis_(years)', 
    color='darkgrey', 
    fill=True, 
    alpha=0.6, 
    linewidth=0, 
    ax=axes[1]
)
sns.kdeplot(
    data=df_KI67[df_KI67['sex'] == 'Female'], 
    x='age_at_diagnosis_(years)', 
    fill=True, 
    alpha=0.5, 
    color='hotpink', 
    linewidth=0, 
    ax=axes[1]
)
axes[1].set_title('', fontsize=16)
axes[1].set_xlabel('', fontsize=14)
axes[1].set_ylabel('', fontsize=14)
axes[1].set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
axes[1].set_xticklabels(['0', '5', '10', '15', '20', '25', '30', '35', '40'], fontsize=20)
axes[1].set_xlim(left=0)
axes[1].grid(axis='y', linestyle='-', alpha=0.8)
axes[1].set_facecolor('white')
for spine in axes[1].spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig('age_at_diagnosis_density_male_female_side_by_side.svg', dpi=600, bbox_inches='tight')
plt.show()

# %%
# overlapping density plot for age at diagnosis (years) by sex
plt.figure(figsize=(16, 6))
sns.kdeplot(
    data=df_KI67, 
    x='age_at_diagnosis_(years)', 
    hue='sex', 
    fill=True, 
    alpha=0.5, 
    palette={'Male': 'cornflowerblue', 'Female': 'hotpink'},
    linewidth=0
)
plt.title('')
plt.legend('')
plt.xlabel('')
plt.xticks(fontsize=16)
plt.xlim(left=0, right=40)
plt.ylabel('')
plt.yticks([0, 0.005, 0.010, 0.015, 0.020, 0.025], ['0', '5', '10', '15', '20', '25'], fontsize=16)
plt.grid(axis='y', linestyle='-', alpha=0.8)
plt.gca().set_facecolor('white')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig('age_at_diagnosis_density_by_sex.svg', dpi=600, bbox_inches='tight')
plt.show()

# %%
sns.kdeplot(
    data=df_KI67[df_KI67['sex'] == 'Male'], 
    x='age_at_diagnosis_(years)', 
    fill=True, 
    alpha=0.5, 
    color='cornflowerblue', 
    linewidth=0, 
    ax=axes[0]
)

sns.kdeplot(
    data=df_KI67[df_KI67['sex'] == 'Male'], 
    x='age_at_diagnosis_(years)', 
    fill=True, 
    alpha=0.5, 
    color='cornflowerblue', 
    linewidth=0, 
    ax=axes[0]
)

sns.kdeplot(
    data=df_KI67, 
    x='age_at_diagnosis_(years)', 
    color='darkgrey', 
    fill=True, 
    alpha=0.6, 
    linewidth=0, 
    ax=axes[1]
)


# %% RACE INSIGHTS
df_KI67 = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/CBTN_KI67.csv')
df_KI67 = df_KI67[['case_id', 'label', 'race']].drop_duplicates()

print('Race distribution:')
race_counts = df_KI67['race'].value_counts()
race_percentages = (race_counts / race_counts.sum()) * 100
race_counts_with_percentages = race_counts.astype(str) + " (" + race_percentages.round(1).astype(str) + "%)"
print(race_counts_with_percentages)

custom_colors = sns.color_palette("Set2", len(race_counts))

# bar plot 
sns.set_style("white") 
race_counts_sorted = race_counts.sort_values(ascending=True)
race_percentages_sorted = (race_counts_sorted / race_counts.sum()) * 100
custom_colors_sorted = [custom_colors[race_counts.index.get_loc(idx)] for idx in race_counts_sorted.index]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(race_counts_sorted.index, race_counts_sorted, color=custom_colors_sorted)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')
plt.gca().set_facecolor('white')

for bar, percentage in zip(bars, race_percentages_sorted):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height() / 2, f"{width} ({percentage:.1f}%)", ha='left', va='center', fontsize=10)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.xticks([])
plt.savefig('race_distribution.svg', dpi=600, bbox_inches='tight')
plt.show()

# %% TUMOR DESCRIPTOR DISTRIBUTION
final_results = pd.read_excel('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/QuPath_Ki-67_summary_analysis.xlsx')
final_results = final_results[['case_id', 'label', 'tumor_descriptor']].drop_duplicates()

print('Tumor descriptor distribution:')
tumor_descriptor_counts = final_results['tumor_descriptor'].value_counts()
tumor_descriptor_percentages = (tumor_descriptor_counts / tumor_descriptor_counts.sum()) * 100
tumor_descriptor_counts_with_percentages = tumor_descriptor_counts.astype(str) + " (" + tumor_descriptor_percentages.round(1).astype(str) + "%)"
print(tumor_descriptor_counts_with_percentages)
print()

print('Subjects with 2 or more tumor descriptors:')
i = 0
# subjects with 2 or more tumor descriptors 
subjects_with_multiple_descriptors = []
for subject in final_results['case_id'].unique():
    descriptor_count = final_results[final_results['case_id'] == subject]['tumor_descriptor'].nunique()
    if descriptor_count > 1:
        i += 1
        subjects_with_multiple_descriptors.append((subject, descriptor_count))
print(f'Total number of subjects with 2 or more tumor descriptors: {i}')
print()

# Print table with case_id and tumor descriptor count
print('| Case ID | Number of Tumor Descriptors |')
print('|---------|-----------------------------|')
for subject, count in subjects_with_multiple_descriptors:
    print(f'| {subject} | {count} |')
print()

# ASTR_LGG tumor descriptor distribution
ASTR_LGG_tumor_descriptor_counts = final_results[final_results['label'] == 'ASTR_LGG']['tumor_descriptor'].value_counts()
ASTR_LGG_tumor_descriptor_percentages = (ASTR_LGG_tumor_descriptor_counts / ASTR_LGG_tumor_descriptor_counts.sum()) * 100
ASTR_LGG_tumor_descriptor_counts_with_percentages = ASTR_LGG_tumor_descriptor_counts.astype(str) + " (" + ASTR_LGG_tumor_descriptor_percentages.round(1).astype(str) + "%)"
print('ASTR_LGG tumor descriptor distribution:')
print(ASTR_LGG_tumor_descriptor_counts_with_percentages)
print()

# ASTR_HGG tumor descriptor distribution
ASTR_HGG_tumor_descriptor_counts = final_results[final_results['label'] == 'ASTR_HGG']['tumor_descriptor'].value_counts()
ASTR_HGG_tumor_descriptor_percentages = (ASTR_HGG_tumor_descriptor_counts / ASTR_HGG_tumor_descriptor_counts.sum()) * 100
ASTR_HGG_tumor_descriptor_counts_with_percentages = ASTR_HGG_tumor_descriptor_counts.astype(str) + " (" + ASTR_HGG_tumor_descriptor_percentages.round(1).astype(str) + "%)"
print('ASTR_HGG tumor descriptor distribution:')
print(ASTR_HGG_tumor_descriptor_counts_with_percentages)
print()

# MED tumor descriptor distribution
MED_tumor_descriptor_counts = final_results[final_results['label'] == 'MED']['tumor_descriptor'].value_counts()
MED_tumor_descriptor_percentages = (MED_tumor_descriptor_counts / MED_tumor_descriptor_counts.sum()) * 100
MED_tumor_descriptor_counts_with_percentages = MED_tumor_descriptor_counts.astype(str) + " (" + MED_tumor_descriptor_percentages.round(1).astype(str) + "%)"
print('MED tumor descriptor distribution:')
print(MED_tumor_descriptor_counts_with_percentages)
print()

# EP tumor descriptor distribution
EP_tumor_descriptor_counts = final_results[final_results['label'] == 'EP']['tumor_descriptor'].value_counts()
EP_tumor_descriptor_percentages = (EP_tumor_descriptor_counts / EP_tumor_descriptor_counts.sum()) * 100
EP_tumor_descriptor_counts_with_percentages = EP_tumor_descriptor_counts.astype(str) + " (" + EP_tumor_descriptor_percentages.round(1).astype(str) + "%)"
print('EP tumor descriptor distribution:')
print(EP_tumor_descriptor_counts_with_percentages)
print()

# GANG tumor descriptor distribution
GANG_tumor_descriptor_counts = final_results[final_results['label'] == 'GANG']['tumor_descriptor'].value_counts()
GANG_tumor_descriptor_percentages = (GANG_tumor_descriptor_counts / GANG_tumor_descriptor_counts.sum()) * 100
GANG_tumor_descriptor_counts_with_percentages = GANG_tumor_descriptor_counts.astype(str) + " (" + GANG_tumor_descriptor_percentages.round(1).astype(str) + "%)"
print('GANG tumor descriptor distribution:')
print(GANG_tumor_descriptor_counts_with_percentages)
print()

# MEN tumor descriptor distribution
MEN_tumor_descriptor_counts = final_results[final_results['label'] == 'MEN']['tumor_descriptor'].value_counts()
MEN_tumor_descriptor_percentages = (MEN_tumor_descriptor_counts / MEN_tumor_descriptor_counts.sum()) * 100
MEN_tumor_descriptor_counts_with_percentages = MEN_tumor_descriptor_counts.astype(str) + " (" + MEN_tumor_descriptor_percentages.round(1).astype(str) + "%)"
print('MEN tumor descriptor distribution:')
print(MEN_tumor_descriptor_counts_with_percentages)
print()

# ATRT tumor descriptor distribution
ATRT_tumor_descriptor_counts = final_results[final_results['label'] == 'ATRT']['tumor_descriptor'].value_counts()
ATRT_tumor_descriptor_percentages = (ATRT_tumor_descriptor_counts / ATRT_tumor_descriptor_counts.sum()) * 100
ATRT_tumor_descriptor_counts_with_percentages = ATRT_tumor_descriptor_counts.astype(str) + " (" + ATRT_tumor_descriptor_percentages.round(1).astype(str) + "%)"
print('ATRT tumor descriptor distribution:')
print(ATRT_tumor_descriptor_counts_with_percentages)
print()

# DNET tumor descriptor distribution
DNET_tumor_descriptor_counts = final_results[final_results['label'] == 'DNET']['tumor_descriptor'].value_counts()
DNET_tumor_descriptor_percentages = (DNET_tumor_descriptor_counts / DNET_tumor_descriptor_counts.sum()) * 100
DNET_tumor_descriptor_counts_with_percentages = DNET_tumor_descriptor_counts.astype(str) + " (" + DNET_tumor_descriptor_percentages.round(1).astype(str) + "%)"
print('DNET tumor descriptor distribution:')
print(DNET_tumor_descriptor_counts_with_percentages)
print()

# DIPG tumor descriptor distribution
DIPG_tumor_descriptor_counts = final_results[final_results['label'] == 'DIPG']['tumor_descriptor'].value_counts()
DIPG_tumor_descriptor_percentages = (DIPG_tumor_descriptor_counts / DIPG_tumor_descriptor_counts.sum()) * 100
DIPG_tumor_descriptor_counts_with_percentages = DIPG_tumor_descriptor_counts.astype(str) + " (" + DIPG_tumor_descriptor_percentages.round(1).astype(str) + "%)"
print('DIPG tumor descriptor distribution:')
print(DIPG_tumor_descriptor_counts_with_percentages)
print()

# bar plot 
custom_colors = sns.color_palette("Set2", len(tumor_descriptor_counts))

sns.set_style("white") 
tumor_descriptor_counts_sorted = tumor_descriptor_counts.sort_values(ascending=False)
tumor_descriptor_percentages_sorted = (tumor_descriptor_counts_sorted / tumor_descriptor_counts.sum()) * 100
custom_colors_sorted = [custom_colors[tumor_descriptor_counts.index.get_loc(idx)] for idx in tumor_descriptor_counts_sorted.index]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(tumor_descriptor_counts_sorted.index, tumor_descriptor_counts_sorted, color=custom_colors_sorted)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')
plt.gca().set_facecolor('white')

for bar, percentage in zip(bars, tumor_descriptor_percentages_sorted):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height} ({percentage:.1f}%)", ha='center', va='bottom', fontsize=10)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.yticks([])
plt.savefig('tumor_descriptor_distribution.svg', dpi=600, bbox_inches='tight')
plt.show()

initial_cns_results = final_results[final_results['tumor_descriptor'] == 'Initial CNS Tumor']
initial_cns_results['label'] = initial_cns_results['label'].replace({
    'ATRT': 'ATRT_initial_CNS',
    'MED': 'MED_initial_CNS',
    'DIPG': 'DIPG_initial_CNS',
    'ASTR_HGG': 'ASTR_HGG_initial_CNS',
    'EP': 'EP_initial_CNS',
    'ASTR_LGG': 'ASTR_LGG_initial_CNS',
    'GANG': 'GANG_initial_CNS',
    'MEN': 'MEN_initial_CNS',
    'DNET': 'DNET_initial_CNS'
})

not_available_results = final_results[final_results['tumor_descriptor'] == 'Not Available']
not_available_results['label'] = not_available_results['label'].replace({
    'ATRT': 'ATRT_not_available',
    'MED': 'MED_not_available',
    'DIPG': 'DIPG_not_available',
    'ASTR_HGG': 'ASTR_HGG_not_available',
    'EP': 'EP_not_available',
    'ASTR_LGG': 'ASTR_LGG_not_available',
    'GANG': 'GANG_not_available',
    'MEN': 'MEN_not_available',
    'DNET': 'DNET_not_available'
})

progressive_results = final_results[final_results['tumor_descriptor'] == 'Progressive']
progressive_results['label'] = progressive_results['label'].replace({
    'MED': 'MED_progressive',
    'ATRT': 'ATRT_progressive',
    'ASTR_HGG': 'ASTR_HGG_progressive',
    'EP': 'EP_progressive',
    'ASTR_LGG': 'ASTR_LGG_progressive',
    'GANG': 'GANG_progressive',
    'DNET': 'DNET_progressive',
    'DIPG': 'DIPG_progressive',
    'MEN': 'MEN_progressive'
})

recurrence_results = final_results[final_results['tumor_descriptor'] == 'Recurrence']
recurrence_results['label'] = recurrence_results['label'].replace({
    'EP': 'EP_recurrence',
    'MED': 'MED_recurrence',
    'ASTR_HGG': 'ASTR_HGG_recurrence',
    'ASTR_LGG': 'ASTR_LGG_recurrence',
    'GANG': 'GANG_recurrence',
    'MEN': 'MEN_recurrence',
    'DIPG': 'DIPG_recurrence',
    'ATRT': 'ATRT_recurrence',
    'DNET': 'DNET_recurrence'
})

second_malignancy_results = final_results[final_results['tumor_descriptor'] == 'Second Malignancy']
second_malignancy_results['label'] = second_malignancy_results['label'].replace({
    'ASTR_HGG': 'ASTR_HGG_second_malignancy',
    'MEN': 'MEN_second_malignancy',
    'ASTR_LGG': 'ASTR_LGG_second_malignancy',
    'DIPG': 'DIPG_second_malignancy',
    'ATRT': 'ATRT_second_malignancy',
    'DNET': 'DNET_second_malignancy',
    'EP': 'EP_second_malignancy',
    'GANG': 'GANG_second_malignancy',
    'MED': 'MED_second_malignancy'
})

combined_tumor_descriptors = pd.concat([
    initial_cns_results,
    progressive_results,
    recurrence_results,
    second_malignancy_results,
    not_available_results
    ], ignore_index=True)

# bar plot
palette = sns.color_palette("dark", 10)  

astr_lgg_color = palette[0]
astr_hgg_color = palette[1]
med_color = palette[2]
ep_color = palette[3]
gang_color = palette[4]
men_color = palette[5]
atrt_color = palette[6]
dnet_color = palette[7]
dipg_color = palette[8]


custom_colors = [
    # initial CNS
    # 186           70         52              48        32          16          9
    astr_lgg_color, med_color, astr_hgg_color, ep_color, gang_color, atrt_color, dipg_color,
    # progressive
    # 53            17              9         7           3           2          1
    astr_lgg_color, astr_hgg_color, ep_color, gang_color, atrt_color, med_color, dnet_color,
    # recurrence
    # 16            8               8         5          3
    astr_lgg_color, astr_hgg_color, ep_color, med_color, gang_color,
    # second malignancy
    # 5        4               3
    men_color, astr_hgg_color, astr_lgg_color,
    # not available
    # 27            19         13          12        12              11          6          2           2
    astr_lgg_color, men_color, dnet_color, ep_color, astr_hgg_color, gang_color, med_color, atrt_color, dipg_color
]
palette = sns.color_palette(custom_colors)

order = [
    'ASTR_LGG_initial_CNS', 'MED_initial_CNS', 'ASTR_HGG_initial_CNS', 'EP_initial_CNS', 'GANG_initial_CNS', 'ATRT_initial_CNS', 'DIPG_initial_CNS',
    'ASTR_LGG_progressive', 'ASTR_HGG_progressive', 'EP_progressive', 'GANG_progressive', 'ATRT_progressive', 'MED_progressive', 'DNET_progressive',
    'ASTR_LGG_recurrence', 'ASTR_HGG_recurrence', 'EP_recurrence', 'MED_recurrence', 'GANG_recurrence',
    'MEN_second_malignancy', 'ASTR_HGG_second_malignancy', 'ASTR_LGG_second_malignancy',
    'ASTR_LGG_not_available', 'MEN_not_available', 'DNET_not_available', 'EP_not_available', 'ASTR_HGG_not_available', 'GANG_not_available', 'MED_not_available', 'ATRT_not_available', 'DIPG_not_available'
]

label_counts = combined_tumor_descriptors['label'].value_counts()
ordered_counts = label_counts.reindex(order).fillna(0)

gap_after_labels = [
    'DIPG_initial_CNS',
    'DNET_progressive',
    'GANG_recurrence',
    'ASTR_LGG_second_malignancy'
]
x_positions = []
x = 0
for label in order:
    x_positions.append(x)
    x += 1
    if label in gap_after_labels:
        x += 0.6 

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(x_positions, ordered_counts.values, color=custom_colors)
for bar, count in zip(bars, ordered_counts.values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f"{int(count)}", ha='center', va='bottom', fontsize=10)
    
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')
plt.gca().set_facecolor('white')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(x_positions)
ax.set_xticklabels(order, rotation=45, ha='right')

plt.tight_layout()
plt.xticks([])
plt.yticks([25, 50, 75, 100, 125, 150, 175, 200], ['25', '50', '75', '100', '125', '150', '175', '200'])
plt.savefig('per_label_tumor_descriptors.svg', dpi=600, bbox_inches='tight')
plt.show()

# %% 2 OR MORE DIAGNOSES
# subjects with 2 or more unique labels
print('Subjects with 2 or more unique labels:')
i = 0
for subject in df_KI67['case_id'].unique():
    if df_KI67[df_KI67['case_id'] == subject]['label'].nunique() > 1:
        i += 1
        print(subject)

print(f'Total number of subjects with 2 or more unique labels: {i}')

# %% SUBJECTS WITH 2 OR TUMOR DESCRIPTORS


# %% SUBJECTS WITH INITIAL CNS TUMOR
print('Subjects with initial CNS tumor:')
i = 0
# subjects with initial CNS tumor
for subject in df_KI67_aligned_with_MRI['case_id'].unique():
    if df_KI67_aligned_with_MRI.loc[df_KI67_aligned_with_MRI['case_id'] == subject, 'tumor_descriptor'].eq('Initial CNS Tumor').any():
        i += 1
        print(subject)
    
print(f'Total number of subjects with initial CNS tumor: {i}')

# %% SUBJECTS WITH SECOND MALIGNANCY TUMOR
print('Subjects with second malignacy:')
i = 0
# subjects with second malignancy
for subject in df_KI67_aligned_with_MRI['case_id'].unique():
    if df_KI67_aligned_with_MRI.loc[df_KI67_aligned_with_MRI['case_id'] == subject, 'tumor_descriptor'].eq('Second Malignancy').any():
        i += 1
        print(subject)
    
print(f'Total number of subjects with second malignacy: {i}')

# %% SUBJECTS WITH PROGRESSIVE TUMOR
print('Subjects with progressive:')
i = 0
# subjects with progressive
for subject in df_KI67_aligned_with_MRI['case_id'].unique():
    if df_KI67_aligned_with_MRI.loc[df_KI67_aligned_with_MRI['case_id'] == subject, 'tumor_descriptor'].eq('Progressive').any():
        i += 1
        print(subject)
    
print(f'Total number of subjects with progressive: {i}')

# %% SUBJECTS WITH RECURRENCE TUMOR
print('Subjects with recurrence:')
i = 0
# subjects with recurrence
for subject in df_KI67_aligned_with_MRI['case_id'].unique():
    if df_KI67_aligned_with_MRI.loc[df_KI67_aligned_with_MRI['case_id'] == subject, 'tumor_descriptor'].eq('Recurrence').any():
        i += 1
        print(subject)
    
print(f'Total number of subjects with recurrence: {i}')

# %%