# %% IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% LOAD CSV FILES
df_KI67 = pd.read_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Data_Files/CBTN_KI67.csv')

# %% DATA SUMMARY
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

# %% TABLE OF CONTENTS
markdown_table = merged_df_KI67.to_markdown(index=False)

print(markdown_table)

# %% BAR PLOT

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

# %%