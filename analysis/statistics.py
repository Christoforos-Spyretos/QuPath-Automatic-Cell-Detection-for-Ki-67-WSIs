# %% IMPORTS
import pandas as pd
from scipy.stats import kruskal

# %% LOAD DATA
summary_results = pd.read_excel('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/QuPath_Ki-67_summary_analysis.xlsx')

# %% EXCLUDE CASES
# remove the excluded cases
summary_results = summary_results[summary_results['Quality'] != 'Exclude']

# %% STATISTICS
# calculate the mean, standard deviation, median, max and min values for each label

# cell density
cell_density_statistics = summary_results.groupby('label').agg( 
    median=('Density', 'median'),
    mean=('Density', 'mean'),
    std=('Density', 'std'),
    max=('Density', 'max'),
    min=('Density', 'min')
).reset_index()
cell_density_statistics = cell_density_statistics.sort_values(by='mean', ascending=False)
cell_density_statistics = cell_density_statistics[['label', 'median', 'mean', 'std', 'max', 'min']]
cell_density_statistics['variable'] = 'Cell_Density'
cell_density_statistics = cell_density_statistics[['variable', 'label', 'median', 'mean', 'std', 'max', 'min']]
cell_density_statistics[[ 'median', 'mean', 'std', 'max', 'min']] = cell_density_statistics[['median', 'mean', 'std', 'max', 'min']].round(2)

# number of positive cells
pos_cells_statistics = summary_results.groupby('label').agg(
    median=('Positive', 'median'),
    mean=('Positive', 'mean'),
    std=('Positive', 'std'),
    max=('Positive', 'max'),
    min=('Positive', 'min')
).reset_index()
pos_cells_statistics = pos_cells_statistics.sort_values(by='mean', ascending=False)
pos_cells_statistics = pos_cells_statistics[['label', 'median', 'mean', 'std', 'max', 'min']]
pos_cells_statistics['variable'] = 'Positive_Cells'
pos_cells_statistics = pos_cells_statistics[['variable', 'label', 'median', 'mean', 'std', 'max', 'min']]
pos_cells_statistics[['median', 'mean', 'std', 'max', 'min']] = pos_cells_statistics[['median', 'mean', 'std', 'max', 'min']].round(2)

# positive cell density
pos_cell_density_statistics = summary_results.groupby('label').agg(
    median=('Pos_Density', 'median'),
    mean=('Pos_Density', 'mean'),
    std=('Pos_Density', 'std'),
    max=('Pos_Density', 'max'),
    min=('Pos_Density', 'min')
).reset_index()
pos_cell_density_statistics = pos_cell_density_statistics.sort_values(by='mean', ascending=False)
pos_cell_density_statistics = pos_cell_density_statistics[['label', 'median', 'mean', 'std', 'max', 'min']]
pos_cell_density_statistics['variable'] = 'Positive_Cells_Density'
pos_cell_density_statistics = pos_cell_density_statistics[['variable', 'label', 'median', 'mean', 'std', 'max', 'min']]
pos_cell_density_statistics[['median', 'mean', 'std', 'max', 'min']] = pos_cell_density_statistics[['median', 'mean', 'std', 'max', 'min']].round(2)

# number of negative cells
neg_cells_statistics = summary_results.groupby('label').agg(
    median=('Negative', 'median'),
    mean=('Negative', 'mean'),
    std=('Negative', 'std'),
    max=('Negative', 'max'),
    min=('Negative', 'min')
).reset_index()
neg_cells_statistics = neg_cells_statistics.sort_values(by='mean', ascending=False)
neg_cells_statistics = neg_cells_statistics[['label', 'median', 'mean', 'std', 'max', 'min']]
neg_cells_statistics['variable'] = 'Negative_Cells'
neg_cells_statistics = neg_cells_statistics[['variable', 'label', 'median', 'mean', 'std', 'max', 'min']]
neg_cells_statistics[['median', 'mean', 'std', 'max', 'min']] = neg_cells_statistics[['median', 'mean', 'std', 'max', 'min']].round(2)

# negative cell density
neg_cell_density_statistics = summary_results.groupby('label').agg(
    median=('Neg_Density', 'median'),
    mean=('Neg_Density', 'mean'),
    std=('Neg_Density', 'std'),
    max=('Neg_Density', 'max'),
    min=('Neg_Density', 'min')
).reset_index()
neg_cell_density_statistics = neg_cell_density_statistics.sort_values(by='mean', ascending=False)
neg_cell_density_statistics = neg_cell_density_statistics[['label', 'median', 'mean', 'std', 'max', 'min']]
neg_cell_density_statistics['variable'] = 'Negative_Cells_Density'
neg_cell_density_statistics = neg_cell_density_statistics[['variable', 'label', 'median', 'mean', 'std', 'max', 'min']]
neg_cell_density_statistics[['median', 'mean', 'std', 'max', 'min']] = neg_cell_density_statistics[['median', 'mean', 'std', 'max', 'min']].round(2)

# Ki67 LI
ki67_LI_statistics = summary_results.groupby('label').agg(
    median=('Pos_Percentage', 'median'),
    mean=('Pos_Percentage', 'mean'),
    std=('Pos_Percentage', 'std'),
    max=('Pos_Percentage', 'max'),
    min=('Pos_Percentage', 'min')
).reset_index()
ki67_LI_statistics = ki67_LI_statistics.sort_values(by='mean', ascending=False)
ki67_LI_statistics = ki67_LI_statistics[['label', 'median', 'mean', 'std', 'max', 'min']]
ki67_LI_statistics['variable'] = 'Ki67_LI'
ki67_LI_statistics = ki67_LI_statistics[['variable', 'label', 'median', 'mean', 'std', 'max', 'min']]
ki67_LI_statistics[['median', 'mean', 'std', 'max', 'min']] = ki67_LI_statistics[['median', 'mean', 'std', 'max', 'min']].round(2)

statistics = pd.concat([
    cell_density_statistics,
    pos_cells_statistics,
    pos_cell_density_statistics,
    neg_cells_statistics,
    neg_cell_density_statistics,
    ki67_LI_statistics
], ignore_index=True)

statistics = statistics.rename(columns={'Pos_Percentage': 'Ki67_LI'})
statistics['variable'] = statistics['variable'].replace('Pos_Percentage', 'Ki67_LI')

print(statistics)

# save to csv file
statistics.to_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Ki67_statistics.csv', index=False)# save the statistics to a csv file

# %% KRUSKAL-WALLIS TEST 
Ki67_correlation_results = [] 

significance_level = 0.05

variables = ['Density', 'Positive', 'Pos_Density', 'Negative', 'Neg_Density', 'Pos_Percentage']

# statistical test
for variable in variables:

    # Bonferroni correction
    num_comparisons = len(summary_results['label'].unique())
    bonferroni_threshold = significance_level / num_comparisons
    
    for label in summary_results['label'].unique():
        data = summary_results[summary_results['label'] == label][variable]
        
        # calculate correlation
        if len(data) > 1:  
            stat, p_value = kruskal(data, summary_results[variable].dropna())
            Ki67_correlation_results.append({
                'label': label,
                'variable': variable,
                'Kruskal_Wallis_Statistic': stat,
                'p_value': p_value,
                'significant': p_value < bonferroni_threshold
            })

Ki67_correlation_results_df = pd.DataFrame(Ki67_correlation_results)

Ki67_correlation_results_df['variable'] = Ki67_correlation_results_df['variable'].replace('Pos_Percentage', 'Ki67_LI')

# print(Ki67_correlation_results_df)
# Save correlation results to a CSV file
Ki67_correlation_results_df.to_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Ki67_correlation_results.csv', index=False)

# %% KRUSKAL TUMOR DESCRIPTOR TEST
tumor_descriptor_results = []

# drop rows where tumor_descriptor equals "Not Available"
filtered_summary_results = summary_results[summary_results['tumor_descriptor'] != 'Not Available']

significance_level = 0.05

# Bonferroni correction
num_comparisons = len(summary_results['tumor_descriptor'].unique())
bonferroni_threshold = significance_level / num_comparisons

# statistical test
for descriptor in filtered_summary_results['tumor_descriptor'].unique():
    data = filtered_summary_results[filtered_summary_results['tumor_descriptor'] == descriptor]['Pos_Percentage'] 
    
    # calculate correlation
    if len(data) > 1:
        stat, p_value = kruskal(data, filtered_summary_results['Pos_Percentage'].dropna())
        tumor_descriptor_results.append({
            'tumor_descriptor': descriptor,
            'Kruskal_Wallis_Statistic': stat,
            'p_value': p_value,
            'significant': p_value < bonferroni_threshold
        })

tumor_descriptor_results_df = pd.DataFrame(tumor_descriptor_results)

tumor_descriptor_results_df = tumor_descriptor_results_df.rename(columns={'Pos_Percentage': 'Ki67_LI'})

print(tumor_descriptor_results_df)

tumor_descriptor_results_df.to_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Ki67_tumor_descriptor_correlation.csv', index=False)

# %% KRUSKAL TUMOR DESCRIPTOR TEST PER LABEL
initial_cns_results = summary_results[summary_results['tumor_descriptor'] == 'Initial CNS Tumor']

# rename label entries
initial_cns_results['label'] = initial_cns_results['label'].replace({
    'ATRT': 'ATRT_initial_CNS',
    'MED': 'MED_initial_CNS',
    'DIPG': 'DIPG_initial_CNS',
    'ASTR_HGG': 'ASTR_HGG_initial_CNS',
    'EP': 'EP_initial_CNS',
    'ASTR_LGG': 'ASTR_LGG_initial_CNS',
    'GANG': 'GANG_initial_CNS'
})

progressive_results = summary_results[summary_results['tumor_descriptor'] == 'Progressive']

# rename label entries
progressive_results['label'] = progressive_results['label'].replace({
    'MED': 'MED_progressive',
    'ATRT': 'ATRT_progressive',
    'ASTR_HGG': 'ASTR_HGG_progressive',
    'EP': 'EP_progressive',
    'ASTR_LGG': 'ASTR_LGG_progressive',
    'GANG': 'GANG_progressive'
})

recurrence_results = summary_results[summary_results['tumor_descriptor'] == 'Recurrence']

# rename label entries
recurrence_results['label'] = recurrence_results['label'].replace({
    'EP': 'EP_recurrence',
    'MED': 'MED_recurrence',
    'ASTR_HGG': 'ASTR_HGG_recurrence',
    'ASTR_LGG': 'ASTR_LGG_recurrence',
    'GANG': 'GANG_recurrence'
})

second_malignancy_results = summary_results[summary_results['tumor_descriptor'] == 'Second Malignancy']

# rename label entries
second_malignancy_results['label'] = second_malignancy_results['label'].replace({
    'ASTR_HGG': 'ASTR_HGG_second_malignancy',
    'MEN': 'MEN_second_malignancy',
    'ASTR_LGG': 'ASTR_LGG_second_malignancy'
})

# concatenate all results
all_results = pd.concat([
    initial_cns_results,
    progressive_results,
    recurrence_results,
    second_malignancy_results
], ignore_index=True)

# %%
# MED statistical test 
med_results = all_results[
    all_results['label'].isin([
        'MED_initial_CNS',
        'MED_progressive',
        'MED_recurrence',
        'MED_second_malignancy'
    ])
]

significance_level = 0.05

# Bonferroni correction
num_comparisons = len(med_results['label'].unique())
bonferroni_threshold = significance_level / num_comparisons

med_tumor_descriptor_results = []

for label in med_results['label'].unique():
    data = med_results[med_results['label'] == label]['Pos_Percentage']

    # calculate correlation
    if len(data) > 1:
        stat, p_value = kruskal(data, med_results['Pos_Percentage'].dropna())
        med_tumor_descriptor_results.append({
            'tumor_descriptor': label,
            'Kruskal_Wallis_Statistic': stat,
            'p_value': p_value,
            'significant': p_value < bonferroni_threshold
        })
med_tumor_descriptor_results_df = pd.DataFrame(med_tumor_descriptor_results)
med_tumor_descriptor_results_df = med_tumor_descriptor_results_df.rename(columns={'Pos_Percentage': 'Ki67_LI'})

# ATRT statistical test
atrt_results = all_results[
    all_results['label'].isin([
        'ATRT_initial_CNS',
        'ATRT_progressive',
        'ATRT_recurrence',
        'ATRT_second_malignancy'
    ])
]

significance_level = 0.05

# Bonferroni correction
num_comparisons = len(atrt_results['label'].unique())
bonferroni_threshold = significance_level / num_comparisons

atrt_tumor_descriptor_results = []
for label in atrt_results['label'].unique():
    data = atrt_results[atrt_results['label'] == label]['Pos_Percentage']

    # calculate correlation
    if len(data) > 1:
        stat, p_value = kruskal(data, atrt_results['Pos_Percentage'].dropna())
        atrt_tumor_descriptor_results.append({
            'tumor_descriptor': label,
            'Kruskal_Wallis_Statistic': stat,
            'p_value': p_value,
            'significant': p_value < bonferroni_threshold
        })
atrt_tumor_descriptor_results_df = pd.DataFrame(atrt_tumor_descriptor_results)
atrt_tumor_descriptor_results_df = atrt_tumor_descriptor_results_df.rename(columns={'Pos_Percentage': 'Ki67_LI'})

# ASTR_HGG statistical test
astr_hgg_results = all_results[
    all_results['label'].isin([
        'ASTR_HGG_initial_CNS',
        'ASTR_HGG_progressive',
        'ASTR_HGG_recurrence',
        'ASTR_HGG_second_malignancy'
    ])
]

significance_level = 0.05

# Bonferroni correction
num_comparisons = len(astr_hgg_results['label'].unique())
bonferroni_threshold = significance_level / num_comparisons

astr_hgg_tumor_descriptor_results = []
for label in astr_hgg_results['label'].unique():
    data = astr_hgg_results[astr_hgg_results['label'] == label]['Pos_Percentage']

    # calculate correlation
    if len(data) > 1:
        stat, p_value = kruskal(data, astr_hgg_results['Pos_Percentage'].dropna())
        astr_hgg_tumor_descriptor_results.append({
            'tumor_descriptor': label,
            'Kruskal_Wallis_Statistic': stat,
            'p_value': p_value,
            'significant': p_value < bonferroni_threshold
        })
astr_hgg_tumor_descriptor_results_df = pd.DataFrame(astr_hgg_tumor_descriptor_results)
astr_hgg_tumor_descriptor_results_df = astr_hgg_tumor_descriptor_results_df.rename(columns={'Pos_Percentage': 'Ki67_LI'})

# EP statistical test
ep_results = all_results[
    all_results['label'].isin([
        'EP_initial_CNS',
        'EP_progressive',
        'EP_recurrence',
        'EP_second_malignancy'
    ])
]

significance_level = 0.05

# Bonferroni correction
num_comparisons = len(ep_results['label'].unique())
bonferroni_threshold = significance_level / num_comparisons

ep_tumor_descriptor_results = []
for label in ep_results['label'].unique():
    data = ep_results[ep_results['label'] == label]['Pos_Percentage']

    # calculate correlation
    if len(data) > 1:
        stat, p_value = kruskal(data, ep_results['Pos_Percentage'].dropna())
        ep_tumor_descriptor_results.append({
            'tumor_descriptor': label,
            'Kruskal_Wallis_Statistic': stat,
            'p_value': p_value,
            'significant': p_value < bonferroni_threshold
        })
ep_tumor_descriptor_results_df = pd.DataFrame(ep_tumor_descriptor_results)
ep_tumor_descriptor_results_df = ep_tumor_descriptor_results_df.rename(columns={'Pos_Percentage': 'Ki67_LI'})

# ASTR_LGG statistical test
astr_lgg_results = all_results[
    all_results['label'].isin([
        'ASTR_LGG_initial_CNS',
        'ASTR_LGG_progressive',
        'ASTR_LGG_recurrence',
        'ASTR_LGG_second_malignancy'
    ])
]

significance_level = 0.05

# Bonferroni correction
num_comparisons = len(astr_lgg_results['label'].unique())
bonferroni_threshold = significance_level / num_comparisons

astr_lgg_tumor_descriptor_results = []
for label in astr_lgg_results['label'].unique():
    data = astr_lgg_results[astr_lgg_results['label'] == label]['Pos_Percentage']

    # calculate correlation
    if len(data) > 1:
        stat, p_value = kruskal(data, astr_lgg_results['Pos_Percentage'].dropna())
        astr_lgg_tumor_descriptor_results.append({
            'tumor_descriptor': label,
            'Kruskal_Wallis_Statistic': stat,
            'p_value': p_value,
            'significant': p_value < bonferroni_threshold
        })
astr_lgg_tumor_descriptor_results_df = pd.DataFrame(astr_lgg_tumor_descriptor_results)
astr_lgg_tumor_descriptor_results_df = astr_lgg_tumor_descriptor_results_df.rename(columns={'Pos_Percentage': 'Ki67_LI'})

# GANG statistical test
gang_results = all_results[
    all_results['label'].isin([
        'GANG_initial_CNS',
        'GANG_progressive',
        'GANG_recurrence',
        'GANG_second_malignancy'
    ])
]

significance_level = 0.05

# Bonferroni correction
num_comparisons = len(gang_results['label'].unique())
bonferroni_threshold = significance_level / num_comparisons

gang_tumor_descriptor_results = []
for label in gang_results['label'].unique():
    data = gang_results[gang_results['label'] == label]['Pos_Percentage']

    # calculate correlation
    if len(data) > 1:
        stat, p_value = kruskal(data, gang_results['Pos_Percentage'].dropna())
        gang_tumor_descriptor_results.append({
            'tumor_descriptor': label,
            'Kruskal_Wallis_Statistic': stat,
            'p_value': p_value,
            'significant': p_value < bonferroni_threshold
        })
gang_tumor_descriptor_results_df = pd.DataFrame(gang_tumor_descriptor_results)
gang_tumor_descriptor_results_df = gang_tumor_descriptor_results_df.rename(columns={'Pos_Percentage': 'Ki67_LI'})

# MEN statistical test
men_results = all_results[
    all_results['label'].isin([
        'MEN_initial_CNS',
        'MEN_progressive',
        'MEN_recurrence',
        'MEN_second_malignancy'
    ])
]

significance_level = 0.05

# Bonferroni correction
num_comparisons = len(men_results['label'].unique())
bonferroni_threshold = significance_level / num_comparisons

men_tumor_descriptor_results = []
for label in men_results['label'].unique():
    data = men_results[men_results['label'] == label]['Pos_Percentage']

    # calculate correlation
    if len(data) > 1:
        stat, p_value = kruskal(data, men_results['Pos_Percentage'].dropna())
        men_tumor_descriptor_results.append({
            'tumor_descriptor': label,
            'Kruskal_Wallis_Statistic': stat,
            'p_value': p_value,
            'significant': p_value < bonferroni_threshold
        })
men_tumor_descriptor_results_df = pd.DataFrame(men_tumor_descriptor_results)
men_tumor_descriptor_results_df = men_tumor_descriptor_results_df.rename(columns={'Pos_Percentage': 'Ki67_LI'})

# DIPG statistical test
dipg_results = all_results[
    all_results['label'].isin([
        'DIPG_initial_CNS',
        'DIPG_progressive',
        'DIPG_recurrence',
        'DIPG_second_malignancy'
    ])
]

significance_level = 0.05

# Bonferroni correction
num_comparisons = len(dipg_results['label'].unique())
bonferroni_threshold = significance_level / num_comparisons

dipg_tumor_descriptor_results = []
for label in dipg_results['label'].unique():
    data = dipg_results[dipg_results['label'] == label]['Pos_Percentage']

    # calculate correlation
    if len(data) > 1:
        stat, p_value = kruskal(data, dipg_results['Pos_Percentage'].dropna())
        dipg_tumor_descriptor_results.append({
            'tumor_descriptor': label,
            'Kruskal_Wallis_Statistic': stat,
            'p_value': p_value,
            'significant': p_value < bonferroni_threshold
        })
dipg_tumor_descriptor_results_df = pd.DataFrame(dipg_tumor_descriptor_results)
dipg_tumor_descriptor_results_df = dipg_tumor_descriptor_results_df.rename(columns={'Pos_Percentage': 'Ki67_LI'})

# concatenate all results
tumor_descriptor_correlation_results = pd.concat([
    tumor_descriptor_results_df,
    med_tumor_descriptor_results_df,
    atrt_tumor_descriptor_results_df,
    astr_hgg_tumor_descriptor_results_df,
    ep_tumor_descriptor_results_df,
    astr_lgg_tumor_descriptor_results_df,
    gang_tumor_descriptor_results_df,
    men_tumor_descriptor_results_df,
    dipg_tumor_descriptor_results_df
], ignore_index=True)

print(tumor_descriptor_correlation_results)

# save the results to a csv file
tumor_descriptor_correlation_results.to_csv('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/Ki67_tumor_descriptor_correlation_results.csv', index=False)


# %%
