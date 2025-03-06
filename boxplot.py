# %% IMPORT
import matplotlib.pyplot as plt
import pandas as pd 

# %% LOAD DATA
# read xlsx file icluding the column names
df = pd.read_excel('/local/data1/chrsp39/QuPath-Automatic-Cell-Detection-for-Ki-67-WSIs/data_files/KI67_sum_up.xlsx')    

# %%
# Print column names to debug
print(df.columns)

# plot a boxplot with labels the label column and Test
df.boxplot(column='KI67_LI_2', by='label')
plt.ylabel('KI67 Label Index')
plt.xlabel('')
# the labels in the x-axis are overlapping
plt.xticks(rotation=60)
plt.tight_layout()
# empty title
plt.title('')

# save plot in high resolution
plt.savefig('boxplot.png', dpi=1000)

# %%
