import numpy as np
import pandas as pd
import matplotlib.pyplot as plt      # for plotting graphs
plt.style.use('seaborn')             # the seaborn stylesheet will make plots look neat/pretty

df = pd.read_csv("world_happiness_15_21.csv")
df.info()
df.isnull().sum() #find total null values in each column
df.duplicated().sum() #find duplicate values

#Keep Only Rows with Happiness Rank 
df = df[df['Happiness Rank'].notna()]
df.info()

#Replace Null Values in Perceptions of Corruption and Generosity Columns Using Imputation 
generosity_median = df['Generosity'].median()
print("Generosity Median is {}".format(generosity_median))
perceptions_of_corruption_median = df['Perceptions of corruption'].median()
print("Perceptions of Corruption Median is {}".format(perceptions_of_corruption_median))

df['Generosity'].fillna(generosity_median, inplace = True)
df['Perceptions of corruption'].fillna(perceptions_of_corruption_median, inplace = True)

df.info()

#plot histogram of data -0 shows frequency distribution of each column
df.hist(bins='auto', figsize=(10,10));
plt.show()
