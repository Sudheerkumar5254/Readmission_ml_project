# import the libraries what we need for this project.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset and check few rows by using head function.
df1=pd.read_csv(r'Mention data set name')
df1.head()

# data exploration 
df1.info()
df1.shape
df1.describe()

# check missing values in dataseet.
df1.isnull.sum()

# Correlation Between Numerical Variables
plt.figure(figsize=(10,6))
sns.heatmap(df1['num_cols'].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation")
plt.show()


# If target is numerical
sns.histplot(df1['re.admission.within.6.months'], bins=30, kde=True)
plt.title("Target Variable Distribution")
plt.show()

# Plot distribution for numerical features
numerical_cols = df1.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12,8))
plt.boxplot(df1[num_cols])
plt.show()


---------------------xxxxxxxxxxxxxxxxxxxx------------------



 