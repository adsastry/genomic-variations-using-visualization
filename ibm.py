import pandas as pd
data = pd.read_csv("C:/Users/aditi/Downloads/test_rows.csv")
data_cleaned = data.dropna(how='all') 
data_cleaned = data_cleaned.dropna(axis=1, how='all')
from sklearn.preprocessing import StandardScaler
numeric_data = data.select_dtypes(include=[float, int])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

#importing packages for visualizations
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#heatmap
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap of Genomic Data')
plt.show()

#scatter plot
from sklearn.impute import SimpleImputer
scaled_data = scaled_data[~np.isnan(scaled_data).any(axis=1)]
imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent'
scaled_data = imputer.fit_transform(scaled_data)
print("Number of NaNs in scaled data:", np.isnan(scaled_data).sum())
print(scaled_data.shape) 
from sklearn.decomposition import PCA
n_components = min(scaled_data.shape)  
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
if pca_result.shape[1] == 1:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_df['PC1'], y=[0] * len(pca_df))  # Y = 0 as a dummy value
    plt.title('PCA of Genomic Data (1 Component)')
    plt.show()
else:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df)
    plt.title('PCA of Genomic Data (2 Components)')
    plt.show()

#violin plot
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.violinplot(data=numeric_data)
plt.title('Violin Plot of Gene Expressions Across Samples')
plt.xticks(rotation=90)
plt.show()

#box plot
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.boxplot(data=numeric_data)
plt.xticks(rotation=90)  
plt.title('Gene Expression Variations Across Samples')
plt.xlabel('Genes')
plt.ylabel('Expression Level')
plt.show()

#histogram
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.histplot(numeric_data.iloc[:, 0], kde=True)  
plt.title('Distribution of Gene 1 Expression')
plt.xlabel('Gene Expression')
plt.ylabel('Frequency')
plt.show()

