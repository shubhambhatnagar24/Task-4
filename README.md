# Task-4
This task involves performing exploratory data analysis on a dataset.

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the sample dataset
iris = sns.load_dataset('iris')

# Display the first few rows of the dataset
print(iris.head())

# Step 2: Create Histograms
def create_histograms(data):
    data.hist(figsize=(10, 8), bins=15, edgecolor='black')
    plt.suptitle('Histograms of All Variables', fontsize=16)
    plt.show()

# Step 3: Create Box Plots
def create_boxplots(data):
    data.plot(kind='box', subplots=True, layout=(2,3), figsize=(12, 8), title='Boxplots of All Variables')
    plt.suptitle('Boxplots of All Variables', fontsize=16)
    plt.show()

# Step 4: Create Pair Plots
def create_pairplots(data):
    sns.pairplot(data)
    plt.suptitle('Pair Plot of All Variables', fontsize=16)
    plt.show()

# Step 5: Create Correlation Heatmap
def create_correlation_heatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap', fontsize=16)
    plt.show()

# Generate the visualizations
create_histograms(iris)
create_boxplots(iris)
create_pairplots(iris)
create_correlation_heatmap(iris)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate a synthetic dataset
np.random.seed(42)
data = pd.DataFrame({
    'Variable1': np.random.normal(loc=0, scale=1, size=100),
    'Variable2': np.random.normal(loc=5, scale=2, size=100),
    'Variable3': np.random.normal(loc=10, scale=3, size=100)
})

# Step 2: Create Histograms
def create_histograms(data):
    data.hist(figsize=(10, 8), bins=15, edgecolor='black')
    plt.suptitle('Histograms of All Variables', fontsize=16)
    plt.show()

# Step 3: Create Box Plots
def create_boxplots(data):
    data.plot(kind='box', subplots=True, layout=(1, 3), figsize=(12, 6), title='Boxplots of All Variables')
    plt.suptitle('Boxplots of All Variables', fontsize=16)
    plt.show()

# Step 4: Create Pair Plots
def create_pairplots(data):
    sns.pairplot(data)
    plt.suptitle('Pair Plot of All Variables', fontsize=16)
    plt.show()

# Step 5: Create Correlation Heatmap
def create_correlation_heatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap', fontsize=16)
    plt.show()

# Generate the visualizations
create_histograms(data)
create_boxplots(data)
create_pairplots(data)
create_correlation_heatmap(data)
