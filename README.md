# Customer Credit Scoring and Segmentation Analysis

## Overview

This project involves analyzing and segmenting customer data from a credit scoring dataset. The analysis includes calculating credit scores, performing clustering to segment customers, and visualizing the results.

## Dependencies

The following Python libraries are required:
- `pandas`
- `seaborn`
- `matplotlib`
- `sklearn`

## Code

```python
# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Data
df = pd.read_csv('credit_scoring.csv')

# Data Exploration
print(df.info())
print(df.describe())

# Data Cleaning and Transformation
education_level_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
employment_status_mapping = {'Unemployed': 0, 'Employed': 1, 'Self-Employed': 2}

df['Education Level'] = df['Education Level'].map(education_level_mapping)
df['Employment Status'] = df['Employment Status'].map(employment_status_mapping)

# Calculate Credit Scores
credit_scores = []
for index, row in df.iterrows():
    payment_history = row['Payment History']
    credit_utilization_ratio = row['Credit Utilization Ratio']
    number_of_credit_accounts = row['Number of Credit Accounts']
    education_level = row['Education Level']
    employment_status = row['Employment Status']

    # Apply the FICO formula to calculate the credit score
    credit_score = (payment_history * 0.35) + (credit_utilization_ratio * 0.30) + (number_of_credit_accounts * 0.15) + (education_level * 0.10) + (employment_status * 0.10)
    credit_scores.append(credit_score)

# Add the credit scores as a new column to the DataFrame
df['Credit Score'] = credit_scores

# Customer Segmentation
X = df[['Credit Score']]
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
kmeans.fit(X)
df['Segment'] = kmeans.labels_
df['Segment'] = df['Segment'].astype('category')

# Visualization
# Box Plot of Credit Utilization Ratio
sns.boxplot(df['Credit Utilization Ratio'])
plt.show()

# Histogram of Loan Amount
sns.histplot(df['Loan Amount'], bins=20)
plt.show()

# Heatmap of Correlation Matrix
cor = df.select_dtypes(['int64', 'float64']).corr()
sns.heatmap(cor, annot=True, cmap='coolwarm')
plt.show()

# Scatter Plot of Customer Segments
sns.scatterplot(data=df, x=df.index, y='Credit Score', hue='Segment', palette=['green', 'blue', 'yellow', 'red'])
plt.xlabel('Customer Index')
plt.ylabel('Credit Score')
plt.title('Customer Segmentation based on Credit Scores')
plt.show()
