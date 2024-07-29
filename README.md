# Customer Credit Scoring and Segmentation Analysis

## Overview

This project involves analyzing and segmenting customer data from a credit scoring dataset. The analysis includes calculating credit scores, performing clustering to segment customers, and visualizing the results.

## Dataset

The dataset `credit_scoring.csv` contains the following columns:
- **Age**: Age of the customer
- **Gender**: Gender of the customer
- **Marital Status**: Marital status of the customer
- **Education Level**: Education level of the customer
- **Employment Status**: Employment status of the customer
- **Credit Utilization Ratio**: Ratio of credit used
- **Payment History**: Payment history of the customer
- **Number of Credit Accounts**: Number of credit accounts held
- **Loan Amount**: Amount of the loan
- **Interest Rate**: Interest rate on the loan
- **Loan Term**: Term of the loan
- **Type of Loan**: Type of loan (Personal, Auto, Home)

## Dependencies

The following Python libraries are required:
- `pandas`
- `seaborn`
- `matplotlib`
- `sklearn`

## Data Preprocessing

1. **Import Libraries and Load Data**
   ```python
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   from sklearn.cluster import KMeans

   df = pd.read_csv('credit_scoring.csv')



2. ** Load Data**
df = pd.read_csv('credit_scoring.csv')

3. ** Data Exploration**
print(df.info())
print(df.describe())

4. **Data Cleaning and Transformation**
education_level_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
employment_status_mapping = {'Unemployed': 0, 'Employed': 1, 'Self-Employed': 2}

df['Education Level'] = df['Education Level'].map(education_level_mapping)
df['Employment Status'] = df['Employment Status'].map(employment_status_mapping)

5. **Calculate Credit Scores**
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

6. **Add the credit scores as a new column to the DataFrame**
df['Credit Score'] = credit_scores

7. **Customer Segmentation**
X = df[['Credit Score']]
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
kmeans.fit(X)
df['Segment'] = kmeans.labels_
df['Segment'] = df['Segment'].astype('category')

8. **Visualization**
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

