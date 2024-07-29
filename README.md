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
