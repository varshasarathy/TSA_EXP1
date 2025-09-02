# Ex.No: 01A PLOT A TIME SERIES DATA
###  Date: 

# AIM:
To Develop a python program to Plot a time series data (population/ market price of a commodity
/temperature.
# ALGORITHM:
1. Import the required packages like pandas and matplot
2. Read the dataset using the pandas
3. Calculate the mean for the respective column.
4. Plot the data according to need and can be altered monthly, or yearly.
5. Display the graph.
# PROGRAM:
```

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from tabulate import tabulate


sns.set(style='whitegrid', palette='muted', color_codes=True)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


import random
random.seed(42)
np.random.seed(42)


import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')`


df = pd.read_csv('/content/archive (4).zip', encoding='ISO-8859-1')

print(df.head())


print('Data Loaded. Shape:', df.shape)
df.head()



df.duplicated().sum()


df.isnull().sum()


print('\nStatistical Summary:')
print(df.describe())

 time_cols = ['timeOpen', 'timeClose', 'timeHigh', 'timeLow']
 for col in time_cols:
    try:
        df[col] = pd.to_datetime(df[col], unit='s')
    except Exception as e:
        print(f"Error converting {col}:", e)
 # Convert price columns from string to numeric. Some strings might have commas or other non-numeric characters.
 price_cols = ['priceOpen', 'priceHigh', 'priceLow', 'priceClose']
 for col in price_cols:
    try:
        df[col] = pd.to_numeric(df[col].str.replace(',', '').str.strip(), errors='coerce')
    except Exception as e:
        print(f"Error converting {col}:", e)
 # Convert 'volume' column to numeric
 try:
    df['volume'] = pd.to_numeric(df['volume'].str.replace(',', '').str.strip(), errors='coerce')
 except Exception as e:
    print('Error converting volume:', e)
 # Display the first few rows to verify changes
 print('Data types after cleaning:')
 print(df.dtypes.head(10))

numeric_cols = df.select_dtypes(include=['number']).columns
 for col in numeric_cols:
    sns.histplot(x=col, data=df, kde=True)
    plt.show()




```


# OUTPUT:

<img width="710" height="245" alt="image" src="https://github.com/user-attachments/assets/d5f3560f-e848-4f59-86f6-bad0c4dd9928" />

<img width="1060" height="222" alt="image" src="https://github.com/user-attachments/assets/b35eede4-f73c-4615-8974-993df379ce5f" />

<img width="567" height="345" alt="image" src="https://github.com/user-attachments/assets/5b2c8428-d2f6-47ac-b71b-fb7380fc4f76" />

<img width="618" height="202" alt="image" src="https://github.com/user-attachments/assets/0c41b386-7659-4ec7-9cde-88d7681adfc9" />

<img width="637" height="230" alt="image" src="https://github.com/user-attachments/assets/21605de9-3e18-437c-b10f-cfc1b29a1a3b" />

<img width="555" height="373" alt="image" src="https://github.com/user-attachments/assets/e238fb71-97a7-43e5-b450-d97d0e5057b0" />

<img width="548" height="379" alt="image" src="https://github.com/user-attachments/assets/87511a81-3efc-4078-ae38-e54779fc2e7d" />

# RESULT:
Thus we have created the python code for plotting the time series of given data.
