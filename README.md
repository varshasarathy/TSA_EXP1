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

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/content/BMW_Car_Sales_Classification[1].csv')
df

df.info()

df.isnull().sum()

df.dtypes

target = 'Year'

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

plt.figure()
sns.countplot(x=target, data=df, palette='pastel')
plt.title('Target Class Distribution')
plt.xlabel(target)
plt.ylabel('Count')
plt.show()


```


# OUTPUT:

<img width="1398" height="516" alt="image" src="https://github.com/user-attachments/assets/a13003d4-e31f-4170-81b3-db1e3a145e1f" />



<img width="537" height="406" alt="image" src="https://github.com/user-attachments/assets/d9811f79-34fb-4b5b-91f7-56bdc319d239" />



<img width="338" height="534" alt="image" src="https://github.com/user-attachments/assets/0c32788b-ec75-476a-ba78-1c9740e8fcad" />



<img width="1184" height="665" alt="image" src="https://github.com/user-attachments/assets/6e6ec9c9-ddbc-4851-b648-2e96b7c09e53" />



<img width="974" height="620" alt="image" src="https://github.com/user-attachments/assets/e66e3b39-0cf1-46d6-969a-dbd6015e549a" />


# RESULT:
Thus we have created the python code for plotting the time series of given data.
