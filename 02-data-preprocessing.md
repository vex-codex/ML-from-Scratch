# DATA PROCESSING AND CLEANING

## WHAT YOU WILL LEARN
- What is Data Preprocessing
- Why it is important in ML
- Types of Data
- Handling Missing Values
- Handling Outliers
- Data Encoding
- Feature Scaling
- Complete Python Code Examples


## 1.WHAT IS DATA PROCESSING

> Data Preprocessing is the process of **transforming raw data into a 
  clean and usable format** before feeding it into a Machine Learning model.

> RAW DATA ──► CLEAN DATA ──► ML MODEL ──► PREDICTIONS 

## 2. WHY IT IS IMPORTANT?

> Missing values -> Model crashes
> Outliers -> Wrong Predictions
> Unscaled features -> Biased results 
> Text categories -> Model cannot read 

## 3.TYPES OF DATA

> **Numerical** = Age, Salary, Score 
> **Categorical** = Gender, Color, City 
> **Text** = Reviews, Comments 
> **DateTime** = Order Date, Login Time 

## 4. HANDLING MISSING VALUES

### Check Missing Values
```python
import pandas as pd
df = pd.read_csv('data.csv')

# Check missing values
print(df.isnull().sum())
```
### Fill Missing Values
```python
# Numerical → Fill with Mean or Median
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].median(), inplace=True)

# Categorical → Fill with Mode
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
```

### Drop Missing Values
```python
# Drop rows
df.dropna(inplace=True)
```

## 4.WHEN TO USE?
| Strategy -> Use When |

> **Mean** -> Numerical, no outliers 
> **Median** -> Numerical, with outliers 
> **Mode** -> Categorical data |
> **Drop** -> Very few missing rows 


## 5.HANDLING OUTLIERS

### Detect with Boxplot
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['Salary'])
plt.show()
```

###  Remove with IQR Method
```python
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

# Keep only values within range
df = df[
    (df['Salary'] >= Q1 - 1.5 * IQR) &
    (df['Salary'] <= Q3 + 1.5 * IQR)
]
```

###  Remove with Z-Score
```python
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(df['Salary']))
df = df[z < 3]
```

## 6.DATA ENCODING(Encoding categorical data)

> Lable encoding -> Ordered categories (Low/Mid/High)
> One Hot encoding -> No order (City, Color)



## 7.FEATURE SCALING

### Min-Max Scaling
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])
# Values scaled between 0 and 1
```

### Standard Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])
# Mean = 0 | Std = 1
```

## 8. TRAIN TEST SPLIT

```python
from sklearn.model_selection import train_test_split

X = df.drop('Target', axis=1)   # Features
y = df['Target']                 # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% Train | 20% Test
    random_state=42
)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
'''
