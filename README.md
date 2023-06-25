# ACTL3143 Assignment: Heart Disease Prediction Using Deep Learning

By Sharon Zhou (z5310019)

## Table of contents
* [Introduction](#introduction)
* [Data Preparation](#datapreparation)
* [Exploratory Data Analysis](#exploratorydatanalysis)
* [Data Preprocessing](#datapreprocessing)
* [Baseline Model: Logistic Regression](#baselinemodel)

## 1. Introduction

This assignment will use past patient variables along with a target condition of having the presence or absence of heart disease to create a predictive model for future patients. This is a binary classification type problem with a "goal" field of 1 = presence and 0 = absence. 

This early draft will conduct exploratory data analysis to gain insights into the dataset, before fitting a logistic regression model as a baseline. Further improvements will be made through deep learning techniques.

### 1.1 Data Dictionary 

The dataset used is from the University of California Irvine (UCI) data repository and contains data on 270 patients with 13 independent predictive variables. Information on the attributes is contained below: 

``` age ```: the patient's age in years
``` sex ```: the patient's sex (1 = male, 0 = female)
``` cp ```: the chest pain experienced (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic)
``` trestbps ```: the patient's resting blood pressure in mm/Hg upon admission to the hospital
``` chol ```: the patient's cholesterol measurement in mg/dl
``` fbs ```: the patient's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
``` restecg ```: the patient's resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = left ventricular hypertrophy)
``` thalach ```: the patient's maximum heart rate achieved
``` exang ```: exercise-induced angina
``` oldpeak ```: the patient's ST depression induced by exercise relative to rest
``` slope ```: the slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)
``` ca ```: the number of major vessels (0-3) coloured by flouroscopy 
``` thal ```: displays the thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
``` num ```: the patient's diagnosis of heart disease (0 = absence; 1-4 = presence)
	
## 2. Data Preparation

### 2.1 Loading all packages 

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import random

from sklearn import set_config
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Reshape, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

set_config(transform_output="pandas")
```

### 2.2 Importing data 


	
## Exploratory Data Analysis
To run this project, install it locally using npm:

```
$ cd ../lorem
$ npm install
$ npm start
```
## Data Preprocessing

## Baseline Model: Logistic Regression
