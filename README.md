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

1.age: The person's age in years

2.sex: The person's sex (1 = male, 0 = female)

3.cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)

4.trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)

5.chol: The person's cholesterol measurement in mg/dl

6.fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)

7.restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)

8.thalach: The person's maximum heart rate achieved

9.exang: Exercise induced angina (1 = yes; 0 = no)

10.oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)

11.slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)

12.ca: The number of major vessels (0-3)

13.thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

14.target: Heart disease (0 = no, 1 = yes)

	
## Data Preparation
Project is created with:
* Lorem version: 12.3
* Ipsum version: 2.33
* Ament library version: 999
	
## Exploratory Data Analysis
To run this project, install it locally using npm:

```
$ cd ../lorem
$ npm install
$ npm start
```
## Data Preprocessing

## Baseline Model: Logistic Regression
