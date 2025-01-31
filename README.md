# Churn Prediction

This repository contains an Analysis and Machine Learning group project focused on customer churn prediction modeling.

## Table of Contents
- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Usage](#usage)
- [Models and Techniques](#models-and-techniques)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Overview
Customer churn prediction is crucial for businesses to retain their customers and improve their services. This project aims to develop a predictive model to identify customers who are likely to churn, using various machine learning techniques.

## Project Objectives
- Analyze customer data to understand factors contributing to churn.
- Preprocess and clean the data for modeling.
- Develop and evaluate machine learning models to predict churn.
- Provide actionable insights and recommendations based on the model's predictions.

## Dataset
The training dataset used for this project contains historical customer data, including demographic information, service usage, and churn status. The additional dataset of 25 000 customers is provided in a semi structured Data Base. After training the model it was converted into the library using MongoDB, and then transformed to a structured Data Frame using the MQL query. Both datasets can be found in the `prediction` directory of this repository.

## Usage
The project was mainly performed in Python, using the following libraries: numpy, pandas, matplotlib, seaborn, scikit-learn.
Load data (both train and new data), prediction, accuracy and saving new data prediction to csv are performed to run the churn prediction model by running `main.py` in the `prediction` directory of this repository.

## Models and Techniques
The project explores various machine learning models and techniques, including:
- Decision Tree
- Random Forest
- KNN (K nearest neighbours) 

## Results
The results of the models can be found in the `prediction` directory. The best-performing model (Random forest) is applied to the new dataset and is further visualized in Tableau and analyzed to provide insights and recommendations. More detailed information and insights can be found in `PowerPoint Presentation`.

## Acknowledgements
We would like to thank our group members and mentors for their support and guidance throughout this project.

## Code
This README file provides a clear structure and detailed information about the project, making it easy for others to understand and use the repository.
