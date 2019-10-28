# Udacity_DSND_Starbucks-Capstone-Challenge
This is Seunggyun Han's project for Starbucks' Capstone Project in Udacity Data Scientist Nanodegree.<br>

# Objective
The Objective of this project is for build recommendation system which finds the best offer for a customer.  <br>

# Problem Statement
As I mentioned before, this project was for solving two main problems.
- Building a machine learning model to predict a success of each offer for each customer.
- Building a offer recommendation system for the company based on the ML model (It means if the company put the demographic data of a customer, the system will return which offer is the best option).
To solving these problems, I followed a strategy below:
1. Data Pre-processing (Cleaning the data, Merging the data, and etc.)
2. Data Exploration (Calculating probabilities of completion for each offer)
3. Building a ML Model (Supervised Learning for predicting that an offer will be completed or not)
4. Building a recommendation system based on ‘2’ and ‘3’.

## Metric 
In this project, I used simple accuracy score to measure a performance of the recommendation system.

# 1. Install
For installation, please clone or download this repository<br>

## 1.1 Requirement
To run the jupyter notebook, please check requirements.txt file.<br><br>
Also, You can create the virtual environment with below code. <br>
`$ conda create --name <env> --file <this file>`<br>

# 2. File description
## 2.1 Data
Sorry, You are not able to use this. Because, I'm not sure there is any license problem with starbucks dataset.<br>
## 2.2 Jupyter Notebook
**Starbucks_Capstone_notebook.ipynb** is a jupyter notebook which contains cleaning processes of datasets and building ML model.<br>

## 2.3 tools.py
python file that contains functions to clean dataset
