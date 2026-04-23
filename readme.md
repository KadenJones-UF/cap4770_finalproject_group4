# Amazon Product Success Analysis
### CAP4770 - Introduction to Data Science
### Team Name: Group 4

## Team Members
- Kaden Jones
- Louiza Joya
- Roshan Mehta
- Jacob Ramos

## Synopsis
The goal of this project is to analyze and predict what makes a product successful on Amazon. This real world problem impacts sellers, businesses, and consumers on one of the largest e-commerce platforms. We will use techniques taught in class including classification and regression for prediction, and clustering for identifying patterns, and data visualizations to demonstrate findings. 

## Dataset
- Source: [Kaggle Amazon Products Dataset 2023] (https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products?select=amazon_products.csv)
- Size: 1.4 million products
- Features: star ratings, reviews, price, productID, best seller

## Project Structure
```
cap4770_finalproject_group4/
    model_comparison.py
    utils.py
cap4770_finalproject_group4/data/
    amazon_products.csv
    amazon_categories.csv
cap4770_finalproject_group4/notebooks/
    preprocessing.ipynb
    exploratory_data_analysis.ipynb
    xgboost_model.ipynb

```

## Tools & Technologies
- Python
- Pandas
- Matplotlib
- Jupyter Notebook
- NumPy
- seaborn
- XGBoost
- scikit-learn

## Installation
- Step 1: Clone the repo
- Step 2: Download the data and ensure it is within the /data folder. (see download_data.sh)
- Step 3: Install dependencies (see requirements.txt)
- Step 4: Run notebooks in order starting with preprocessing.ipynb

## Evaluation Metrics
- Precision
- Recall
- Confusion matrix with sensitivity and specificity
