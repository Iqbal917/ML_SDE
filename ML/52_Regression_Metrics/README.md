# 📊 Regression Evaluation Metrics

This repository/section provides a comprehensive overview of the most commonly used evaluation metrics for regression models in machine learning. Understanding these metrics is crucial for diagnosing model performance, comparing different algorithms, and communicating results effectively.

## 📖 Table of Contents
* [Introduction](#introduction)
* [Notation](#notation)
* [1. Mean Absolute Error (MAE)](#1-mean-absolute-error-mae)
* [2. Mean Squared Error (MSE)](#2-mean-squared-error-mse)
* [3. Root Mean Squared Error (RMSE)](#3-root-mean-squared-error-rmse)
* [4. R-Squared ($R^2$) Score](#4-r-squared-r2-score)
* [5. Adjusted R-Squared Score](#5-adjusted-r-squared-score)
* [Summary & When to Use Which](#summary--when-to-use-which)

## Introduction
Regression models predict continuous numerical values. Unlike classification, where predictions are either right or wrong, regression predictions are rarely exact. Therefore, we evaluate these models by measuring the "distance" or "error" between the predicted values and the actual values. 

## Notation
For the mathematical definitions below, we use the following standard notation:
* $n$: Total number of observations (samples)
* $y_i$: The actual (true) value of the $i$-th observation
* $\hat{y}_i$: The predicted value for the $i$-th observation
* $\bar{y}$: The mean of all actual values
* $p$: The number of independent variables (predictors/features) in the model

---

## 1. Mean Absolute Error (MAE)
MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It is simply the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.

**Formula:**
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Pros & Cons:**
* **Pros:** Highly interpretable as it scales in the same units as the target variable. It is robust to outliers.
* **Cons:** The absolute value function is not everywhere differentiable, which can make it less convenient for mathematical optimizations like gradient descent.

---

## 2. Mean Squared Error (MSE)
MSE measures the average of the squares of the errors. By squaring the errors before averaging them, MSE gives a relatively high weight to large errors. This means it heavily penalizes models that make large mistakes.

**Formula:**
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Pros & Cons:**
* **Pros:** Differentiable everywhere, making it the default loss function for many optimization algorithms.
* **Cons:** Highly sensitive to outliers. The unit is the square of the target variable's unit, making it harder to interpret directly.

---

## 3. Root Mean Squared Error (RMSE)
RMSE is the square root of the MSE. It serves the same purpose as MSE but brings the metric back to the original unit of the target variable, making it much easier to interpret alongside your data.

**Formula:**
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Pros & Cons:**
* **Pros:** Shares the same unit as the target variable. Penalizes large errors, which is useful when large deviations are particularly undesirable.
* **Cons:** Still sensitive to outliers compared to MAE.

---

## 4. R-Squared ($R^2$) Score
Also known as the Coefficient of Determination, the $R^2$ score represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It provides an indication of the goodness of fit of a set of predictions to the actual values.

**Formula:**
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

*Note: The denominator is the total sum of squares (variance of the data), and the numerator is the residual sum of squares (variance explained by the model).*

**Pros & Cons:**
* **Pros:** Provides a scale-free score between $-\infty$ and $1.0$ (where $1.0$ is a perfect fit, and $0.0$ is a baseline model that just predicts the mean).
* **Cons:** $R^2$ will always increase (or stay the same) when you add more features to the model, even if those features are completely useless. 

---

## 5. Adjusted R-Squared Score
The Adjusted $R^2$ modifies the standard $R^2$ score to account for the number of predictors in the model. It penalizes the addition of features that do not improve the model's explanatory power, solving the main drawback of the standard $R^2$ metric.

**Formula:**
$$Adjusted\ R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

**Pros & Cons:**
* **Pros:** The best metric for comparing regression models with a differing number of independent variables. Prevents overfitting inflation.
* **Cons:** Can be slightly less intuitive to explain to non-technical stakeholders compared to a standard $R^2$ percentage.

---

## Summary & When to Use Which

| Metric | Primary Use Case | Outlier Sensitivity | Interpretable Units? |
| :--- | :--- | :---: | :---: |
| **MAE** | When you want an interpretable error metric and your dataset has outliers you want to ignore. | Low | Yes |
| **MSE** | When you want to heavily penalize large errors or are training an optimization algorithm. | High | No (Squared) |
| **RMSE** | When you want to penalize large errors but need the metric to be in the original units. | High | Yes |
| **$R^2$** | When you need a quick, scale-less baseline of how well your model explains the data variance. | Moderate | No (Ratio) |
| **$R^2_{adjusted}$**| When you are doing feature selection and comparing models with different numbers of predictors. | Moderate | No (Ratio) |
