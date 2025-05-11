# **Residuals in Machine Learning**

## **1. Concept**

A **residual** is the **difference between the actual value (real y)** and the **predicted value** of a model.

Formula:

$$\text{Residual}_i = y_i - \hat{y}_i$$

Where:  
- $y_i$ is the actual value of sample i  
- $\hat{y}_i = F(x_i)$ is the predicted value of the model

## **2. Explanation with Visualization and Examples**

Suppose we have data:

| $x$ | $y$ (actual) | $\hat{y}$ (predicted) | Residual |
|-----|--------------|------------------------|----------|
| 1   | 2            | 2.5                    | 2 - 2.5 = **-0.5** |
| 2   | 3            | 2.7                    | 3 - 2.7 = **+0.3** |
| 3   | 2.5          | 2.9                    | 2.5 - 2.9 = **-0.4** |
| 4   | 5            | 4.2                    | 5 - 4.2 = **+0.8** |

**Meaning**:
- If **residual > 0** → model **under-predicts** (prediction lower than actual)
- If **residual < 0** → model **over-predicts** (prediction higher than actual)
- If **residual = 0** → **perfect** prediction

**Visual representation** (conceptual):

```
          y (actual value)
          *
          |    *   <-- predicted (lower than true)
          | *
----------|----------------> x
```

The blue line is the actual value, the red line is the prediction → the distance between the two lines is the **residual**

## **3. Role of Residuals in Gradient Boosting**

In Gradient Boosting:
- Residuals are what the **new model needs to learn** at each step
- Idea: If the old model predicts poorly (residual ≠ 0), we fit a weak learner to the **residuals** so the new model corrects the errors

In each boosting round:

$$r_i = y_i - F_{m-1}(x_i)$$

Then we find $h_m(x)$ (small decision tree) such that:

$$h_m(x) \approx r_i$$

In other words:
> Each weak learner learns to **predict the deviation (residual)** of the previous model.

## **4. Practical Applications of Residuals**

| Application | Role of Residuals |
|-------------|-------------------|
| **Gradient Boosting** | Residuals signal how to train new models to correct errors in previous ones |
| **Regression Analysis** | Residuals help check if the model fits well (random residuals → good model) |
| **Model Diagnostics** | Used to detect overfitting/underfitting or outliers |
| **Linear Model Assumption Testing** | Residual plots help test linearity and homoscedasticity assumptions |

## **5. Important Notes**

- In **Gradient Boosting**, "residual" is typically **generalized** to the **negative gradient** of the loss function.  
If the loss function isn't MSE (e.g., **log loss** in classification), the residual isn't simply $y - \hat{y}$ but the derivative:

$$r_i = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$$

## **6. Brief Conclusion**

> **Residual = Error remaining after prediction**  
It's the **fuel** for Gradient Boosting to **continue improving the model**