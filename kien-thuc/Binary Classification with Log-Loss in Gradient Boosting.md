# **Binary Classification with Log-Loss in Gradient Boosting**

This section explains how Gradient Boosting is applied to binary classification problems using the Log-Loss function.

## **Mathematical Foundation**

In **binary classification**, we typically:
- Assign labels $y \in \{0, 1\}$ or $y \in \{-1, +1\}$
- Need to predict probability $P(y=1|x)$

**Log-Loss** (Binary Cross-Entropy):

With $y \in \{0, 1\}$:
$$L(y, p) = -y\log(p) - (1-y)\log(1-p)$$

With $y \in \{-1, +1\}$:
$$L(y, F(x)) = \log(1 + e^{-2yF(x)})$$

In Gradient Boosting:
- $F(x)$ is not directly a probability
- It's the log-odds or "raw score": $F(x) = \log\frac{P(y=1|x)}{1-P(y=0|x)}$
- Probability is calculated as: $P(y=1|x) = \frac{1}{1+e^{-F(x)}}$ (sigmoid function)

## **Calculating the Negative Gradient**

**Step 1**: Find the derivative of the loss function with respect to model $F(x)$

With $y \in \{-1, +1\}$:
$$\frac{\partial L(y, F(x))}{\partial F(x)} = \frac{-2y}{1 + e^{2yF(x)}}$$

With $y \in \{0, 1\}$:
$$\frac{\partial L(y, p)}{\partial F(x)} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial F(x)} = \left(-\frac{y}{p} + \frac{1-y}{1-p}\right) \cdot p(1-p) = p - y$$

**Step 2**: Calculate negative gradient

$$r_i = -\frac{\partial L}{\partial F(x_i)} = y_i - p_i = y_i - \frac{1}{1+e^{-F(x_i)}}$$

→ This is the **residual between actual label and predicted probability**

## **Calculation Example (Binary Classification with Log-Loss)**

Data:

| $x$ | $y$ (1 = positive, 0 = negative) |
|-----|-----------------------------------|
| 1   | 1                                 |
| 2   | 0                                 |
| 3   | 1                                 |
| 4   | 0                                 |

**Step 0**: Initialize model
- $F_0(x) = \log\frac{P(y=1)}{P(y=0)} = \log\frac{0.5}{0.5} = 0$ (Assuming prior probability = 0.5)

**Step 1**: Calculate negative gradient (pseudo-residual)
- Predicted probabilities: $p_i = \frac{1}{1+e^{-F_0(x_i)}} = \frac{1}{1+e^{0}} = 0.5$ (all = 0.5 in first iteration)
- Negative gradient:
  - $r_1 = 1 - 0.5 = 0.5$
  - $r_2 = 0 - 0.5 = -0.5$
  - $r_3 = 1 - 0.5 = 0.5$
  - $r_4 = 0 - 0.5 = -0.5$

**Step 2**: Fit decision tree to negative gradient

Assume the tree learns the rule:
$$h_1(x) = 
\begin{cases}
-0.4 & \text{if } x \in \{2, 4\} \\
+0.4 & \text{if } x \in \{1, 3\}
\end{cases}$$

**Step 3**: Calculate gamma (line search)
$$\gamma_1 = \arg\min_{\gamma} \sum_{i=1}^4 \log(1 + e^{-y_i(F_0(x_i) + \gamma h_1(x_i))})$$

Assume $\gamma_1 = 1.0$

**Step 4**: Update model with learning rate $\nu = 0.1$
$$F_1(x) = F_0(x) + \nu \cdot \gamma_1 h_1(x) = 0 + 0.1 \cdot 1.0 \cdot h_1(x) = 0.1 \cdot h_1(x)$$

Therefore:
- $F_1(1) = 0.1 \cdot 0.4 = 0.04$
- $F_1(2) = 0.1 \cdot (-0.4) = -0.04$
- $F_1(3) = 0.1 \cdot 0.4 = 0.04$
- $F_1(4) = 0.1 \cdot (-0.4) = -0.04$

**Step 5**: Update predicted probabilities:
- $p_1 = \frac{1}{1+e^{-0.04}} \approx 0.51$
- $p_2 = \frac{1}{1+e^{0.04}} \approx 0.49$
- $p_3 = \frac{1}{1+e^{-0.04}} \approx 0.51$
- $p_4 = \frac{1}{1+e^{0.04}} \approx 0.49$

→ Probabilities have been adjusted slightly in the correct direction

**Repeat the process** until the defined iterations are reached or the model converges.

## **Key Features of Binary Classification in Gradient Boosting**

1. **Link Function**: Use sigmoid to convert raw score to probability
   $P(y=1|x) = \frac{1}{1+e^{-F(x)}}$

2. **Log-Loss decreases slower than MSE**: Requires more iterations to converge

3. **Model Interpretation**:
   - Large $F(x)$ values → high confidence in positive label
   - Small $F(x)$ values (large negative) → high confidence in negative label
   - $F(x) \approx 0$ → uncertainty

### **Practical Applications**

| Application | Description |
|-------------|-------------|
| **Credit Scoring** | Predicting probability of customer default/payment |
| **Churn Prediction** | Predicting probability of customer leaving service |
| **Medical Diagnosis** | Detecting disease based on symptoms and test indices |
| **Fraud Detection** | Detecting fraudulent transactions with high probability |
| **Marketing Campaign** | Predicting user response to advertising campaign |

## **Important Notes**

- **Class Imbalance**: When data is imbalanced, negative gradient for fewer points will get less attention. Solutions:
  - Scale samples by weight
  - Use sampling techniques
  - Adjust prior probability

- **Calibration**: Sometimes predicted probabilities need calibration to accurately reflect actual probabilities
  - Platt Scaling
  - Isotonic Regression

- **One-vs-Rest**: Generalize to multi-class by training n_classes binary models

## **Summary Formulas for Binary Classification**

| Component | Formula |
|-----------|---------|
| Log-Loss (y ∈ {0,1}) | $L(y, p) = -y\log(p) - (1-y)\log(1-p)$ |
| Negative Gradient | $r_i = y_i - p_i = y_i - \frac{1}{1+e^{-F(x_i)}}$ |
| Predicted Probability | $P(y=1|x) = \frac{1}{1+e^{-F(x)}}$ |