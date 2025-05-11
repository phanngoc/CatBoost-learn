# **Negative Gradient in Gradient Boosting**

## **1. Concept**

In Gradient Boosting, at each iteration, we **don't simply learn residual = y − ŷ**.  
Instead, we treat the problem as **optimization of a loss function**.

→ The **negative gradient** of the loss function is the **"pseudo-residual"** → what the next model needs to learn.

**Definition** (for each data point $i$):

$$r_i^{(m)} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$$

- $L(y, F(x))$ = loss function  
- $F_{m-1}(x)$ = current model (sum of previous trees)  
- **r** = negative gradient (pseudo-residual)

**Intuitive idea**  
> Negative gradient gives us the **direction in which predictions should change to decrease loss fastest**.

## **2. Mathematical Explanation (with examples)**

### **Regression Case (MSE Loss Function)**

$$L(y, F(x)) = \frac{1}{2}(y - F(x))^2$$

Derivative with respect to $F(x)$:

$$\frac{\partial L}{\partial F(x)} = -(y - F(x))$$

Therefore:

$$r_i^{(m)} = -\left( - (y_i - F_{m-1}(x_i)) \right) = y_i - F_{m-1}(x_i)$$

**→ For MSE, the negative gradient is simply the regular residual**

### **Classification Case (Log Loss)**

$$L(y, F(x)) = \log(1 + e^{-2yF(x)})$$

(assuming binary classification, y ∈ {−1, +1})  
Derivative:

$$\frac{\partial L}{\partial F(x)} = \frac{-2y}{1 + e^{2yF(x)}}$$

Negative gradient:

$$r_i^{(m)} = \frac{2y_i}{1 + e^{2y_i F_{m-1}(x_i)}}$$

→ This is the **pseudo-residual** for classification (no longer simply $y - \hat{y}$)

## **3. Calculation Steps (Gradient Boosting Framework)**

Assume dataset $\{ (x_i, y_i) \}_{i=1}^n$

**Step 0** (Initialize model)

$$F_0(x) = \arg\min_{\gamma} \sum_{i=1}^n L(y_i, \gamma)$$

**Step 1** (Iterate from m = 1 to M)

1. Calculate **pseudo-residuals** for each point:

$$r_i^{(m)} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$$

2. Fit a **weak learner** (typically a small decision tree) to data $(x_i, r_i^{(m)})$

→ We get tree $h_m(x)$

3. Calculate **step size (shrinkage)**:

$$\gamma_m = \arg\min_{\gamma} \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$$

4. Update the model:

$$F_m(x) = F_{m-1}(x) + \nu \cdot \gamma_m h_m(x)$$

→ $\nu \in (0,1]$ is the learning rate

## **4. Practical Applications of Negative Gradient**

| Application | Role of Negative Gradient |
|-------------|---------------------------|
| **Regression** | Used to learn residual (in MSE) |
| **Binary Classification** | Used to update probabilities (log-loss) |
| **Multi-class Classification** | Calculate gradient for each class to improve softmax |
| **Ranking** | Used in LambdaRank, LambdaMART (ranking-based loss functions) |
| **Gradient Boosted Decision Trees (GBDT)** | Mathematical foundation for XGBoost, LightGBM, CatBoost |

## **5. Calculation Example (Regression with MSE)**

Data:

| $x$ | $y$ |
|-----|-----|
| 1   | 3   |
| 2   | 2   |
| 3   | 4   |

**Step 0:**  
Initialize $F_0(x) = \bar{y} = \frac{3+2+4}{3} = 3$

**Step 1:** Calculate residuals

$$r_1 = 3 - 3 = 0 \\
r_2 = 2 - 3 = -1 \\
r_3 = 4 - 3 = +1$$

**Step 2:** Fit tree $h_1(x)$ to { (1,0), (2,−1), (3,+1) }

Assume the tree learns a simple rule:
$$h_1(x) = 
\begin{cases}
-1 & x < 2.5 \\
+1 & x \geq 2.5
\end{cases}$$

**Step 3:** Calculate gamma

$$\gamma_1 = 1 \quad \text{(for regression, typically =1 if fit correctly to residuals)}$$

**Step 4:** Update model

$$F_1(x) = F_0(x) + \nu h_1(x) = 3 + \nu h_1(x)$$

Assume $\nu = 0.1$

$$F_1(x) = 3 + 0.1 h_1(x)$$

## **6. Summary of Important Formulas**

| Component | Formula |
|-----------|---------|
| Negative Gradient | $r_i = -\frac{\partial L}{\partial F(x_i)}$ |
| Model Update | $F_m = F_{m-1} + \nu \cdot \gamma_m h_m(x)$ |

## **Conclusion**

> **Negative Gradient** is the **direction of fastest error reduction** in each boosting round → new models learn it to correct errors
