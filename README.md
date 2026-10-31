# Census Income Classification & Ensemble Engine

An end-to-end Machine Learning pipeline and custom **Random Forest** and **Decision Tree** classifier built completely from scratch using **Python and NumPy**, applied to the classic UCI Adult Census Income dataset.

This project avoids high-level machine learning libraries (such as `scikit-learn`) for core algorithms, implementing custom optimization, tree splitting, bootstrap aggregation, and evaluation metrics from first principles.

---

##  Key Features & From-Scratch Implementations

* **Custom Decision Tree Classifier:** 
  * Supports both **Gini Impurity** and **Shannon Entropy** splitting criteria.
  * Implements vectorised split-scoring using cumulative class distributions for $O(N \log N)$ threshold evaluation per feature.
  * Configurable hyperparameters including maximum depth, minimum samples per split, and minimum leaf size.
* **Random Forest Ensemble:**
  * Implements **Bootstrap Aggregating (Bagging)** via custom random sampling with replacement ($\approx 63.2\%$ in-bag, $\approx 36.8\%$ out-of-bag per tree).
  * Enforces **Random Feature Subspaces** ($m$ features sampled independently at every internal node).
  * Generates robust ensemble predictions via hard majority voting and class probability averaging.
* **Rigorous Evaluation & Diagnostics:**
  * Built-in **Out-of-Bag (OOB) Error Evaluation** for unbiased generalisation tracking without a validation split.
  * **Feature Importance Estimation** via total impurity reduction and validation permutation importance.
  * Custom evaluation metrics suite: Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
* **Data Preprocessing & Handling:**
  * Strict training-set-only statistical imputation (mode/median) to prevent data leakage.
  * Custom one-hot encoding handling categorical attributes and unseen test categories.
  * Numerical correlation and redundancy checks.

---

## 📂 Repository Structure

```text
├── data/                    # UCI Adult Census dataset (adult.data, adult.test)
├── Q1.ipynb                 # Complete implementation notebook with experiments & plots
└── README.md                # Project documentation
