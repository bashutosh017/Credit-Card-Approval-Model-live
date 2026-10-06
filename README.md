# India Credit Card Approval Predictor

A robust, production-grade Machine Learning pipeline built using **LightGBM** to predict credit card application approval statuses based on open-market financial indicators in the Indian banking ecosystem.

This project addresses real-world challenges in risk management—including **Class Imbalance** (80/20 market skew) and **Stochastic Market Noise** (field verification failures, documentation errors)—shifting it from a simple academic exercise to an enterprise-grade classification model.

[![Architecture diagram of bashutosh017/credit-card-approval-ml-model](https://gitdiagram.com/bashutosh017/credit-card-approval-ml-model/diagram.png)](https://gitdiagram.com/bashutosh017/credit-card-approval-ml-model?utm_source=readme&utm_medium=picture)

[![Architecture diagram of bashutosh017/credit-card-approval-model-live](https://gitdiagram.com/bashutosh017/credit-card-approval-model-live/diagram.png)](https://gitdiagram.com/bashutosh017/credit-card-approval-model-live?utm_source=readme&utm_medium=picture)



---

## 📌 Project Overview
Traditional credit scoring models rely on closed banking data. This project simulates an open-market aggregator or fintech platform layout, evaluating applications via standalone financial strength and credit bureau health metrics. 

The underlying data model incorporates strict financial guardrails aligned with current Indian retail banking benchmarks (e.g., TransUnion CIBIL standards and standard Fixed Income to Obligation Ratio limits), yielding a realistic **ROC-AUC score of 0.92**.

---

## 📊 Dataset Schema Design
The dataset maps 1,000 distinct, realistic customer application vectors across the following structured parameters:

| Column Name | Data Type | Description | Real-World Context / Constraint |
| :--- | :--- | :--- | :--- |
| `application_id` | Integer | Unique primary tracking key | Excluded during training to prevent data leakage. |
| `age` | Integer | Chronological age of applicant | Enforced boundary of 21–65 years (Indian credit limit). |
| `city_tier` | Integer | Categorical ordinal rank (1, 2, 3) | Tier 1 (Metros), Tier 2 (State Capitals), Tier 3 (Towns). |
| `employment_type`| Category | Primary income source bucket | `Working`, `Commercial associate`, `Pensioner`, `State servant`. |
| `monthly_income` | Float | Net take-home salary in INR | Set with sectoral standard distributions; floor min of ₹10,000. |
| `cibil_score` | Integer | Bureau credit reputation score | Ranged from 300 to 900. Core driver of creditworthiness. |
| `foir_percentage`| Float | Fixed Income to Obligation Ratio | Measures existing EMI burdens. Optimal bounds sit below 50%. |
| `approval` | Binary | Target Label (1=Approve, 0=Reject) | 80.3% Rejections / 19.7% Approvals (Imbalanced Market Class). |

### 🧠 Simulated Market Noise (Handling Determinism)
To prevent perfect algorithmic shortcutting (1.0 Accuracy), the dataset injects random entropy simulating true operational friction:
* **8% False Negatives:** Premium profiles rejected due to failed physical address checks, KYC typos, or employment verification anomalies.
* **3% False Positives:** Sub-prime profiles approved under high-net-worth relational exemptions or aggressive bank credit campaigns.

---

## ⚙️ Core Architecture & Pipeline

The pipeline isolates the machine learning training runtime from execution logic to uphold clean **Separation of Concerns**:

1. **`model.py` (The Training Engine):** 
   Loads the dataset, handles text categorical casting natively, addresses the 80/20 class skew using cost-sensitive learning (`is_unbalance=True`), evaluates metrics, and packages the components into serialized binaries using `joblib`.
2. **`predict.py` (The Execution Gateway):** 
   A lightweight, zero-dependency console interface that deserializes the `.pkl` files and executes sub-second validation matrix matching against live user runtime inputs.

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
Ensure you have Python 3.9+ installed. Clone your repository and install the dependencies:
```bash
pip install pandas numpy lightgbm scikit-learn joblib matplotlib seaborn
```

### 2. Run the Machine Learning Pipeline
Train the LightGBM classifier and serialize the pipeline states:
```bash
python model.py
```
*This produces `lightgbm_credit_model.pkl` and `model_features.pkl` in your root directory.*

### 3. Test Live Console Predictions
Launch the prediction interactive layout to test manual metrics:
```bash
python predict.py
```

---

## 📈 Model Performance & Evaluation

The LightGBM configuration employs early regularization (`max_depth=5`, `learning_rate=0.05`) to safeguard generalization properties:

### Confusion Matrix Insights
Evaluated on a 200-sample randomized test holdout split:
```text
[[161   0]
 [  6  33]]
```
* **True Negatives (161):** High-risk profiles correctly blocked by the risk model.
* **False Positives (0):** Bad debt prevention score. The model displays a highly conservative risk ceiling.
* **False Negatives (6):** Clean profiles snagged by structural verification anomalies.
* **True Positives (33):** Safe, high-yield retail banking targets captured.

### Key Evaluation Scores
* **Weighted Average F1-Score:** 0.97
* **ROC-AUC Performance Evaluation:** 0.9230 (Highly reliable, presentation-grade deployment curve)

---

## 🔑 Key Hyperparameters Explained

* **`is_unbalance=True`**: Mandates LightGBM to automatically penalize minority class errors inversely proportional to their presence, preventing the model from defaulting to blanket rejections.
* **`n_estimators=100`**: Iterates up to 100 progressive gradient boosting trees to refine decision boundary splits.
* **`max_depth=5`**: Limits tree depth to 5 hierarchical levels, preventing the algorithm from memorizing noise profiles (**Overfitting**).
* **`random_state=42`**: Anchors pseudo-random generation factors to establish exact metric reproducibility across execution environments.