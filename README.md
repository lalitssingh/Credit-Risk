# Credit Risk Modelling & Loan Default Prediction

**An End-to-End Machine Learning Project | Python | Scikit-learn | XGBoost | TensorFlow**

---

## 1. Project Overview

This project focuses on credit risk analysis and loan default prediction using historical LendingClub loan data. It explores how borrower characteristics, financial indicators and loan attributes can be used to identify patterns associated with loan defaults.

The project applies exploratory data analysis (EDA), statistical feature selection, data preprocessing and multiple machine learning classification algorithms to evaluate borrower credit risk.

The primary objective is to understand the factors associated with loan default and develop classification models that can support better-informed lending and risk assessment decisions.

## 2. Business Problem

Lending institutions face two major risks when evaluating loan applications:

- **Credit Loss:** Approving applications from borrowers who are likely to default can result in financial losses for the lender.
- **Opportunity Cost:** Rejecting borrowers who are likely to repay their loans can result in lost business and interest income.

The challenge is to distinguish between borrowers with different repayment risk profiles using historical loan and borrower information.

### Business Objectives

1. Identify borrower and loan characteristics associated with default risk.
2. Analyse relationships between financial indicators and loan repayment behaviour.
3. Apply statistical techniques to identify relevant predictive features.
4. Develop machine learning models to classify loan outcomes.
5. Compare model performance using classification evaluation metrics.
6. Translate analytical findings into insights that can support credit risk assessment.

## 3. Dataset Description

The project uses historical LendingClub loan data containing approximately **2.26 million loan records**.

The dataset includes borrower information, loan characteristics, credit history and repayment-related variables.

### Key Dataset Variables

| Category | Variables |
|---|---|
| Loan Characteristics | Loan Amount, Funded Amount, Interest Rate, Installment, Loan Term |
| Borrower Information | Annual Income, Employment Length, Home Ownership |
| Credit Profile | FICO Score, DTI, Revolving Balance, Credit Utilisation |
| Credit History | Delinquencies, Open Accounts, Total Accounts, Public Records |
| Loan Performance | Loan Status, Total Payments, Recoveries, Last Payment |

### Target Variable

The target variable is `loan_status`, transformed into a binary classification variable:

- **0 – Good Loan:** Fully Paid, Current, In Grace Period and loans marked as Fully Paid under the credit-policy status.
- **1 – Bad Loan:** Remaining loan status categories, including default-related outcomes.

The resulting dataset contains:

| Classification | Records | Percentage |
|---|---:|---:|
| Good Loans (0) | 1,965,465 | 86.94% |
| Bad Loans (1) | 295,173 | 13.06% |
| **Total** | **2,260,638** | **100%** |

This demonstrates an imbalanced classification problem, with significantly more good loans than bad loans.

## 4. Exploratory Data Analysis (EDA)

The first stage involved understanding the dataset, examining variable distributions and identifying relationships between borrower characteristics and loan outcomes.

### Key Areas of Analysis

- Distribution of loan status and repayment performance.
- Loan amount, interest rate and monthly installment distributions.
- Borrower income and employment characteristics.
- Loan grades and subgrades.
- Debt-to-income (DTI) ratios.
- FICO scores and credit history.
- Revolving balances and credit utilisation.
- Loan purpose, home ownership and loan tenure.
- Correlation analysis between numerical variables.

### Key Findings

**1. Loan Grades and Credit Risk**

- Grades A–C were dominated by Fully Paid and Current loans.
- Lower grades, particularly D–G, showed higher concentrations of Charged Off and Late loans.
- F and G subgrades exhibited particularly unfavourable repayment patterns.

**2. Interest Rates**

- Most loans had interest rates between approximately 8% and 16%.
- Higher interest-rate bands showed greater concentrations of Charged Off and Default loans.
- Interest rates were associated with differences in borrower risk profiles.

**3. Loan Tenure**

- 36-month loans represented approximately 65% of the dataset.
- 60-month loans showed relatively greater default exposure.

**4. Debt-to-Income Ratio**

- Most borrowers had DTI values concentrated between 15 and 20.
- Higher DTI ranges showed increased concentrations of negative loan outcomes.

**5. Credit Utilisation**

- Lower revolving credit utilisation was generally associated with more favourable repayment outcomes.
- Higher utilisation levels showed greater concentrations of Charged Off loans.

**6. Other Observations**

- Mortgage and Rent were the most common home ownership categories.
- Debt Consolidation and Credit Card were among the most common loan purposes.
- Borrower income alone did not clearly distinguish between successful and unsuccessful repayment outcomes.

## 5. Feature Selection: Weight of Evidence (WOE) & Information Value (IV)

Before developing the machine learning models, Weight of Evidence (WOE) and Information Value (IV) were used to evaluate the predictive relevance of categorical and numerical variables.

### Weight of Evidence (WOE)

WOE measures the relationship between the distribution of good and bad loan observations across different categories or intervals.

It helps identify how individual feature categories are associated with good and bad loan outcomes.

### Information Value (IV)

IV measures the overall predictive separation provided by a variable.

The project used the following interpretation:

| Information Value | Interpretation |
|---|---|
| Below 0.02 | Weak predictive power |
| 0.02–0.10 | Weak |
| 0.10–0.30 | Medium |
| 0.30–0.50 | Strong |
| Above 0.50 | Potentially suspicious |

The notebook applied an acceptance threshold of 0.02–0.50.

### Selected Features

**Categorical Features:**
- Term
- Grade
- Home Ownership
- Verification Status
- Issue Date
- Purpose
- Initial Listing Status

**Numerical Features:**
- Loan Amount
- Interest Rate
- Installment
- Annual Income
- Debt-to-Income Ratio
- FICO Range (Low and High)
- Credit Inquiries
- Revolving Utilisation
- Total Payments
- Total Interest Received

The final modelling dataset contained 18 selected features.

## 6. Data Preprocessing

The selected variables underwent preprocessing before being passed into the machine learning models.

### Missing Value Treatment

- Identified missing values across selected features.
- Applied median imputation to DTI and revolving utilisation.
- Used a ColumnTransformer-based preprocessing workflow.

### Outlier Treatment

Applied the Interquartile Range (IQR) method to identify and cap extreme numerical observations.

The boundaries were calculated using:

- Lower Bound = Q1 − 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

### Feature Transformation

- Converted loan tenure into numerical values.
- Applied Min-Max Scaling to numerical features.
- Applied One-Hot Encoding to categorical variables.
- Integrated transformations into a Scikit-learn Pipeline.

### Train-Test Split

The initial model development used an 80:20 training and testing split with a random state of 42. The Random Forest and subsequent model workflow also used stratification to preserve class proportions.

## 7. Machine Learning Models

Five classification approaches were implemented and compared.

### 1. Logistic Regression

Used as a baseline classification model to estimate the relationship between borrower characteristics and the probability of a bad loan outcome.

### 2. Decision Tree Classifier

Implemented to identify non-linear relationships and decision rules associated with loan repayment behaviour.

The initial model showed overfitting, leading to an additional hyperparameter-tuning exercise using GridSearchCV.

Parameters explored:
- Maximum Tree Depth
- Minimum Samples Split

Recall was used as the tuning objective.

### 3. Random Forest Classifier

Applied an ensemble of decision trees to improve classification and capture complex relationships between borrower characteristics.

The model was evaluated using precision, recall, F1-score and confusion matrices.

### 4. XGBoost Classifier

Implemented gradient-boosted decision trees to improve the identification of high-risk borrowers.

The model incorporated `scale_pos_weight` to account for the imbalance between good and bad loan observations.

### 5. Artificial Neural Network (ANN)

Developed a feed-forward neural network using TensorFlow and Keras.

Architecture:
- Input Layer: 16 neurons, ReLU activation.
- Hidden Layer: 8 neurons, ReLU activation.
- Output Layer: 1 neuron, Sigmoid activation.

Model configuration:
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Training Epochs: 20
- Batch Size: 32

## 8. Model Evaluation & Performance Comparison

The models were evaluated using multiple classification metrics.

| Evaluation Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Proportion of predicted bad loans that were actually bad |
| Recall | Proportion of actual bad loans correctly identified |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish between the two classes |

### Preliminary Results Reported by the Notebook

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 93.86% | 92.17% | 57.68% | 70.96% |
| Decision Tree | 97.00% | 89.00% | 87.00% | 88.00% |
| Random Forest | 94.00% | 99.61% | 53.68% | 69.76% |
| XGBoost | 96.18% | 80.90% | 92.56% | 86.34% |
| Artificial Neural Network | 99.00% | 99.00% | 93.00% | 96.00% |

*Values are rounded where the notebook's comparison output was rounded.*

The notebook also reports ROC-AUC values of 0.78, 0.93, 0.77, 0.95 and 0.96, respectively.

**Important validation note:** These are preliminary notebook results, not validated out-of-sample performance claims. The notebook calculates ROC-AUC using hard class predictions rather than predicted probabilities. In addition, some selected features, particularly total payments and interest received, contain information generated after loan origination and may introduce target leakage. The ANN also requires further independent validation. These issues should be addressed before interpreting the scores as evidence of real-world predictive performance.

## 9. Business Interpretation & Practical Applications

The analysis highlights how borrower characteristics and loan attributes can contribute to credit risk assessment.

Potential applications include:

**Risk-Based Lending**
- Identify borrower profiles associated with higher default exposure.
- Incorporate credit grades, interest rates and repayment indicators into risk assessment.

**Credit Policy Development**
- Evaluate the relationship between loan tenure and default patterns.
- Review lending criteria for higher-risk borrower segments.

**Portfolio Risk Management**
- Monitor the concentration of loans across different credit grades.
- Identify segments requiring closer repayment monitoring.

**Model-Based Decision Support**
- Use classification probabilities to support risk categorisation.
- Evaluate precision and recall trade-offs when selecting lending thresholds.

These applications represent potential uses of the analysis, not a deployed lending system.

## 10. Conclusion

This project demonstrates an end-to-end approach to credit risk analysis using exploratory data analysis, statistical feature selection and machine learning.

By examining approximately 2.26 million historical loan records, the analysis identified meaningful relationships between repayment outcomes and factors such as loan grades, interest rates, loan tenure, DTI and credit utilisation.

The comparison of Logistic Regression, Decision Trees, Random Forest, XGBoost and Artificial Neural Networks provided an opportunity to evaluate different approaches to credit risk classification.

An important learning outcome was understanding that accuracy alone is insufficient when evaluating credit risk models. Precision and recall are particularly important because incorrectly classifying a high-risk borrower can have significant financial consequences for a lender.

The project strengthened practical understanding of credit risk analytics, statistical modelling, machine learning and the application of quantitative techniques to financial decision-making.

## 11. Limitations & Future Improvements

The project provides a foundation for further development. The following improvements are important:

- Remove post-origination variables that could introduce target leakage.
- Develop a prediction framework using only information available at loan application.
- Introduce chronological train-test validation.
- Calculate ROC-AUC using predicted probabilities.
- Perform cross-validation and more extensive hyperparameter optimisation.
- Explore model explainability using SHAP.
- Evaluate model calibration and lending decision thresholds.
- Incorporate cost-sensitive learning to account for the financial impact of default misclassification.
- Develop an interactive credit risk dashboard.

## 12. Technologies & Libraries

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn, HvPlot
- **Machine Learning:** Scikit-learn, XGBoost
- **Deep Learning:** TensorFlow, Keras
- **Statistical Analysis:** SciPy
- **Environment:** Jupyter Notebook

## 13. Repository Structure

```text
credit-risk-modelling/
│
├── README.md
├── credit_risk_modelling.ipynb
│
├── data/
│   └── Loan_data.csv
│
├── visualizations/
│
└── requirements.txt
```

## 14. How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/yourusername/credit-risk-modelling.git
```

2. Navigate to the project directory:

```bash
cd credit-risk-modelling
```

3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost tensorflow hvplot
```

4. Place the dataset in the project directory, using the filename `Loan_data.csv`.

5. Open the Jupyter Notebook:

```bash
jupyter notebook credit_risk_modelling.ipynb
```

6. Execute the notebook cells sequentially.

*Note: The original LendingClub dataset is not included in this repository. Ensure you have the appropriate rights to access and redistribute the dataset before publishing it.*

## 15. Disclaimer

This project is intended for educational and analytical purposes. The dataset contains historical lending information, and the findings should not be interpreted as financial advice or as a validated credit approval model.

The reported model metrics are preliminary and require additional validation, particularly with respect to target leakage, temporal validation and probability-based evaluation.

---

**Author:** Lalit Kumar Singh

**Project:** Credit Risk Modelling & Loan Default Prediction

**Focus Areas:** Credit Risk Analytics | Quantitative Analysis | Machine Learning | Financial Data Science
