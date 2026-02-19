# Offer_Acceptance_Prediction
Predicting client acceptance of financial offers using process mining (pm4py) and machine learning. Models trained: Logistic Regression, Decision Tree, SVM, XGBoost and ANN. Includes EDA, feature engineering and an interactive Gradio demo.

# Business Problem
Financial institutions generate a large number of credit offers for clients, but not all offers are accepted. Without a predictive system, the offering process is costly and inefficient — resources are allocated uniformly, regardless of the actual probability that a client will accept. Early identification of offers with a high likelihood of acceptance allows the institution to prioritize and personalize its commercial interventions.

# Business Impact
A model that predicts offer acceptance delivers concrete value to the financial institution by:

- Reducing operational costs associated with rejected offers
- Improving conversion rates by targeting clients with a high probability of acceptance
- Optimizing financial parameters (amount, installments, monthly cost) based on the client's profile
- Increasing client satisfaction and enabling a more efficient allocation of commercial resources


# Project Overview

- Data source: BPI Challenge 2017 – Offer Log
- Format: XES (event log), read with pm4py
- Size: ~43,000 traces at offer level
- Target variable: selected_by_client (True / False)
- Problem type: Binary classification
- Final model: XGBoost


# Key Steps
1. Data Preprocessing

Reading the XES event log using pm4py
- Converting to DataFrame and renaming columns for readability
- Aggregating events at offer level (offer_id) through feature engineering
- Extracting new variables: time_to_accept_days, was_returned, was_cancelled, was_refused, n_events
- Checking and handling missing values

2. Exploratory Data Analysis (EDA)

- Distribution of accepted vs. refused offers at application level
- Number of offers per application
- Waiting time until offer acceptance
- Distribution of offered amounts, monthly costs and number of terms for accepted offers
- Correlation analysis between numerical variables
- Monthly evolution of created offers
- Outlier analysis through boxplots


# Top 5 Factors Influencing Offer Acceptance

- Credit Score — the dominant factor; clients with a high score accept offers much more frequently
- Number of Terms — clients prefer offers with a longer duration, which reduces the monthly installment
- Offered Amount — offers with slightly higher amounts have a marginally better acceptance rate
- Monthly Cost — lower monthly costs slightly increase the chances of acceptance
- First Withdrawal Amount — limited influence, acts in combination with the other factors


# Modeling
The following models were trained and compared:
- Logistic Regression — Baseline model; dominant on credit_score
- Decision Tree — Trained with and without pruning
- SVM — RBF kernel with StandardScaler
- XGBoost — Final model, best overall results
- ANN — Neural network with Early Stopping (TensorFlow / Keras)

# Hyperparameter Tuning
GridSearchCV with cross-validation (cv=5) was applied to improve model performance:

- Decision Tree — tuning on criterion, max_depth, min_samples_split, min_samples_leaf, max_features
- SVM — tuning on C and gamma
- XGBoost — tuning on n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma


# Predictions
The final model (XGBoost) generates for each offer:

- Decision — Accepted / Refused
- Probability of Acceptance — value between 0 and 1
- Risk of Rejection — complement of the acceptance probability

At the end of the notebook there is a Gradio interface where you can manually enter offer parameters and get the model's prediction in real time.

# Business Insights

- The distribution of accepted vs. refused offers is nearly perfectly balanced (~50/50), indicating a well-calibrated offering process
- Credit score is the only factor with strong discriminatory power in linear models; the other financial variables act in combination
- Offers that were initially returned are accepted in the majority of cases afterwards, suggesting that post-return adjustments are effective
- Most applications have a single offer, but there are cases with up to 7 offers for the same application
- Financial parameters (amount, monthly cost, number of terms) are strongly correlated with each other, reflecting the logic of credit calculation


# Technologies Used
- Python
- pm4py
- pandas / numpy
- matplotlib / seaborn
- scikit-learn
- XGBoost
- TensorFlow / Keras
- Gradio
