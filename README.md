# Tennis-Betting-ML

A Machine Learning project for predicting tennis match outcomes using Logistic Regression with Stochastic Gradient Descent (SGD). The model achieves a 66% accuracy rate on approximately 125,000 singles matches, utilizing data from the [Kaggle Tennis Dataset](https://www.kaggle.com/ehallmar/a-large-tennis-dataset-for-atp-and-itf-betting?select=all_matches.csv).

---

## Project Overview
This project predicts the outcome of tennis matches using historical player statistics and betting odds. The methodology is divided into four key steps:

1. **Data Processing:** Cleaning and transforming the dataset.
2. **Feature Engineering:** Creating meaningful features such as Elo ratings, service ratings, and return ratings.
3. **Model Training:** Training a logistic regression model with Stochastic Gradient Descent.
4. **Interpretability:** Understanding feature contributions using Odds Ratios and SHAP values.

---

## Dataset
The dataset, sourced from Kaggle, contains match data and betting odds scraped from the ATP World Tour website. It includes information on player performance, match outcomes, and betting probabilities.

### Preprocessing Steps:
- Filtered matches from 2000 onwards.
- Merged duplicate rows for unified player statistics.
- Standardized player names and cleaned missing data.

---

## Features
The model uses the following features:
1. **Elo Ratings:** A dynamic ranking system for players.
2. **Exponential Moving Average (EMA):** Tracks dynamic features like service and return ratings.
3. **Service Rating:** Aggregates six service-related statistics using PCA.
4. **Return Rating:** Captures return performance using PCA.
5. **Head-to-Head Balance:** Historical performance between players.
6. **Performance Metric:** Weighted results based on match outcomes and Elo differences.
7. **Service Mistakes:** Models consistency using double faults and errors.
8. **Age Difference:** Accounts for player experience.

---

## Model and Training

### Logistic Regression
The logistic regression model predicts the probability of a player winning a match:
\[
p = \sigma(z) = \frac{1}{1 + e^{-z}}
\]
Where \(z = \beta_0 + \beta_1x_1 + \dots + \beta_nx_n\), \(x\) represents features, and \(\beta\) are model parameters.

### Training
- **Dataset Split:** 70% training, 30% testing.
- **Regularization:** L2 regularization to prevent overfitting.
- **Optimization:** Random grid search with K-fold cross-validation (30 folds) to optimize hyperparameters.

---

## Results
- **Accuracy:** 66%
- **Precision-Recall Curve:** Average precision of 0.71.
- **ROC Curve:** AUC of 0.72, indicating good classification performance.

---

## Interpretability
1. **Odds Ratios:** Shows the effect of each feature on match predictions.
2. **SHAP Values:** Highlights the most influential features:
   - **Elo Ratings**: Strongest impact.
   - **Service Mistakes**: Negative correlation.

---

## Future Improvements
1. Introduce non-linear dimensionality reduction techniques (e.g., autoencoders).
2. Explore advanced models like ensemble methods or neural networks.
3. Incorporate more features and improve data quality.

---

## Requirements
The project is implemented in Python and requires the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `shap`

---

## How to Run
1. Download the dataset from [Kaggle](https://www.kaggle.com/ehallmar/a-large-tennis-dataset-for-atp-and-itf-betting?select=all_matches.csv).
2. Clone this repository.
3. Install the required Python libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn shap
4.	Open the Tennis-Betting-ML.ipynb notebook in Jupyter or another Python environment.
5.	Follow the notebook’s instructions to preprocess the data, train the model, and evaluate the results.

References

1.	M. Sipko - Machine Learning for the Prediction of Professional Tennis Matches
2.	Producing Win Probabilities for Professional Tennis Matches from Any Score: Link
3.	Kaggle Dataset: Link

License
This project is licensed under the MIT License.
