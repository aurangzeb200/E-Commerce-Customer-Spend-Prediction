# 📊 E-Commerce Customer Spend Prediction

This Python project analyzes e-commerce customer behavior to predict total spending using machine learning models. It includes exploratory data analysis (EDA) with visualizations and regression models (Lasso, Random Forest, Gradient Boosting, Decision Tree) to identify factors influencing customer spending and satisfaction. Built with Pandas, Scikit-learn, and Seaborn, it’s a great example of data preprocessing, feature engineering, and model evaluation.

## Features
- **Exploratory Data Analysis (EDA)**: Visualizes relationships between customer attributes (e.g., membership type, spend category, gender, city) and satisfaction level.
- **Feature Engineering**: Creates `Spend Category` and `Buyer’s Type` using binning, and encodes categorical variables with LabelEncoder and one-hot encoding.
- **Machine Learning Models**: Implements Lasso Regression, Random Forest, Gradient Boosting, and Decision Tree to predict `Total Spend`.
- **Model Evaluation**: Compares models using RMSE and MSE, with residual analysis and correlation heatmaps.
- **Data Preprocessing**: Handles missing values, standardizes features, and removes highly correlated features.

## Interesting Techniques Used
Let’s break down some key techniques in this project, step-by-step, like we’re sketching it out on a whiteboard. I’ll keep it clear and intuitive, so you can follow along even if you’re new to machine learning.

### 1. Feature Engineering with Binning
To make sense of `Total Spend` and `Days Since Last Purchase`, we create new features (`Spend Category` and `Buyer’s Type`) by binning numerical values into categories. This helps the models understand patterns better.

- **What’s happening?**: We divide `Total Spend` into `Low Spend`, `Medium Spend`, and `High Spend` using percentiles (33rd, 66th, 99th). For example, spends below 660.3 are `Low Spend`, between 660.3 and 830.9 are `Medium Spend`, and above 830.9 are `High Spend`. Similarly, `Days Since Last Purchase` is binned into `Frequent Buyers` (0–25 days), `Regular Buyers` (25–40 days), etc.
- **How it works**: We use Pandas’ `pd.cut` function with bins `[0, 660.3, 830.9, inf]` and labels `["Low Spend", "Medium Spend", "High Spend"]`. For each customer, `Total Spend` is checked against these bins, and a label is assigned. For example, a spend of 700 gets `Medium Spend`.
- **Why it’s cool**: Binning transforms continuous data into categorical data, which can capture non-linear patterns (e.g., high spenders are always satisfied). It’s a simple way to make numerical data more interpretable for both EDA and models. Check out [Pandas’ documentation on `cut`](https://pandas.pydata.org/docs/reference/api/pandas.cut.html) for more details.

### 2. Encoding Categorical Variables
Since machine learning models need numbers, not strings, we encode categorical features like `Membership Type`, `City`, and `Gender` using LabelEncoder and one-hot encoding.

- **What’s happening?**: `LabelEncoder` converts categories (e.g., `Silver`, `Gold`) into integers (e.g., 0, 1). For `City` and `Gender`, we use `pd.get_dummies` to create binary columns (e.g., `City_Chicago`, `City_New York`, `Gender_Male`, `Gender_Female`), where 1 means the category applies, and 0 means it doesn’t.
- **How it works**: For a customer with `City=Chicago`, `pd.get_dummies` sets `City_Chicago=1` and other city columns to 0. We then concatenate these binary columns to the dataset and drop the original categorical columns. For example, a customer might have `[City_Chicago=1, City_New York=0, Gender_Male=1]`.
- **Why it’s cool**: One-hot encoding avoids implying an order between categories (unlike LabelEncoder alone), which is critical for features like `City` where `Chicago` isn’t “less than” `New York`. This prevents models from misinterpreting categorical data. See [Scikit-learn’s preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html) for more.

### 3. Model Selection with Grid Search
We use GridSearchCV to tune the Lasso Regression model’s `alpha` parameter and optimize hyperparameters for Random Forest, ensuring the best model performance.

- **What’s happening?**: For Lasso, we test `alpha` values `[0.1, 1, 10, 100]` using 5-fold cross-validation. GridSearchCV trains the model on each `alpha`, calculates the mean squared error (MSE) on validation folds, and picks the best `alpha`. For Random Forest, we use parameters like `n_estimators=125` and `min_samples_split=10`.
- **How it works**: Let’s say we’re testing Lasso with `alpha=0.1`. GridSearchCV splits the training data into 5 folds, trains on 4 folds, tests on 1, and repeats 5 times. It computes the average MSE for each `alpha` and selects the one with the lowest MSE. The best model is then used to predict `Total Spend` on the test set.
- **Why it’s cool**: Grid search automates hyperparameter tuning, saving us from manually trying different values. It ensures our model is robust by evaluating performance across multiple data splits. Check out [Scikit-learn’s GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) for details.

## Non-Obvious Libraries/Tools Used
- **[Pandas](https://pandas.pydata.org/)**: Used for data manipulation, binning (`pd.cut`), and one-hot encoding (`pd.get_dummies`). Its flexibility makes EDA and preprocessing a breeze.
- **[Seaborn](https://seaborn.pydata.org/)**: Simplifies complex visualizations like bar plots and heatmaps, with built-in support for grouping and hue-based coloring.
- **[Scikit-learn](https://scikit-learn.org/)**: Provides `LabelEncoder`, `StandardScaler`, `GridSearchCV`, and regression models (Lasso, Random Forest, Gradient Boosting, Decision Tree) for robust ML workflows.
- **[Matplotlib](https://matplotlib.org/)**: Used alongside Seaborn for customizing plots, such as setting figure sizes and adding labels to bars.

## Project Folder Structure

- **E-commerce Customer Behavior.py**: The main Python script with EDA, preprocessing, and model training/evaluation.
- **README.md**: This documentation file.
