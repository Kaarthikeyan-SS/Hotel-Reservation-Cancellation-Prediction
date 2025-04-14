# 🏨 Hotel Reservation Cancellation Prediction

This machine learning project focuses on predicting hotel booking cancellations based on various booking details and customer behavior patterns. The notebook performs thorough exploratory data analysis, statistical testing, feature engineering, preprocessing, and model building using a range of classification algorithms.

---

## 📌 Project Summary

Hotel booking cancellations can lead to revenue loss and operational inefficiencies. This project aims to use machine learning to identify which bookings are most likely to be canceled, allowing hotels to take proactive measures.

We use a publicly available dataset of 36,000+ hotel reservations, process the data, analyze trends, and train models to accurately predict cancellations.

---

## 🗃️ Dataset Information

- **File:** `Hotel Reservations.csv`
- **Rows:** 36,275
- **Columns:** 19
- **Target Variable:** `booking_status` – indicates whether a booking was canceled or not.

### 🔑 Key Features:

| Column Name                         | Description |
|------------------------------------|-------------|
| `no_of_adults`                     | Number of adults in the booking |
| `no_of_children`                   | Number of children in the booking |
| `no_of_weekend_nights`            | Weekend nights included in the stay |
| `no_of_week_nights`               | Weekday nights included in the stay |
| `type_of_meal_plan`               | Type of meal plan selected |
| `required_car_parking_space`      | Indicates if a parking space is needed |
| `room_type_reserved`              | Type of room reserved |
| `lead_time`                       | Number of days between booking and check-in |
| `arrival_year`, `month`, `date`   | Components of arrival date |
| `market_segment_type`             | Source of booking (e.g., Online, Offline) |
| `repeated_guest`                  | Indicates if the guest is a returning customer |
| `no_of_previous_cancellations`    | How many times the customer canceled previously |
| `avg_price_per_room`              | Average room price per night |
| `no_of_special_requests`          | Number of special requests made |
| `booking_status`                  | Target variable: Canceled / Not_Canceled |

---

## 🎯 Project Goals

- Understand trends and behaviors that lead to cancellations
- Identify significant features using statistical testing
- Build robust classification models to predict cancellations
- Evaluate models based on accuracy, recall, and AUC
- Select the most effective model for deployment

---

## 📊 Exploratory Data Analysis

The notebook starts with univariate and bivariate analysis using plots like:

- Count plots for categorical variables
- Histograms and box plots for numerical features
- Heatmap to show feature correlations
- Booking cancellation rates by segment, time, and room type

---

## 📈 Statistical Testing

To ensure meaningful features are selected, several statistical tests were used:

- **Shapiro-Wilk Test**: Check normality of distributions
- **Levene’s Test**: Test equality of variances
- **Z-score**: Detect outliers
- **Variance Inflation Factor (VIF)**: Detect multicollinearity
- **Mann-Whitney U Test**: Compare means for non-normally distributed features

These tests helped eliminate irrelevant or redundant features before model training.

---

## 🧹 Data Preprocessing

- Removed unused columns (`Booking_ID`, etc.)
- Encoded categorical variables using `LabelEncoder`
- Used `RobustScaler` for scaling to reduce influence of outliers
- Train-test split using `train_test_split` (typically 80:20)

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

1. **Logistic Regression** – simple baseline model
2. **Decision Tree Classifier**
3. **Random Forest Classifier** – ensemble of decision trees
4. **AdaBoost Classifier** – boosting approach
5. **Gradient Boosting Classifier** – uses gradient descent to improve predictions
6. **XGBoost Classifier** – highly optimized and regularized boosting method

### 🔍 Hyperparameter Tuning
Used `GridSearchCV` to find the best parameters for:
- Decision Trees
- Random Forest
- XGBoost

---

## 📊 Model Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | How often the model is correct overall |
| **Recall** | Ability to correctly identify cancellations (important!) |
| **Precision** | How many predicted cancellations were actually correct |
| **F1-score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Measures classifier’s ability to distinguish classes |

### 📈 ROC Curve:
Visualizes trade-offs between true positive and false positive rates across thresholds for each model.

---

## 🏆 Final Results

After tuning and comparison, **XGBoost** emerged as the best-performing model due to:

- Highest Recall (very important to identify cancellations)
- Strong AUC score (good discrimination between classes)
- Balance of performance across all metrics

---
