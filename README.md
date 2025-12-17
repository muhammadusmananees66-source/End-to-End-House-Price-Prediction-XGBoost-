# End-to-End-House-Price-Prediction-XGBoost-Project

**1. Overview**

**1.1 Project Overview**

An end-to-end machine learning project that predicts California house prices using XGBoost. The workflow includes data loading, preprocessing, exploratory analysis, model training, evaluation with regression metrics, and comparison with baseline models.

**1.2 Dataset overview**

- The dataset contains 20,640 records and 9 numerical features, with no missing values, indicating good data quality.
- MedHouseVal is the target variable, ranging from ~0.15 to ~5.0, and represents median house prices (in hundreds of thousands of dollars).
- Median income (MedInc) shows wide variation and is the most influential socioeconomic feature related to house prices.
- Several features such as Population, AveRooms, and AveOccup are highly skewed and contain extreme outliers.
- AveRooms and AveBedrms are closely related, suggesting multicollinearity between size-related features.
- Latitude and Longitude capture geographical effects, enabling the model to learn spatial price patterns.
- The dataset is well-suited for tree-based models like XGBoost, while linear models may require scaling and outlier handling.

**2. Project Structure**

```bash
california-housing-xgboost/
│
├── data/
│   └── README.md                     # Optional: Describe the data source (California Housing dataset)
│
├── notebooks/
│   ├── EDA_and_Preprocessing.ipynb   # Notebook for EDA, preprocessing, visualizations
│   └── Model_Training_Tuning.ipynb   # Notebook for baseline model, hyperparameter tuning, final model
│
├── src/
│   ├── data_preprocessing.py         # Functions to preprocess, scale, split data
│   ├── model.py                      # Functions to train, evaluate, and save/load XGBoost model
│   └── visualization.py              # Functions for plotting correlation, feature importance, histograms
│
├── models/
│   └── xgboost_california_model.pkl  # Saved trained model
│
├── requirements.txt                  # List of dependencies: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib
├── .gitignore                        # Ignore files like __pycache__, .ipynb_checkpoints, *.pkl
└── README.md                         # Project overview, setup, instructions)))
```
**3. Data Cleaning & Preprocessing**

- The dataset is loaded successfully as a pandas DataFrame with 20,640 rows and 9 columns.
- All columns are of type float64, indicating a fully numerical dataset suitable for regression tasks.
- There are no missing values across any features, reducing the need for imputation during preprocessing.
- The target variable, MedHouseVal, is included within the same DataFrame as the input features.
- The dataset has a compact memory footprint (~1.4 MB), allowing efficient in-memory processing.
- Since all features are numeric, no categorical encoding is required.
- The data is clean and ready for feature scaling, splitting, and model training.

**4. Exploratory Data Analysis (EDA)**

**Histogram**

Below is a clear, report-ready interpretation of each histogram.

**MedInc (Median Income)**

- The distribution is right-skewed, indicating more low-to-mid income areas than very high-income ones.
- Mean is slightly greater than the median, confirming positive skewness.
- Majority of observations lie between 1 and 6, showing a concentrated income range.
- A long right tail suggests outliers or high-income regions.
- Income varies significantly, making it a strong discriminating feature.
- This explains its strong correlation with house prices observed earlier.

**HouseAge**

- The distribution is fairly spread out across the range.
- Multiple peaks indicate non-uniform construction periods.
- Mean and median are close, suggesting near-symmetric distribution.
- No extreme outliers are visible.
- Older and newer houses are well represented.
- Indicates house age alone has limited linear impact on prices.

**AveRooms**

- The distribution is highly right-skewed.
- Most values are concentrated at the lower end.
- Presence of a long tail suggests extreme room counts in some districts.
- Mean is much higher than median due to outliers.
- Indicates potential need for transformation in linear models.
- Tree-based models like XGBoost can handle this skewness well.

**AveBedrms**

- Strong right-skewed distribution similar to AveRooms.
- Majority of values are very small.
- A few extreme values pull the mean to the right.
- Shows redundancy with AveRooms, as seen in correlation analysis.
- Outliers may represent abnormal housing units.
- Can be retained safely for XGBoost but risky for linear models.

**Population**

- Extremely right-skewed distribution.
- Most regions have low population density.
- A small number of districts have very high population.
- Mean is far greater than median, indicating strong skew.
- These extreme values can dominate scale-sensitive models.
- Population shows weak direct relationship with house prices.

**AveOccup (Average Occupancy)**

- Highly right-skewed with extreme outliers.
- Most observations lie close to the lower end.
- Mean is inflated by a few unusually high values.
- Suggests abnormal household sizes in some areas.
- Has very low correlation with house prices.
- Best handled using robust or tree-based models.

**Latitude**

- Distribution shows multiple peaks, indicating geographic clustering.
- Values are concentrated within a narrow band.
- Mean lies centrally, showing balanced spread.
- Reflects regional housing patterns.
- Indicates spatial segmentation of districts.
- Geographic features likely influence price non-linearly.

**Longitude**

- Multimodal distribution similar to Latitude.
- Shows clustering around major geographic regions.
- Mean and median are close.
- Strongly correlated with Latitude.
- Indicates location-based grouping of houses.
- Essential feature for capturing spatial effects.

**MedHouseVal (Target Variable)**

- Distribution is right-skewed, typical for housing prices.
- Mean is higher than median due to expensive properties.
- A visible cap at the upper end suggests price capping in the dataset.
- Majority of values lie in the mid-price range.
- Skewness justifies non-linear modeling.
- Target distribution supports the use of XGBoost.

**Insight**

The dataset exhibits significant skewness and outliers in several numerical features, particularly population and occupancy-related variables. Income and location emerge as dominant drivers, while non-linear relationships justify the use of tree-based models. The EDA strongly points toward XGBoost (or other tree-based models) for this dataset. Here’s why, based on observations:

**1.	Right-skewed features:** 

- Features like MedInc, AveRooms, AveBedrms, Population, and AveOccup have strong positive skew and outliers.
- Linear models (like linear regression) are sensitive to outliers and skewness, often performing poorly unless you transform the data.
- XGBoost, being tree-based, naturally handles skewed distributions and extreme values without needing heavy preprocessing.

**2.	Non-linear relationships:**

- Features like Latitude and Longitude show clustering and likely non-linear effects on MedHouseVal.
- Tree-based models capture such non-linear patterns easily, while linear models would struggle without feature engineering.

**3.	Outliers in target and features:**

- The target (MedHouseVal) has an upper cap and right skew.
- XGBoost can handle this more robustly than linear regression, which would be heavily influenced by extreme house prices.

**4.	Redundant or correlated features:**

- Features like AveRooms and AveBedrms are correlated.
- XGBoost handles multicollinearity well, whereas linear regression would require dropping or combining correlated features.

**5.	Overall EDA insight:**

**Conclusion:**
The dataset exhibits significant skewness and outliers in several numerical features, particularly population and occupancy-related variables. Income and location emerge as dominant drivers, while non-linear relationships justify the use of tree-based models. XGBoost is a strong choice here. If you want, I can also suggest a step-by-step plan to preprocess your data and train an optimized XGBoost model for this dataset.

**5. Correlation Heatmap**

**5.1 Correlation Between Features (Feature–Feature Correlation / Multicollinearity)**

- **Strong multicollinearity detected:** AveRooms and AveBedrms show a very high positive correlation (~0.85), meaning they carry largely redundant information. For linear models, one of them could be dropped to avoid instability.
- Geographical redundancy:** Latitude and Longitude are strongly negatively correlated (~ -0.92), indicating location-related dependence. This is common in spatial data and can introduce multicollinearity in linear regression.
- **Moderate relationship with income:** MedInc has a moderate positive correlation with AveRooms (~0.33), suggesting higher-income areas tend to have houses with more rooms.
- **House age vs population:** HouseAge is moderately negatively correlated with Population (~ -0.30), indicating older housing areas tend to be less densely populated.
- **Weak correlations for occupancy:** AveOccup has near-zero correlation with most features, implying it contributes largely independent information.
- **Most features are weakly correlated otherwise:** Apart from the pairs mentioned above, correlations are relatively low, meaning severe multicollinearity is limited to a few feature pairs. This is generally acceptable for tree-based models like XGBoost but needs care in linear models.
  
**5.2 Correlation Between Features and Target (Feature–Target Correlation)**

- **Strongest predictor:** MedInc has a strong positive correlation (~0.69) with MedHouseVal, indicating median income is the most influential numerical feature for house price prediction.
- **Weak direct impact of size-related features:** AveRooms shows a weak positive correlation (~0.15) and AveBedrms a very weak negative correlation (~ -0.05) with MedHouseVal, suggesting limited linear influence individually.
- **House age effect is small:** HouseAge has a very weak positive correlation (~0.11) with house value, meaning newer vs older houses alone do not strongly determine prices.
- Population & occupancy are poor predictors:** Population and AveOccup have near-zero correlation with MedHouseVal, indicating minimal direct contribution in a linear sense.
- **Location matters but non-linearly:** Latitude shows a moderate negative correlation (~ -0.14) and Longitude a very weak negative correlation, implying geographic influence that is likely non-linear or interaction-based.

**6. Feature Engineering**

**Scaling**

- **Standardization:** Each feature in x_train is scaled to have mean = 0 and standard deviation = 1.
- **Centering data:** The fit_transform method calculates the mean of each feature and subtracts it, centering the data around zero.
- **Scaling variance:** It then divides by the standard deviation of each feature, ensuring all features have the same variance.
- **Improves model performance:** Features with large numeric ranges no longer dominate those with smaller ranges. This is especially important for distance-based models (e.g., KNN, SVM) or gradient-based models (e.g., neural networks).
- **Prepares data for algorithms:** Models like linear regression, logistic regression, SVM, and neural networks converge faster and more reliably when features are standardized.
- **fit_transform vs transform:**
- fit_transform → computes mean & std from training data and scales it.
- transform → uses the same mean & std to scale new/test data consistently.

**7. Model Training**

- xgboost model is preferred for training purpose 
- XGBRegressor is XGBoost’s implementation for regression problems.
- XGBRegressor is a gradient boosting model that builds many decision trees sequentially to predict a continuous target variable.
- **Tree & boosting control parameters**
    - **booster:** type of booster (gbtree, dart, gblinear)
    - **n_estimators:** number of trees to build
    - **learning_rate:** how much each new tree contributes (smaller = slower but more stable learning)
    - **max_depth, max_leaves, grow_policy:** control tree complexity
- **Sampling & regularization parameters**
    - **colsample_bytree, colsample_bylevel, colsample_bynode:** fraction of features used when building trees
    - **gamma:** minimum loss reduction required to make a split
    - **min_child_weight, max_delta_step:** help prevent overfitting
- **Handling data characteristics**
    - **missing:** value treated as missing (NaN by default)
    - **enable_categorical, max_cat_to_onehot:** support for categorical features
    - **monotone_constraints:** enforce monotonic relationships between features and target
- **Training & performance options**
    - **eval_metric:** metric to evaluate model performance (e.g., RMSE, MAE)
    - **early_stopping_rounds:** stops training if no improvement is seen
    - **n_jobs, device:** control parallelism and CPU/GPU usage

**Conclusion**

The output shows all configurable hyperparameters of XGBRegressor, which you tune to balance accuracy, speed, and overfitting.

**8. Model Evaluation**

**Interpretation of Model Results**

- Each number represents a predicted value produced by your regression model (e.g., XGBRegressor).
- The predictions are continuous values (since it’s a regression task).
- The length of this array equals the number of input samples you passed to model.predict(X).
- dtype=float32 indicates predictions are stored as 32-bit floating-point numbers for efficiency.
- During training, the model learns patterns and updates trees/weights.
- During evaluation, the trained model is used to predict unseen data, producing this array.
- On average, model’s squared prediction error is 0.2226.
- rmse = np.sqrt(mse)  # ≈ 0.47
- The R² score (Coefficient of Determination) for XGBoost regression model is R² = 0.83 means
- The model explains ~83% of the variance in the target variable.
- The hyperparameters represent the best trade-off between bias and variance found by GridSearchCV.
- XGBoost model achieves highest R² (~0.844) on cross-validated training data.
- The predictions are closely aligned with the actual values.
- **Rule of thumb**
  - R² < 0.3 → weak model
  - 0.3 – 0.6 → moderate
  - 0.6 – 0.8 → good
  - > 0.8 → very good / strong

**Paramerter tuning**

- We tuned hyperparameters of the XGBRegressor using GridSearchCV.
- GridSearchCV trains multiple models with different parameter combinations and selects the best-performing model based on an evaluation metric (e.g., lowest MSE or highest R²).
- Default XGBoost parameters may not be optimal for the dataset.
- As Tuning helps:
  - Improve model accuracy
  - Reduce overfitting / underfitting
  - Find the best bias–variance trade-off

**9. Interpretation of Feature Importance**

**Feature importances**

- Each number corresponds to a feature used by the model.
- Higher values → the feature is more important for predicting the target (prices).
- Values are normalized (sum to 1 in most implementations), showing relative contribution.

**Interpretation of values**

- In our Model, most important feature: 0.353. This feature contributes ~35% of the predictive power.
- Least important features: 0.025 and 0.028. These features contribute very little to predictions.
- Moderately important features: 0.131, 0.157, 0.155. Still meaningful; the model uses them, but they are less dominant.
- Feature importances helps understand which features drive the predictions.
- Drop very low-importance features to simplify the model.
- Focus on high-importance features for insights or explanation.
- These values show how much each feature contributed to the XGBoost model’s prediction of prices, with some features being much more influential than others.

**1. MedInc (Median Income)**

- Most Important as it dominates the model
- Indicates house prices are strongly driven by income levels
- Higher neighborhood income → higher house prices
- This aligns perfectly with real-world economics

**2. Latitude & Longitude**

- They are the second strongest predictors
- Shows the model is learning geographical price patterns

**3. AveOccup (Average Occupancy)**

- Reflects crowding / household density
- Higher occupancy can indicate:
- Dense urban areas, smaller or lower-priced homes
- Moderate influence on price

**4. AveRooms**

- Indicates house size
- Larger average rooms → generally higher prices
- Less impact than income and location

**5. HouseAge**

- Older houses slightly affect price
- Impact is smaller because:
- Age alone doesn’t capture renovation or location quality

**6. AveBedrms & Population — Least Important**

- Minimal contribution
- Redundant with other features
- Weak standalone predictive power
- These features add limited new information

**Insights**

The model relies primarily on economic (income) and geographical (location) features to predict house prices, while structural and demographic features play a secondary role.

This shows:
- Correct feature learning
- No obvious data leakage
- Real-world alignment

**10. Making Predictions on Data**

- We used the California Housing dataset
- Target (MedHouseVal) is in $100,000 units
- Model Output	Actual Price
```bash
0.55	~$55,000
1.75	~$175,000
4.72	~$472,000
5.26	~$526,000 )))
```
So model is making realistic predictions 

**Requirements**

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib
- License
**11. Limitations of Current Model**

- **Limited features:** Dataset lacks critical insurance cost drivers such as medical history, claim history, and lifestyle details, leading to weak explanatory power.
- **Low R² scores (~10–15%):** Models explain only a small portion of cost variance, reflecting the complexity of real-world insurance pricing and missing variables.
- **Small dataset size:** Limited number of samples restricts model generalization and reduces the effectiveness of complex models like Random Forest and XGBoost.
- **High prediction error:** MAE around $1,050–$1,120 makes the model unsuitable for precise premium estimation or financial decision-making.
- **Outlier sensitivity:** Extreme medical cost cases increase RMSE and negatively impact model stability.
- **No production constraints:** The project does not address fairness, regulatory compliance, model monitoring, or concept drift handling.

**12. Future Improvements (Highly Recommended)**

- RandomizedSearchCV
- Optuna
- Try Other Models
- CatBoost
- LightGBM
- Logistic Regression with class weights

**13. Deploy on AWS SageMaker**

- Trained models are packaged and uploaded to Amazon S3.
- Amazon SageMaker is used to create a managed inference endpoint for real-time predictions.
- A custom inference script (inference.py) handles input preprocessing and output formatting.
- The model is deployed using a SageMaker endpoint, enabling scalable and low-latency predictions.
- Endpoint testing is performed using Boto3 to validate successful deployment.
- This deployment demonstrates end-to-end ML workflow, from training to production inference in a cloud environment.

**14. Conclusion**

This project demonstrates an End-to-End-House-Price-Prediction-XGBoost, covering data preprocessing, exploratory analysis, model training, evaluation, and Future cloud deployment. XGBOOST model was implemented, with XGBoost achieving the best performance among the tested approaches. Although overall predictive power remained limited due to the constrained feature set and dataset size, the results reflect real-world challenges in House Price prediction modeling. The project highlights practical ML engineering skills, including in future deployment on AWS SageMaker, and serves as a strong foundation for further improvements through feature enrichment, hyperparameter tuning, and production-grade MLOps practices.





