# -Interactive-Mapping-Application-for-Environmental-Monitoring-and-Forecasting
1. Project Overview

This project develops an integrated environmental monitoring and NO₂
prediction system using UK air-quality data. The system combines machine
learning prediction, SHAP-based explainability, K-Means spatial
clustering, Folium mapping and a Flask web application.

The project uses two complementary datasets:

Primary dataset: airQuality_2024 data.csv from DEFRA UK-AIR.
It provides the environmental observations used for exploratory
analysis, preprocessing, feature engineering and machine-learning
modelling.

Secondary dataset: AURN_2015_2023.csv. It provides
monitoring-station geographical information, particularly station
name, latitude and longitude, for spatial clustering and hotspot
visualisation.

The datasets support different parts of the project rather than being
treated as two independent prediction datasets.

2. Research Aim

The aim is to develop and evaluate an integrated environmental
monitoring framework that can predict NO₂ concentrations, explain model
predictions, identify relative geographical hotspots and present the
results through a web-based prototype.

3. Main Objectives

Analyse temporal and spatial patterns in UK NO₂ observations.

Preprocess and prepare the DEFRA UK-AIR data for modelling.

Develop temporal and historical NO₂ features.

Compare Linear Regression, Decision Tree, Random Forest and XGBoost.

Evaluate models using MAE, RMSE and R².

Investigate unusually high preliminary model performance and
potential target leakage.

Apply SHAP to explain the final Random Forest predictions.

Apply K-Means clustering to identify relative geographical
groupings.

Visualise monitoring locations using Folium.

Integrate the final Random Forest model into a Flask application.

4. Dataset Details

Primary Dataset

File: airQuality_2024 data.csv

Source: DEFRA UK-AIR

Recorded structure: approximately 1,048,575 rows and 10 columns.

Purpose:

Exploratory Data Analysis

Data preprocessing

NO₂ target extraction

Temporal feature engineering

Historical feature engineering

Machine-learning model development

Secondary Dataset

File: AURN_2015_2023.csv

Purpose:

Obtain monitoring-station coordinates

Support geographical analysis

K-Means spatial clustering

Relative hotspot identification

Folium map visualisation

The spatial analysis uses station name, latitude and longitude from the
secondary dataset.

5. Final Prediction Features

The corrected final Random Forest model uses six features:

Hour

Day

Month

Weekday

NO2_lag1_corrected

NO2_lag2_corrected

The corrected lag features are constructed from previous observations
for the same monitoring station.

The original rolling feature was removed from the final prediction model
after feature-validation experiments identified a potential
target-leakage problem with the original construction.

6. Machine Learning Models

Four regression models were evaluated:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

XGBoost Regressor

Final Experiment D Results

Model                        MAE         RMSE           R²

Linear Regression         4.0435       6.1987       0.8405
Decision Tree             6.1917       9.2909       0.6417
Random Forest         3.8766   5.9625   0.8524
XGBoost                   3.9394       6.0162       0.8498

Random Forest achieved the strongest overall performance across the
three evaluation metrics in the corrected final experiment.

7. Feature Validation and Target Leakage

Initial experiments produced unusually high model performance. In the
original feature configuration, the rolling NO₂ feature was constructed
in a way that incorporated the current target observation.

This created a potential target-leakage problem.

Controlled experiments were therefore performed:

Experiment A: Original feature configuration

Experiment B: Temporal features without historical NO₂ features

Experiment C: Corrected historical features

Experiment D: Corrected lag features only

The final model is based on Experiment D.

The final configuration excludes the problematic original rolling
feature and uses station-specific historical lag variables based only on
previous observations.

This validation step is important because the final performance should
reflect information that would be available before the prediction point.

8. SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to interpret the final
Random Forest model.

The corrected SHAP analysis should use:

explainer = shap.TreeExplainer(final_rf_model)
shap_values = explainer.shap_values(X_D_test)

The analysis is based on the final six prediction features:

Hour

Day

Month

Weekday

NO₂ Lag 1 corrected

NO₂ Lag 2 corrected

SHAP values describe the contribution of features to model predictions.
They should be interpreted as model associations and explanations rather
than causal effects.

9. Spatial Analysis

K-Means clustering is used to group monitoring locations geographically.

Configuration

Method: K-Means

Spatial variables: Latitude and Longitude

Number of clusters: 5

Relative hotspot: Cluster 3

Mean NO₂ concentration: approximately 18.71 µg/m³

Cluster 3 is described as a relative cluster-level hotspot because
it has the highest mean NO₂ concentration among the five clusters.

The result does not represent a statistically validated hotspot or a
continuous UK-wide pollution map.

10. Folium Mapping

Folium is used to visualise the monitoring locations and their K-Means
cluster assignments.

The map provides geographical context for the spatial analysis. It
represents available monitoring locations rather than predicted
pollution concentrations across unmonitored areas.

11. Flask Application

The project includes a local Flask prototype for NO₂ prediction.

The application loads the corrected Random Forest model:

rf_model_corrected.pkl

The prediction interface accepts:

Hour
Day
Month
Weekday
NO₂ Lag 1
NO₂ Lag 2

The application validates input values before passing them to the model.

Example Scenario Tests

Scenario                Inputs                  Prediction / Response

Normal                  Hour 14, Day 15, Month  19.69 µg/m³
6, Weekday 3, Lag1
18.5, Lag2 17.8

High recent NO₂         Hour 16, Day 15, Month  56.44 µg/m³
6, Weekday 3, Lag1 60,
Lag2 55

Low recent NO₂          Hour 16, Day 15, Month  6.81 µg/m³
6, Weekday 3, Lag1 6,
Lag2 5

Invalid input           Hour 25                 Input rejected

The Flask system is a local prototype and is not intended to represent
production deployment.

12. Project Structure

A recommended project structure is:

UK-NO2-Environmental-Monitoring/
│
├── data/
│   ├── airQuality_2024 data.csv
│   └── AURN_2015_2023.csv
│
├── notebooks/
│   └── Akash_aston_final (1).ipynb
│
├── models/
│   └── rf_model_corrected.pkl
│
├── flask_app/
│   ├── app.py
│   ├── templates/
│   │   ├── home.html
│   │   └── index.html
│   └── uk_no2_hotspot_map.html
│
├── figures/
│   ├── final_corrected_model_comparison_actual.png
│   ├── final_MAE_comparison_actual.png
│   ├── final_R2_comparison_actual.png
│   ├── linear_regression_regression_performance.png
│   ├── decision_tree_regression_performance.png
│   ├── random_forest_regression_performance.png
│   └── xgboost_regression_performance.png
│
└── README.md

13. Software and Libraries

The project uses Python and common data-science libraries, including:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
joblib
flask
folium

Install the main packages with:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib flask folium

14. Running the Notebook

Open:

Final submission.ipynb

The notebook contains the main workflow:

Dataset Loading
      ↓
Data Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Temporal Feature Engineering
      ↓
Historical NO₂ Features
      ↓
Feature Validation
      ↓
Experiment A–D
      ↓
Final Random Forest
      ↓
SHAP Explainability
      ↓
K-Means Spatial Clustering
      ↓
Folium Visualisation
      ↓
Model Saving
      ↓
Flask Integration

Run the corrected modelling and SHAP sections after the required
datasets and variables have been loaded.

15. Running the Flask Application

Place the saved model in the Flask application directory:

rf_model_corrected.pkl

Then run:

python app.py

The application will normally be available locally at:

http://127.0.0.1:5000/

The prediction page can then be used to enter the six required model
features.

16. Final Model File

The final deployed model is:

rf_model_corrected.pkl

It corresponds to the corrected Random Forest configuration using six
features.

Do not replace this model with the earlier model that used the original
rolling NO₂ feature.

17. Important Interpretation Notes

The final Random Forest result is based on the corrected feature
configuration.

The original near-perfect results are preliminary diagnostic results
and should not be treated as final predictive performance.

Historical lag features are valid when they use observations
available before the prediction point.

The original incorrectly constructed rolling feature was removed
from the final model.

SHAP explains model behaviour and does not establish causal
relationships.

K-Means identifies relative geographical groupings rather than
statistically validated pollution hotspots.

Folium visualises monitoring locations and cluster assignments
rather than continuous pollution across unmonitored areas.

The Flask application is a local prototype.

18. Key Final Result

The corrected Random Forest Regressor produced:

MAE  = 3.8766
RMSE = 5.9625
R²   = 0.8524

It was selected as the final model for SHAP interpretation and Flask
integration.

19. Future Work

Potential future improvements include:

External validation using independent monitoring stations

Longer temporal evaluation

Additional meteorological variables

Spatial-temporal modelling

Statistically validated hotspot methods

Automated data updates

Cloud deployment

Broader user evaluation of the application

20. Dissertation Context

This repository supports the MSc dissertation project on:

Machine Learning-Based UK NO₂ Prediction with Explainable AI, Spatial
Hotspot Analysis and Flask Integration

The project integrates:

Environmental Data
        ↓
Machine Learning
        ↓
NO₂ Prediction
        ↓
SHAP Explainability
        ↓
K-Means Spatial Analysis
        ↓
Folium Visualisation
        ↓
Flask Web Application
