 AquaSafe Durban

 Machine-Learning Forecasting of Community Water-Service Events in eThekwini


 1. Project overview

AquaSafe Durban is an educational machine-learning project that analyses factual community water-service events documented in official eThekwini Municipality public notices. The project builds a historical community-risk view and an exploratory system-wide forecast of future affected-area activity.

The project includes the complete ML lifecycle in a Jupyter Notebook and an interactive Streamlit dashboard for presenting the results.

Main objective

To investigate whether historical eThekwini water-event records can be used to forecast the next week's number of affected-area records, compare three regression algorithms fairly using chronological validation, select the best-performing model, and communicate the results through an interactive dashboard.



2. Dataset

The source dataset contains factual water-service events compiled from official eThekwini Municipality public notices.

- Affected-area rows: 651
- Independent municipal events: 32
- Unique community / area labels: 335
- Date coverage: 15 February 2023 to 31 August 2026
- Municipality: eThekwini Metropolitan Municipality
- Province: KwaZulu-Natal
- Country: South Africa

One municipal event may affect many communities. For this reason, affected-area rows are not treated as independent incidents.

3. Machine-learning problem

The supervised-learning target is:

Next week's number of affected-area records in official eThekwini water notices.

A system-wide target is used because the public-notice dataset contains too few independent events to support a reliable suburb-specific forecasting claim.

Time-series features

The notebook engineers features including:

- current weekly affected-area count;
- lagged counts at 1, 2, 3, 4, 8 and 12 weeks;
- rolling means and standard deviations over 4, 8 and 12 weeks;
- month-based sine and cosine seasonality features; and
- a time-trend feature.

4. Models compared

Exactly three regression models are compared using the same features and the same chronological validation folds:

1. Linear Regression
2. Decision Tree
3. Random Forest

The notebook uses 5-fold `TimeSeriesSplit` walk-forward validation rather than a random train/test split. This helps prevent future observations from leaking into earlier training periods.

Model-selection rule

The final model is selected using the lowest mean RMSE (Root Mean Squared Error) across the walk-forward folds. RMSE is used as the primary metric because it gives larger forecasting errors more weight.



5. Model results

| Model | Mean MAE | Mean RMSE | Mean R² | Selection |
|---|---:|---:|---:|---|
| Random Forest | 8.255 | 11.380 | -3.077 | Selected |
| Decision Tree | 8.879 | 12.011 | -3.396 | |
| Linear Regression | 7.547 | 12.214 | -2.009 | |

Random Forest is selected because it achieved the lowest mean RMSE: **11.380**.

Although Linear Regression has the lowest mean MAE, the project uses RMSE as the pre-defined primary selection metric. The negative mean R² values also show that this sparse public-notice dataset is difficult to forecast. Therefore, AquaSafe should be treated as an educational and exploratory forecasting system, not as an operational municipal outage-warning service.

6. Forecast outputs

The trained Random Forest model is used to generate:

- a recursive 12-week forecast; and
- summary forecasts for approximately 7, 14 and 30 days.

Current saved forecast summary:

| Horizon | Forecast affected-area records |
|---|---:|
| Next 7 days | 4.82 |
| Next 14 days | 10.60 |
| Next 30 days | 20.99 |

These are system-wide affected-area activity forecasts. They are not probabilities that a specific household or suburb will lose water.

 7. Historical community-risk index

The project also creates an explainable historical community-risk score:

- 50% frequency percentile;
- 30% recency score; and
- 20% unplanned-event share.

This score summarises documented historical patterns. It is not a claim that a community currently has no water or that a future interruption is certain.

 8. Interactive Streamlit dashboard

The Streamlit application includes:

- project overview and key metrics;
- historical weekly water-event activity;
- historical community-risk ranking;
- event-category visualisation;
- a live official eThekwini suburb polygon map when the municipal GIS service is available;
- 7-, 14- and 30-day forecast summaries;
- a 12-week forecast visualisation;
- model leaderboard and RMSE comparison;
- event filtering and official source links; and
- data limitations, forecasting scope and privacy notes.

The live map requires internet access to query the official eThekwini GIS service.

9. Repository structure

text
AquaSafe-Durban/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── AquaSafe_Final_3_Model_ML_Lifecycle.ipynb
├── aquasafe_best_selected_model.joblib
├── AquaSafe_eThekwini_Official_Water_Events.csv
├── AquaSafe_Upgraded_Community_Risk_Snapshot.csv
├── AquaSafe_Upgraded_Weekly_System_Series.csv
├── AquaSafe_3_Model_Leaderboard.csv
├── AquaSafe_RF_Forecast_7_14_30_Days.csv
├── AquaSafe_RF_12_Week_Forecast.csv
└── SCHOOL_SUBMISSION_CHECKLIST.md

 10. How to run the project locally

A. Clone or download the repository

bash
git clone https://github.com/YOUR-USERNAME/AquaSafe-Durban.git
cd AquaSafe-Durban

B. Create a virtual environment (recommended)

Windows PowerShell:

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

C. Install the dependencies

```powershell
python -m pip install -r requirements.txt
```
D. Run the Streamlit dashboard

powershell
python -m streamlit run app.py

11. How to run the notebook

Open:

text
AquaSafe_Final_3_Model_ML_Lifecycle.ipynb


in Jupyter Notebook, JupyterLab or VS Code and run the cells from top to bottom. The notebook expects the main source CSV to be in the same directory.

Running the complete notebook regenerates the risk table, weekly time-series data, model leaderboard, trained-model bundle and forecast CSV outputs.



12. Streamlit Community Cloud deployment

For Streamlit Community Cloud, connect this repository to Streamlit and use:

- **Branch:** `main`
- **Main file path:** `app.py`

The file must be named `requirements.txt` so Streamlit can install the required Python packages automatically.

13. Reproducibility

The project uses `random_state=42` for the tree-based models. The Random Forest configuration in the notebook uses 500 trees and the same chronological validation strategy used for all three models.

The final trained model is stored in:

text
aquasafe_best_selected_model.joblib


The dashboard itself currently reads the saved CSV outputs rather than loading the `.joblib` model at runtime. The model file is included in the repository as evidence of the completed ML lifecycle and for reproducibility.

14. Limitations

Important limitations include:

- only 32 independent municipal events are represented;
- one event can produce many affected-area rows;
- official public notices are not the municipality's complete operational fault log;
- zero-event weeks mean no matching public-notice records were captured for that week, not proof that no household experienced a water problem;
- negative mean R² values indicate difficult forecasting conditions; and
- future forecasts are exploratory and should not be interpreted as guaranteed outages.



15. Privacy and responsible use

AquaSafe does not use resident names, telephone numbers or email addresses. The application presents historical and forecast information for educational analysis only and should not be represented as an official eThekwini Municipality service.


16. ML lifecycle covered

The Jupyter Notebook documents:

1. imports and configuration;
2. factual-data loading and ML problem definition;
3. data-quality auditing;
4. exploratory data analysis;
5. historical community-risk engineering;
6. complete weekly time-series construction;
7. time-series feature engineering;
8. walk-forward validation and three-model comparison;
9. automatic best-model selection;
10. out-of-fold evaluation;
11. final model fitting and saving;
12. model explanation / feature importance;
13. recursive 12-week forecasting;
14. 7-, 14- and 30-day forecast summaries;
15. real eThekwini suburb-map integration;
16. saving ML lifecycle outputs; and
17. monitoring and retraining planning.



