# Smart Lender - Loan Eligibility Predictor

Flask web app with XGBoost model predicting loan eligibility.

## Setup
1. `python -m venv venv`
2. `.\venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. Place `loan_data.csv` in project root
5. `python preprocessing.py`
6. `python train_model.py`
7. `python eda.py`
8. `python app.py`

## Run
`python app.py` → http://127.0.0.1:5000

## Test
`pytest tests\test_app.py -v`

## Deploy
`ibmcloud cf push`
