import pandas as pd
import numpy as np

np.random.seed(42)

n = 614

loan_ids = [f'LP{str(i).zfill(4)}' for i in range(1, n + 1)]

gender = np.random.choice(['Male', 'Female'], n, p=[0.8, 0.2])
married = np.random.choice(['Yes', 'No'], n, p=[0.65, 0.35])
dependents = np.random.choice(['0', '1', '2', '3+'], n, p=[0.4, 0.25, 0.2, 0.15])
education = np.random.choice(['Graduate', 'Not Graduate'], n, p=[0.78, 0.22])
self_employed = np.random.choice(['No', 'Yes'], n, p=[0.85, 0.15])

applicant_income = np.random.gamma(shape=2.5, scale=2000, size=n).astype(int) + 1500
coapplicant_income = (np.random.gamma(shape=1.5, scale=1500, size=n) * np.random.choice([0, 1], n, p=[0.4, 0.6])).astype(int)

loan_amount = np.random.gamma(shape=3, scale=40, size=n).astype(float) + 20
loan_amount = np.clip(loan_amount, 9, 700)

loan_term = np.random.choice([360, 180, 120, 60, 36], n, p=[0.6, 0.2, 0.1, 0.05, 0.05]).astype(float)

property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], n, p=[0.3, 0.4, 0.3])

credit_history = np.random.choice([1, 0], n, p=[0.7, 0.3]).astype(float)

income_bonus = np.clip((applicant_income - 2000) / 20000, 0, 1)
loan_penalty = np.clip(loan_amount / 400, 0, 1)

prob = np.where(
    credit_history == 1,
    0.70 + 0.20 * income_bonus - 0.20 * loan_penalty + 0.05 * (married == 'Yes') + 0.04 * (education == 'Graduate') + 0.03 * (coapplicant_income > 0),
    0.15 + 0.10 * income_bonus - 0.10 * loan_penalty
)
prob = np.clip(prob, 0.05, 0.95)
noise = np.random.normal(0, 0.06, n)
prob = np.clip(prob + noise, 0.05, 0.95)
loan_status = np.where(np.random.binomial(1, prob) == 1, 'Y', 'N')

introduce_some_missing = lambda arr, rate: arr.copy()
gender[np.random.choice(n, int(n * 0.02), replace=False)] = np.nan
married[np.random.choice(n, int(n * 0.01), replace=False)] = np.nan
dependents[np.random.choice(n, int(n * 0.03), replace=False)] = np.nan
self_employed[np.random.choice(n, int(n * 0.03), replace=False)] = np.nan
loan_amount[np.random.choice(n, int(n * 0.04), replace=False)] = np.nan
loan_term[np.random.choice(n, int(n * 0.02), replace=False)] = np.nan
credit_history[np.random.choice(n, int(n * 0.05), replace=False)] = np.nan

df = pd.DataFrame({
    'Loan_ID': loan_ids,
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_history,
    'Property_Area': property_area,
    'Loan_Status': loan_status
})

df.to_csv('loan_data.csv', index=False)
print(f"Synthetic dataset saved: {len(df)} rows, {len(df.columns)} columns")
print(f"\nLoan Status distribution:\n{df['Loan_Status'].value_counts()}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_dict()}")
