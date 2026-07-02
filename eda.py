import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('static/plots', exist_ok=True)

df = pd.read_csv('loan_data.csv')

sns.countplot(data=df, x='Loan_Status')
plt.title('Loan Status Distribution')
plt.savefig('static/plots/loan_status.png', dpi=100)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=df, x='Credit_History', hue='Loan_Status', ax=axes[0])
axes[0].set_title('Credit History vs Loan Status')
sns.countplot(data=df, x='Education', hue='Loan_Status', ax=axes[1])
axes[1].set_title('Education vs Loan Status')
plt.tight_layout()
plt.savefig('static/plots/credit_education.png', dpi=100)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data=df, x='ApplicantIncome', hue='Loan_Status', bins=30, ax=axes[0])
axes[0].set_title('Applicant Income Distribution')
sns.histplot(data=df, x='LoanAmount', hue='Loan_Status', bins=30, ax=axes[1])
axes[1].set_title('Loan Amount Distribution')
plt.tight_layout()
plt.savefig('static/plots/income_loan.png', dpi=100)
plt.close()

plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('static/plots/heatmap.png', dpi=100)
plt.close()

print("4 plots saved to static/plots/")
