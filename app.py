from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'smartlender-enterprise-2026'

scaler = joblib.load('scaler.pkl')

MODELS = {
    'XGBoost': joblib.load('xgboost_model.pkl'),
}
DEFAULT_MODEL = 'XGBoost'

MODEL_INFO = {
    'XGBoost': {'name': 'XGBoost Classifier', 'accuracy': '86.2%'},
}

FEATURE_ORDER = ['Gender', 'Married', 'Dependents', 'Education',
                 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                 'Property_Area']

GENDER_MAP = {'Male': 1, 'Female': 0}
MARRIED_MAP = {'Yes': 1, 'No': 0}
DEP_MAP = {'0': 0, '1': 1, '2': 2, '3+': 3}
EDU_MAP = {'Graduate': 1, 'Not Graduate': 0}
SELF_EMP_MAP = {'Yes': 1, 'No': 0}
AREA_MAP = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}

HISTORY_FILE = 'prediction_history.json'

FEATURE_LABELS = {
    'Gender': 'Gender', 'Married': 'Marital Status', 'Dependents': 'Dependents',
    'Education': 'Education', 'Self_Employed': 'Employment',
    'ApplicantIncome': 'Applicant Income', 'CoapplicantIncome': 'Coapplicant Income',
    'LoanAmount': 'Loan Amount', 'Loan_Amount_Term': 'Loan Term',
    'Credit_History': 'Credit History', 'Property_Area': 'Property Area'
}

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def get_kpi_stats():
    history = load_history()
    total = len(history)
    approved = sum(1 for h in history if h.get('result') == 'Approved')
    rejected = sum(1 for h in history if h.get('result') == 'Rejected')
    avg_conf = np.mean([h.get('confidence', 0) for h in history]) if history else 0
    avg_income = np.mean([h.get('income', 0) for h in history]) if history else 0
    avg_loan = np.mean([h.get('loan_amount', 0) for h in history]) if history else 0
    approval_rate = (approved / total * 100) if total > 0 else 0

    last_model = history[-1].get('model_used', DEFAULT_MODEL) if history else DEFAULT_MODEL
    model_info = MODEL_INFO.get(last_model, MODEL_INFO[DEFAULT_MODEL])

    return {
        'total': total, 'approved': approved, 'rejected': rejected,
        'approval_rate': f'{approval_rate:.1f}',
        'avg_confidence': f'{avg_conf:.1%}' if history else '0%',
        'avg_income': f'${avg_income:,.0f}' if history else '$0',
        'avg_loan': f'${avg_loan:,.0f}' if history else '$0',
        'model_accuracy': model_info['accuracy'], 'model_name': last_model
    }

def get_feature_importance(raw_data):
    importance = {
        'Credit History': 0.35 if raw_data.get('Credit_History', 0) == 1 else -0.25,
        'Applicant Income': 0.15 * min(raw_data.get('ApplicantIncome', 0) / 15000, 1),
        'Loan Amount': -0.12 * min(raw_data.get('LoanAmount', 0) / 500, 1),
        'Marital Status': 0.08 if raw_data.get('Married', 0) == 1 else 0,
        'Education': 0.06 if raw_data.get('Education', 0) == 1 else 0,
        'Coapplicant Income': 0.05 * min(raw_data.get('CoapplicantIncome', 0) / 5000, 1),
        'Property Area': 0.04 if raw_data.get('Property_Area', 0) == 1 else 0,
        'Self Employed': -0.04 if raw_data.get('Self_Employed', 0) == 1 else 0,
        'Dependents': -0.03 * DEP_MAP.get(str(raw_data.get('Dependents', '0')), 0),
        'Loan Term': 0.02 if raw_data.get('Loan_Amount_Term', 0) == 360 else -0.01
    }
    return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))

def generate_insights(raw_data, result, prob):
    strengths = []
    weaknesses = []
    if raw_data.get('Credit_History') == 1:
        strengths.append('Strong credit history indicates reliable repayment behavior')
    else:
        weaknesses.append('Poor credit history is the primary risk factor')
    if raw_data.get('ApplicantIncome', 0) > 5000:
        strengths.append(f'Good income level (${raw_data["ApplicantIncome"]:,.0f}/year)')
    else:
        weaknesses.append(f'Income (${raw_data["ApplicantIncome"]:,.0f}/year) is below average')
    if raw_data.get('LoanAmount', 0) < 200:
        strengths.append('Conservative loan amount relative to income')
    else:
        weaknesses.append('High loan amount increases default risk')
    if raw_data.get('Married') == 1:
        strengths.append('Married applicants show higher stability')
    if raw_data.get('Education') == 1:
        strengths.append('Graduate education correlates with higher income')
    if raw_data.get('Self_Employed') == 1:
        weaknesses.append('Self-employment adds income variability risk')
    if raw_data.get('CoapplicantIncome', 0) > 0:
        strengths.append('Co-applicant income strengthens application')

    risk_score = max(0, min(100, int((1 - prob) * 100)))
    loan_amt = raw_data.get('LoanAmount', 0)
    if loan_amt > 1000:
        loan_amt = loan_amt / 1000
    term = raw_data.get('Loan_Amount_Term', 360)
    if term <= 0:
        term = 360
    rate = 0.008
    if loan_amt > 0:
        emi = (loan_amt * 1000 * rate * (1 + rate) ** term) / ((1 + rate) ** term - 1)
    else:
        emi = 0

    return {
        'strengths': strengths, 'weaknesses': weaknesses,
        'risk_score': risk_score, 'estimated_emi': f'${emi:,.2f}',
        'default_risk': f'{(1-prob)*100:.1f}%',
        'factors': [
            {'name': 'Credit History', 'impact': 'high', 'positive': raw_data.get('Credit_History') == 1},
            {'name': 'Income Level', 'impact': 'high', 'positive': raw_data.get('ApplicantIncome', 0) > 5000},
            {'name': 'Loan Amount', 'impact': 'medium', 'positive': raw_data.get('LoanAmount', 0) < 200},
            {'name': 'Employment', 'impact': 'low', 'positive': raw_data.get('Self_Employed') == 0},
            {'name': 'Education', 'impact': 'low', 'positive': raw_data.get('Education') == 1},
        ]
    }

@app.route('/')
def dashboard():
    stats = get_kpi_stats()
    history = load_history()[-5:]
    return render_template('dashboard.html', stats=stats, recent=history)

@app.route('/predict-page')
def predict_page():
    history = load_history()
    last_model = history[-1].get('model_used', DEFAULT_MODEL) if history else DEFAULT_MODEL
    model_info = MODEL_INFO.get(last_model, MODEL_INFO[DEFAULT_MODEL])
    return render_template('predict.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        loan_amt_raw = float(request.form['LoanAmount'])
        if loan_amt_raw > 1000:
            loan_amt_raw = loan_amt_raw / 1000
        term_raw = float(request.form['Loan_Amount_Term'])
        if term_raw <= 0:
            term_raw = 360
        raw = {
            'Gender': GENDER_MAP[request.form['Gender']],
            'Married': MARRIED_MAP[request.form['Married']],
            'Dependents': DEP_MAP[request.form['Dependents']],
            'Education': EDU_MAP[request.form['Education']],
            'Self_Employed': SELF_EMP_MAP[request.form['Self_Employed']],
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': loan_amt_raw,
            'Loan_Amount_Term': term_raw,
            'Credit_History': int(request.form['Credit_History']),
            'Property_Area': AREA_MAP[request.form['Property_Area']]
        }
        model_name = request.form.get('model_name', DEFAULT_MODEL)
        if model_name not in MODELS:
            model_name = DEFAULT_MODEL
        selected_model = MODELS[model_name]

        df = pd.DataFrame([raw])[FEATURE_ORDER]
        scaled = scaler.transform(df)
        pred = int(selected_model.predict(scaled)[0])
        prob = float(selected_model.predict_proba(scaled)[0][1])
        result = 'Approved' if pred == 1 else 'Rejected'

        importance = get_feature_importance(raw)
        insights = generate_insights(raw, result, prob)
        risk_score = insights['risk_score']
        emi = insights['estimated_emi']

        record = {
            'id': f'SL-{datetime.now().strftime("%Y%m%d%H%M%S")}',
            'timestamp': datetime.now().isoformat(),
            'result': result, 'probability': f'{prob:.1%}', 'confidence': prob,
            'risk_score': risk_score, 'emi': emi, 'model_used': model_name,
            'income': raw['ApplicantIncome'], 'loan_amount': raw['LoanAmount'],
            'applicant': f'Applicant {datetime.now().strftime("%H:%M")}',
            'form_data': {
                'Gender': request.form['Gender'], 'Married': request.form['Married'],
                'Dependents': request.form['Dependents'], 'Education': request.form['Education'],
                'Self_Employed': request.form['Self_Employed'],
                'ApplicantIncome': raw['ApplicantIncome'],
                'CoapplicantIncome': raw['CoapplicantIncome'],
                'LoanAmount': raw['LoanAmount'],
                'Loan_Amount_Term': raw['Loan_Amount_Term'],
                'Credit_History': raw['Credit_History'],
                'Property_Area': request.form['Property_Area']
            }
        }
        history = load_history()
        history.append(record)
        save_history(history)

        model_info = MODEL_INFO.get(model_name, MODEL_INFO[DEFAULT_MODEL])

        return render_template('result.html', result=result, probability=f'{prob:.1%}',
                             record=record, importance=importance, insights=insights, model_info=model_info)
    except Exception as e:
        return render_template('result.html', result='Error', probability=str(e),
                             record={}, importance={}, insights={'strengths':[], 'weaknesses':[]},
                             model_info=MODEL_INFO[DEFAULT_MODEL])

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/history')
def history():
    records = load_history()
    records.reverse()
    return render_template('history.html', records=records)

@app.route('/insights')
def insights():
    records = load_history()
    last = records[-1] if records else None
    importance = get_feature_importance(last.get('form_data', {})) if last else {}
    insight_data = generate_insights(last.get('form_data', {}), last.get('result', ''), last.get('confidence', 0)) if last else {'strengths':[], 'weaknesses':[]}
    model_name = last.get('model_used', DEFAULT_MODEL) if last else DEFAULT_MODEL
    model_info = MODEL_INFO.get(model_name, MODEL_INFO[DEFAULT_MODEL])
    return render_template('insights.html', last=last, importance=importance, insights=insight_data, model_info=model_info)

@app.route('/api/history', methods=['GET'])
def api_history():
    return jsonify(load_history())

@app.route('/api/stats', methods=['GET'])
def api_stats():
    return jsonify(get_kpi_stats())

@app.route('/api/delete/<record_id>', methods=['POST'])
def delete_record(record_id):
    history = load_history()
    history = [h for h in history if h.get('id') != record_id]
    save_history(history)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
