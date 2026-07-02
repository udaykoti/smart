from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from pathlib import Path
import traceback

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'smartlender-enterprise-2026'

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Configure template and static folders with proper paths
template_folder = BASE_DIR / 'templates'
static_folder = BASE_DIR / 'static'

# Update Flask config
app.template_folder = str(template_folder)
app.static_folder = str(static_folder)
app.static_url_path = '/static'

# Initialize model variables
scaler = None
MODELS = {}
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

HISTORY_FILE = str(BASE_DIR / 'prediction_history.json')

FEATURE_LABELS = {
    'Gender': 'Gender', 'Married': 'Marital Status', 'Dependents': 'Dependents',
    'Education': 'Education', 'Self_Employed': 'Employment',
    'ApplicantIncome': 'Applicant Income', 'CoapplicantIncome': 'Coapplicant Income',
    'LoanAmount': 'Loan Amount', 'Loan_Amount_Term': 'Loan Term',
    'Credit_History': 'Credit History', 'Property_Area': 'Property Area'
}

def initialize_models():
    """Load models on startup"""
    global scaler, MODELS
    try:
        scaler_path = BASE_DIR / 'scaler.pkl'
        model_path = BASE_DIR / 'xgboost_model.pkl'
        
        if scaler_path.exists():
            scaler = joblib.load(str(scaler_path))
        if model_path.exists():
            MODELS['XGBoost'] = joblib.load(str(model_path))
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def load_history():
    """Load prediction history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    """Save prediction history to JSON file"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except:
        pass

def get_kpi_stats():
    """Calculate KPI statistics from history"""
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
    """Calculate feature importance scores"""
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
    """Generate detailed insights from prediction"""
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

# Routes

@app.route('/api/health')
def health():
    """Health check endpoint - should always work"""
    models_loaded = len(MODELS) > 0 and scaler is not None
    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'models_loaded': models_loaded,
        'template_folder_exists': template_folder.exists(),
        'static_folder_exists': static_folder.exists()
    })

@app.route('/')
def index():
    """Root endpoint - returns health info if templates not available"""
    try:
        if template_folder.exists():
            stats = get_kpi_stats()
            history = load_history()[-5:]
            return render_template('dashboard.html', stats=stats, recent=history)
        else:
            return jsonify({
                'message': 'Smart Lender API is running',
                'endpoints': {
                    '/api/health': 'Health check',
                    '/api/stats': 'Get statistics',
                    '/api/history': 'Get prediction history'
                }
            })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """API endpoint for statistics"""
    try:
        return jsonify(get_kpi_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def api_history():
    """API endpoint for prediction history"""
    try:
        return jsonify(load_history())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<record_id>', methods=['POST'])
def delete_record(record_id):
    """Delete a prediction record"""
    try:
        history = load_history()
        history = [h for h in history if h.get('id') != record_id]
        save_history(history)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-page')
def predict_page():
    """Prediction form page"""
    try:
        if template_folder.exists():
            history = load_history()
            last_model = history[-1].get('model_used', DEFAULT_MODEL) if history else DEFAULT_MODEL
            model_info = MODEL_INFO.get(last_model, MODEL_INFO[DEFAULT_MODEL])
            return render_template('predict.html', model_info=model_info)
        return jsonify({'message': 'Templates not deployed'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Process prediction request"""
    try:
        if not MODELS or scaler is None:
            return jsonify({'error': 'Models not loaded'}), 503
            
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

        if template_folder.exists():
            return render_template('result.html', result=result, probability=f'{prob:.1%}',
                                 record=record, importance=importance, insights=insights, model_info=model_info)
        else:
            return jsonify({'result': result, 'probability': f'{prob:.1%}', 'record': record})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# Initialize models when app starts
initialize_models()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
