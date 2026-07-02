# Smart Lender - Loan Approval Prediction

An AI-powered loan approval prediction system built with Flask and XGBoost, deployed on Vercel.

## Features

- **ML-Powered Predictions**: Uses XGBoost classifier for accurate loan approval predictions
- **Feature Importance**: Displays which factors most influence the decision
- **Risk Analysis**: Calculates risk scores and default probabilities
- **EMI Calculation**: Estimates monthly payment amounts
- **Prediction History**: Tracks all predictions with detailed analytics
- **Dashboard**: Real-time statistics and insights

## Project Structure

```
smart/
├── api/
│   ├── app.py              # Flask application
│   └── requirements.txt     # Python dependencies
├── templates/              # HTML templates
├── static/                 # CSS and JavaScript files
├── *.pkl                   # Trained models (XGBoost, scaler)
├── vercel.json            # Vercel deployment config
└── README.md
```

## Prerequisites

- Python 3.8+
- Flask 3.1.0
- XGBoost 2.1.3
- Pandas
- NumPy
- scikit-learn

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r api/requirements.txt
```

## Local Development

```bash
cd api
python app.py
```

The application will be available at `http://localhost:5000`

## Deployment on Vercel

### Prerequisites
- Vercel account
- Vercel CLI installed

### Deploy

```bash
vercel
```

The deployment will:
1. Detect Python runtime automatically
2. Install dependencies from `api/requirements.txt`
3. Run Flask app through Vercel's Python runtime
4. Serve all routes through the `/api/app.py` handler

### Important Notes

- All files listed in `.vercelignore` are excluded from deployment
- Model files (`.pkl`) and scaler are included for inference
- JSON model files are kept for compatibility
- Large CSV files are excluded to reduce deployment size

## Routes

- `GET /` - Dashboard with KPI statistics
- `GET /predict-page` - Prediction form
- `POST /predict` - Process loan prediction
- `GET /analytics` - Analytics page
- `GET /history` - Prediction history
- `GET /insights` - Latest prediction insights
- `GET /api/history` - JSON API for history
- `GET /api/stats` - JSON API for statistics
- `POST /api/delete/<record_id>` - Delete a prediction record

## Model Information

- **Model**: XGBoost Classifier
- **Accuracy**: 86.2%
- **Input Features**: 11 (Gender, Marital Status, Education, Income, Loan Amount, etc.)
- **Output**: Loan Approved/Rejected with confidence score

## License

MIT License
