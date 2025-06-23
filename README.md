# Time Series Analysis with Cryptocurrency

A data-driven forecasting system for cryptocurrency trends using statistical and machine learning models. The project analyzes historical data, predicts future prices, and includes market sentiment analysis, visualizations, and an interactive dashboard.

## Features

- Time Series Forecasting using ARIMA, LSTM, Prophet
- Market Sentiment Analysis using NLP on Twitter/News Data
- Interactive Dashboard for trends, risks, and insights
- Dockerized and deployable using Kubernetes
- Cloud-ready (AWS/GCP with CI/CD via GitHub Actions)

## Tech Stack

- **Frontend**: React, Recharts, TailwindCSS
- **Backend**: Flask / FastAPI
- **ML Models**: Scikit-learn, TensorFlow/Keras, Prophet
- **Data**: Crypto APIs (CoinGecko, Binance, etc.)
- **Infra**: Docker, Kubernetes, GitHub Actions
- **Cloud**: AWS (EC2, S3, Lambda, SageMaker) or Google Cloud AI Platform

## Project Structure
time-series-crypto-forecasting/
│
├── data/ # Raw and processed datasets
├── models/ # Saved ML models
├── notebooks/ # Jupyter notebooks for EDA and training
├── scripts/ # Data processing and training scripts
├── dashboard/ # Frontend (React)
├── api/ # Backend API (Flask or FastAPI)
├── Dockerfile # Docker configuration
├── docker-compose.yml # Docker services
└── README.md


## Installation

```bash
git clone https://github.com/vinayreddy9854/time-series-crypto-forecasting.git
cd time-series-crypto-forecasting
pip install -r requirements.txt

## Run Locally


python scripts/train_model.py
uvicorn api.main:app --reload
cd dashboard && npm install && npm start
