# SAGE-Twin - AI-Powered Digital Twin for Business Intelligence

A multi-agent Digital Twin platform for business simulation, predictive analytics, and strategic decision-making with **multi-user session isolation**.

![SAGE-Twin](https://img.shields.io/badge/SAGE--Twin-Digital%20Twin-8b5cf6)
![Python](https://img.shields.io/badge/Python-3.11+-3776ab)
![React](https://img.shields.io/badge/React-18+-61dafb)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ed)

## 🏢 What is SAGE-Twin?

SAGE-Twin is an **AI-powered Digital Twin platform** that models your business as an interconnected system. Unlike simple dashboards, it:

- **Models interdependencies** between business metrics using causal graphs
- **Simulates forward-state changes** using a multi-agent causal engine
- **Predicts future outcomes** with ML-powered forecasting and churn prediction
- **Monitors data drift** to ensure model accuracy over time
- **Provides AI-powered recommendations** for strategic decisions
- **Supports multiple users** with isolated session states

> "This system models interdependencies and simulates forward-state changes using a multi-agent causal engine with machine learning capabilities."

---

## ✨ Key Features

### 🤖 Multi-Agent System (6 Specialized AI Agents)

| Agent | Role | Capabilities |
|-------|------|--------------|
| **Revenue Agent** | Financial analysis | Adjusts revenue based on customers, pricing, marketing |
| **Customer Agent** | Customer lifecycle | Manages customer count, churn, retention predictions |
| **Sentiment Agent** | Brand health | Tracks sentiment score and brand perception |
| **Operations Agent** | Operational efficiency | Manages delivery delay and efficiency metrics |
| **Risk Agent** | Risk assessment | Calculates financial, operational, and overall risk scores |
| **Strategy Agent** | Executive brain | Generates recommendations, warnings, and tradeoff analysis |

### 🧠 Machine Learning Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Horizon Forecasting** | Revenue predictions for 7, 30, 90, and 180 days |
| **Churn Prediction** | Identify at-risk customers with risk scoring |
| **Drift Monitoring** | Detect data drift and model performance degradation |
| **Feature Engineering** | Automated feature extraction from business data |

### 🔗 Causal Relationship Graph (35+ Relationships)

```
Marketing ↑ → Customers ↑ → Revenue ↑
Delivery Delay ↑ → Sentiment ↓ → Churn ↑ → Revenue ↓
Costs ↑ → Profit ↓ → Risk ↑
Price ↑ → Revenue ↑ (short-term) → Churn ↑ (long-term)
```

### 👥 Multi-User Session Management

- **Isolated user sessions** - Each user gets their own Digital Twin state
- **Persistent session data** - Data survives page refreshes
- **Concurrent users** - Multiple users can run simulations simultaneously
- **Session-aware ML models** - Each session has its own trained models

### 📂 Multi-Source Data Ingestion

| Category | Formats | Data Types |
|----------|---------|------------|
| 💰 Revenue & Financials | CSV | Revenue, costs, profit, pricing |
| 👥 Customer Data | CSV | Customer counts, segments, demographics |
| 💬 Customer Reviews | CSV, DOCX, TXT | Feedback, reviews, sentiment |
| 📢 Marketing Campaigns | CSV | Ad spend, conversions, ROI |
| 🚚 Operations & Delivery | CSV | Delivery times, fulfillment, logistics |
| 📊 General Data | CSV | Any other business metrics |

### 📊 Business Health Dashboard

- **Real-time KPI cards** - Revenue, Customers, Sentiment, Risk Score
- **Health Index (0-100)** - Unified business health score
- **Interactive What-If Simulation** - Adjust parameters and see cascading effects
- **3-Month Forecast** with trend projections

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/sage-twin-poc.git
cd sage-twin-poc

# Start with Docker Compose
docker-compose up --build -d

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup

#### Prerequisites
- Python 3.11+
- Node.js 18+
- npm or yarn

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (optional)
echo "OPENAI_API_KEY=your_key_here" > .env

# Run server
uvicorn main:app --reload --port 8000
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### Access Points
| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 (Docker) or http://localhost:5173 (Dev) |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |

---

## 📁 Project Structure

```
sage-twin-poc/
├── backend/
│   ├── main.py                 # FastAPI application & endpoints
│   ├── state_engine.py         # Business state management
│   ├── session_manager.py      # Multi-user session handling
│   ├── causal_graph.py         # Causal relationship map
│   ├── multi_agents.py         # 6-agent simulation system
│   ├── forecast.py             # 3-month projections
│   ├── data_sources.py         # Multi-source data ingestion
│   ├── ml_api.py               # ML API endpoints
│   ├── ml/
│   │   ├── models/
│   │   │   ├── revenue_forecaster.py  # Multi-horizon forecasting
│   │   │   └── churn_predictor.py     # Churn prediction model
│   │   └── feature_engineering.py     # Feature extraction
│   ├── monitoring/
│   │   └── drift_detector.py   # Data drift monitoring
│   ├── schemas/                # Pydantic schemas
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main React component
│   │   ├── App.css             # Premium dark theme styles
│   │   └── main.jsx            # Entry point
│   ├── Dockerfile
│   └── package.json
│
├── docker-compose.yml          # Docker orchestration
└── sample_data/                # Test data files
```

---

## 🛠️ API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/session/create` | POST | Create new user session |
| `/upload/{category}` | POST | Upload file to category |
| `/simulate` | POST | Run multi-agent simulation |
| `/state` | GET | Get current Digital Twin state |
| `/reset` | POST | Reset to baseline state |
| `/sources` | GET | Get data sources status |
| `/health` | GET | Business health index |
| `/causal-graph` | GET | Get relationship map |

### ML Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ml/forecast/multi-horizon` | GET | Multi-horizon revenue forecast |
| `/ml/predict/churn` | GET | Churn prediction for customers |
| `/ml/monitoring/drift` | GET | Data drift analysis |

All endpoints support `?session_id=xxx` query parameter for session isolation.

---

## 🎮 Usage Guide

### 1. Start a Session
When you open the application, a unique session is automatically created for you.

### 2. Upload Your Data
Click on the category cards in the Data Sources section:
- Upload CSV files to appropriate categories
- Data is parsed and aggregated automatically
- Business state initializes with your data

### 3. Run What-If Simulations
Use the sliders to adjust parameters:
- **Price Change** (-20% to +30%)
- **Marketing Spend** (-50% to +100%)
- **Cost Change** (-30% to +30%)
- **Delivery Delay** (-5 to +10 days)
- **Market Shock** (toggle for crisis simulation)

### 4. Review Simulation Results
After running a simulation, you'll see:
- **Impact Analysis** - Before/after comparisons with % changes
- **Agent Activity** - What each AI agent detected and decided
- **Strategic Priority** - GROWTH, RISK MITIGATION, RETENTION, etc.
- **Recommendations** - AI-generated action items
- **Warnings** - Risk alerts from agents
- **Tradeoffs** - Business tradeoff analysis
- **3-Month Forecast** - Revenue projections with outlook

### 5. Explore ML Insights
Navigate to the Forecasts, Churn, and Monitoring tabs for:
- Multi-horizon revenue predictions
- Customer churn risk analysis
- Data drift monitoring

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for AI insights | Optional |
| `PYTHONUNBUFFERED` | Python output buffering | Auto-set in Docker |

### Docker Configuration

The `docker-compose.yml` configures:
- **Backend**: Python 3.11 + FastAPI on port 8000
- **Frontend**: Node.js + Vite + Nginx on port 3000
- Automatic container restart on failure
- Volume mounting for development

---

## 🧪 Sample Data

Test files are provided in `sample_data/`:

| File | Description |
|------|-------------|
| `revenue_data.csv` | Financial metrics (revenue, costs, profit) |
| `customer_data.csv` | Customer counts and segments |
| `marketing_data.csv` | Campaign performance data |
| `operations_data.csv` | Delivery and logistics metrics |
| `customer_reviews.txt` | Text feedback for sentiment analysis |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                          │
│   (Dashboard, What-If, Forecasts, Churn, Monitoring)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
├─────────────────────────────────────────────────────────────┤
│  Session Manager  │  State Engine  │  Data Sources          │
├───────────────────┼────────────────┼────────────────────────┤
│  Multi-Agent      │  Causal Graph  │  ML Models             │
│  Engine           │  (35+ rules)   │  (Forecast, Churn)     │
├───────────────────┼────────────────┼────────────────────────┤
│  Revenue Agent    │  Risk Agent    │  Drift Detector        │
│  Customer Agent   │  Strategy Agent│                        │
│  Sentiment Agent  │  Operations    │                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📄 License

MIT License - feel free to use for your own projects!

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

Built with:
- **FastAPI** - Modern Python web framework
- **React** - UI library
- **Recharts** - Charting library
- **Vite** - Frontend build tool
- **Docker** - Containerization

---

<p align="center">
Built with ❤️ for intelligent business decision-making
</p>
