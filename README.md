# SAGE-Twin - Digital Twin Business Simulation

A multi-agent Digital Twin POC for business simulation and strategic decision-making.

![SAGE-Twin](https://img.shields.io/badge/SAGE--Twin-Digital%20Twin-8b5cf6)
![Python](https://img.shields.io/badge/Python-3.11+-3776ab)
![React](https://img.shields.io/badge/React-18+-61dafb)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)

## 🏢 What is SAGE-Twin?

SAGE-Twin is a **Digital Twin POC** that models your business as an interconnected system. Unlike simple dashboards, it:

- **Models interdependencies** between business metrics
- **Simulates forward-state changes** using a multi-agent causal engine
- **Propagates effects** through a knowledge graph
- **Provides AI-powered recommendations** for strategic decisions

> "This system models interdependencies and simulates forward-state changes using a multi-agent causal engine."

## ✨ Features

### 🤖 Multi-Agent System (6 Specialized Agents)
| Agent | Role |
|-------|------|
| Revenue Agent | Adjusts revenue based on customers, pricing, marketing |
| Customer Agent | Manages customer count, churn, retention |
| Sentiment Agent | Tracks sentiment score and brand health |
| Operations Agent | Manages delivery delay and efficiency |
| Risk Agent | Calculates financial, operational, and overall risk |
| Strategy Agent | Executive brain - recommendations, warnings, tradeoffs |

### 🔗 Causal Relationship Graph (35+ Relationships)
```
Marketing ↑ → Customers ↑ → Revenue ↑
Delivery Delay ↑ → Sentiment ↓ → Churn ↑ → Revenue ↓
Costs ↑ → Profit ↓ → Risk ↑
```

### 📂 Multi-Source Data Ingestion
Upload different file types for different business areas:

| Category | Formats | Description |
|----------|---------|-------------|
| 💰 Revenue | CSV | Financial metrics |
| 👥 Customers | CSV | Customer data |
| 💬 Reviews | CSV, DOCX, TXT | Feedback & sentiment |
| 📢 Marketing | CSV | Campaign performance |
| 🚚 Operations | CSV | Delivery & logistics |

### 📊 Business Health Index
A unified 0-100 score combining:
- Revenue growth
- Customer sentiment
- Risk score
- Stability metrics

### 🔮 3-Month Forecast
- Trend projections
- Risk trajectory
- Executive outlook summary

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm or yarn

### Backend Setup

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

# Create .env file with your OpenAI API key (optional, for AI insights)
echo "OPENAI_API_KEY=your_key_here" > .env

# Run server
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### Access the Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📁 Project Structure

```
sage-twin-poc/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── state_engine.py      # Business state management
│   ├── causal_graph.py      # Causal relationship map
│   ├── multi_agents.py      # 6-agent system
│   ├── forecast.py          # 3-month projections
│   ├── data_sources.py      # Multi-source ingestion
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── App.css          # Styles
│   │   └── index.css        # Global styles
│   ├── package.json
│   └── vite.config.js
│
└── sample_data/
    ├── revenue_data.csv
    ├── customer_data.csv
    ├── marketing_data.csv
    ├── operations_data.csv
    └── customer_reviews.txt
```

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload/{category}` | POST | Upload file to category |
| `/simulate` | POST | Run multi-agent simulation |
| `/causal-graph` | GET | Get relationship map |
| `/health` | GET | Business health index |
| `/forecast` | GET | 3-month projections |
| `/agents` | GET | List all agents |
| `/sources` | GET | Data sources status |

## 🎮 Usage

1. **Upload Data**: Click "Data Sources" and upload CSV files to different categories
2. **Adjust Parameters**: Use sliders to simulate changes (price, marketing, costs, delay)
3. **Run Simulation**: Click "Run Simulation" to see cascading effects
4. **Review Results**: See health score, agent activity, recommendations, and forecast

## 🧪 Sample Data

Use the files in `sample_data/` folder to test:
- `revenue_data.csv` - Financial metrics
- `customer_data.csv` - Customer metrics  
- `marketing_data.csv` - Campaign data
- `operations_data.csv` - Delivery & logistics
- `customer_reviews.txt` - Feedback for sentiment

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for AI insights | Optional |

## 📄 License

MIT License - feel free to use for your own projects!

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

Built with ❤️ using FastAPI, React, and Recharts
