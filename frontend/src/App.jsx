import { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  AreaChart,
  Area,
  Line,
  LineChart,
  ComposedChart,
  Legend,
  PieChart,
  Pie,
} from "recharts";
import "./App.css";

// Use environment variable for API URL, fallback to localhost for development
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const COLORS = {
  primary: "#8b5cf6",
  secondary: "#3b82f6",
  success: "#10b981",
  warning: "#f59e0b",
  danger: "#ef4444",
  cyan: "#06b6d4",
  pink: "#ec4899",
};

const CATEGORY_ICONS = {
  revenue: "💰",
  customers: "👥",
  reviews: "💬",
  marketing: "📢",
  operations: "🚚",
  general: "📊",
};

function App() {
  // State
  const [initialized, setInitialized] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("dashboard");

  // Data Sources
  const [categories, setCategories] = useState({});
  const [dataSources, setDataSources] = useState({});
  const [uploadingCategory, setUploadingCategory] = useState(null);

  // Twin State
  const [twinState, setTwinState] = useState(null);
  const [healthScore, setHealthScore] = useState(0);
  const [confidenceScore, setConfidenceScore] = useState(0);

  // Simulation
  const [adjustments, setAdjustments] = useState({
    price: 0,
    marketing_spend: 0,
    costs: 0,
    delivery_delay: 0,
    market_shock: false,
  });
  const [simulationResult, setSimulationResult] = useState(null);

  // ML State (NEW)
  const [mlEnabled, setMlEnabled] = useState(false);
  const [forecastData, setForecastData] = useState(null);
  const [churnData, setChurnData] = useState(null);
  const [driftData, setDriftData] = useState(null);
  const [multiHorizonForecast, setMultiHorizonForecast] = useState(null);

  // Session State (Multi-User Support)
  const [sessionId, setSessionId] = useState(null);

  // UI State
  const [showUploadPanel, setShowUploadPanel] = useState(true);

  const fileInputRefs = useRef({});

  // Initialize session on mount
  useEffect(() => {
    initializeSession();
  }, []);

  const initializeSession = async () => {
    // Check if session exists in localStorage
    let existingSession = localStorage.getItem("sage_twin_session");

    if (existingSession) {
      // Verify session is still valid
      try {
        const res = await axios.get(`${API_URL}/session/${existingSession}`);
        if (!res.data.error) {
          setSessionId(existingSession);
          loadDataSources(existingSession);
          checkMLStatus();
          return;
        }
      } catch (e) {
        // Session expired or invalid, create new one
      }
    }

    // Create new session
    try {
      const res = await axios.post(`${API_URL}/session/create`);
      const newSessionId = res.data.session_id;
      localStorage.setItem("sage_twin_session", newSessionId);
      setSessionId(newSessionId);
      checkMLStatus();
    } catch (error) {
      console.error("Failed to create session:", error);
      // Fallback to global state (no session)
      loadDataSources(null);
      checkMLStatus();
    }
  };

  // Load categories when session is ready
  useEffect(() => {
    if (sessionId !== null) {
      axios.get(`${API_URL}/data-categories`).then(res => {
        setCategories(res.data.categories || {});
      });
      loadDataSources(sessionId);
    }
  }, [sessionId]);

  const checkMLStatus = async () => {
    try {
      const res = await axios.get(`${API_URL}/`);
      setMlEnabled(res.data.ml_enabled || false);
    } catch (error) {
      console.error("Failed to check ML status:", error);
    }
  };

  const loadDataSources = async (sid = sessionId) => {
    try {
      const params = sid ? `?session_id=${sid}` : "";
      const res = await axios.get(`${API_URL}/sources${params}`);
      setDataSources(res.data);

      // Check if initialized
      if (res.data.sources?.total_files > 0) {
        setInitialized(true);
        // Load state
        const stateRes = await axios.get(`${API_URL}/state${params}`);
        if (!stateRes.data.error) {
          setTwinState(stateRes.data);
          setHealthScore(stateRes.data.health_score || 0);
        }
        // Load ML data if enabled
        loadMLData();
      } else {
        setInitialized(false);
      }
    } catch (error) {
      console.error("Failed to load sources:", error);
    }
  };

  const loadMLData = async () => {
    if (!mlEnabled) return;

    try {
      // Load multi-horizon forecast - may fail if no data
      const forecastRes = await axios.get(`${API_URL}/ml/forecast/multi-horizon`);
      setMultiHorizonForecast(forecastRes.data);
    } catch (error) {
      console.log("Forecast data not available yet");
      setMultiHorizonForecast(null);
    }

    try {
      // Load churn prediction
      const churnRes = await axios.post(`${API_URL}/ml/predict/churn`, { revenue_per_customer: 1000 });
      setChurnData(churnRes.data);
    } catch (error) {
      console.log("Churn data not available yet");
      setChurnData(null);
    }

    try {
      // Load drift monitoring
      const driftRes = await axios.get(`${API_URL}/ml/monitoring/drift`);
      setDriftData(driftRes.data);
    } catch (error) {
      console.log("Drift data not available yet");
      setDriftData(null);
    }
  };

  // File Upload Handler
  const handleFileUpload = async (event, category) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setUploadingCategory(category);
    setLoading(true);

    try {
      const params = sessionId ? `?session_id=${sessionId}` : "";
      await axios.post(`${API_URL}/upload/${category}${params}`, formData);
      await loadDataSources();
    } catch (error) {
      console.error("Upload failed:", error);
      alert(`Failed to upload file: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
      setUploadingCategory(null);
      if (fileInputRefs.current[category]) {
        fileInputRefs.current[category].value = "";
      }
    }
  };

  // Clear category
  const clearCategory = async (category) => {
    if (!confirm(`Clear all ${category} data?`)) return;

    setLoading(true);
    try {
      const params = sessionId ? `?session_id=${sessionId}` : "";
      await axios.delete(`${API_URL}/sources/${category}${params}`);
      await loadDataSources();
      setSimulationResult(null);
    } catch (error) {
      console.error("Failed to clear:", error);
    } finally {
      setLoading(false);
    }
  };

  // Run Simulation
  const runSimulation = async () => {
    setLoading(true);
    try {
      const params = sessionId ? `?session_id=${sessionId}` : "";
      const res = await axios.post(`${API_URL}/simulate${params}`, adjustments);
      setSimulationResult(res.data);
      if (res.data.projected_state) {
        setTwinState(prev => ({
          ...prev,
          current_state: res.data.projected_state,
        }));
        setHealthScore(res.data.new_health_score || healthScore);
        setConfidenceScore(res.data.confidence_score || 0);
      }
      // Reload ML predictions after simulation
      loadMLData();
    } catch (error) {
      console.error("Simulation failed:", error);
    } finally {
      setLoading(false);
    }
  };

  // Reset State
  const resetState = async () => {
    setLoading(true);
    try {
      const params = sessionId ? `?session_id=${sessionId}` : "";
      await axios.post(`${API_URL}/reset${params}`);
      const stateRes = await axios.get(`${API_URL}/state${params}`);
      if (!stateRes.data.error) {
        setTwinState(stateRes.data);
        setHealthScore(stateRes.data.health_score || 0);
      }
      setSimulationResult(null);
      setAdjustments({
        price: 0,
        marketing_spend: 0,
        costs: 0,
        delivery_delay: 0,
        market_shock: false,
      });
      loadMLData();
    } catch (error) {
      console.error("Reset failed:", error);
    } finally {
      setLoading(false);
    }
  };

  // Format helpers
  const formatValue = (value, metric) => {
    if (value === undefined || value === null) return "N/A";
    if (metric === "revenue" || metric === "profit" || metric === "costs" || metric === "marketing_spend") {
      return `$${Number(value).toLocaleString()}`;
    }
    if (metric === "margin" || metric === "growth" || metric === "sentiment" || metric === "risk_score") {
      return Number(value).toFixed(1);
    }
    return Number(value).toLocaleString();
  };

  const getHealthColor = (score) => {
    if (score >= 70) return COLORS.success;
    if (score >= 50) return COLORS.warning;
    return COLORS.danger;
  };

  const getDeltaClass = (delta) => {
    if (!delta) return "";
    return delta > 0 ? "delta-positive" : delta < 0 ? "delta-negative" : "";
  };

  const getRiskColor = (risk) => {
    if (risk === "low") return COLORS.success;
    if (risk === "medium") return COLORS.warning;
    if (risk === "high") return COLORS.danger;
    return COLORS.danger;
  };

  const getCategoryStatus = (category) => {
    const sources = dataSources.sources?.by_category || {};
    return sources[category]?.files?.length > 0;
  };

  const getCategoryFiles = (category) => {
    const sources = dataSources.sources?.by_category || {};
    return sources[category]?.files || [];
  };

  // Tab content rendering
  const renderDashboardTab = () => (
    <>
      {/* KPI Cards Row */}
      <div className="kpi-row">
        <div className="kpi-card">
          <div className="kpi-icon">💰</div>
          <div className="kpi-content">
            <div className="kpi-label">Revenue</div>
            <div className="kpi-value">{formatValue(twinState?.current_state?.revenue, "revenue")}</div>
            {simulationResult?.impact_analysis?.revenue && (
              <div className={`kpi-delta ${getDeltaClass(simulationResult.impact_analysis.revenue.delta)}`}>
                {simulationResult.impact_analysis.revenue.delta > 0 ? "+" : ""}
                {simulationResult.impact_analysis.revenue.delta?.toFixed(1)}%
              </div>
            )}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon">👥</div>
          <div className="kpi-content">
            <div className="kpi-label">Customers</div>
            <div className="kpi-value">{formatValue(twinState?.current_state?.customers, "customers")}</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon">💬</div>
          <div className="kpi-content">
            <div className="kpi-label">Sentiment</div>
            <div className="kpi-value">{formatValue(twinState?.current_state?.sentiment, "sentiment")}</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon">⚠️</div>
          <div className="kpi-content">
            <div className="kpi-label">Risk Score</div>
            <div className="kpi-value">{formatValue(twinState?.current_state?.risk_score, "risk_score")}</div>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="dashboard-grid">
        {/* Simulation Panel */}
        <div className="panel simulation-panel">
          <h3>🎮 What-If Simulation</h3>
          <div className="slider-group">
            <label>
              Marketing Spend
              <span className={adjustments.marketing_spend >= 0 ? "value-positive" : "value-negative"}>
                {adjustments.marketing_spend >= 0 ? "+" : ""}{adjustments.marketing_spend}%
              </span>
            </label>
            <input
              type="range"
              min="-50"
              max="50"
              value={adjustments.marketing_spend}
              onChange={(e) => setAdjustments({ ...adjustments, marketing_spend: parseInt(e.target.value) })}
            />
          </div>

          <div className="slider-group">
            <label>
              Cost Change
              <span className={adjustments.costs <= 0 ? "value-positive" : "value-negative"}>
                {adjustments.costs >= 0 ? "+" : ""}{adjustments.costs}%
              </span>
            </label>
            <input
              type="range"
              min="-50"
              max="50"
              value={adjustments.costs}
              onChange={(e) => setAdjustments({ ...adjustments, costs: parseInt(e.target.value) })}
            />
          </div>

          <div className="slider-group">
            <label>
              Delivery Delay
              <span className={adjustments.delivery_delay <= 0 ? "value-positive" : "value-negative"}>
                {adjustments.delivery_delay >= 0 ? "+" : ""}{adjustments.delivery_delay} days
              </span>
            </label>
            <input
              type="range"
              min="-5"
              max="10"
              value={adjustments.delivery_delay}
              onChange={(e) => setAdjustments({ ...adjustments, delivery_delay: parseInt(e.target.value) })}
            />
          </div>

          <div className="checkbox-group">
            <label>
              <input
                type="checkbox"
                checked={adjustments.market_shock}
                onChange={(e) => setAdjustments({ ...adjustments, market_shock: e.target.checked })}
              />
              Market Shock 💥
            </label>
          </div>

          <button className="btn-primary" onClick={runSimulation} disabled={loading}>
            {loading ? "Simulating..." : "▶ Run Simulation"}
          </button>
        </div>

        {/* Impact Analysis */}
        {simulationResult && (
          <div className="panel impact-panel">
            <h3>📊 Impact Analysis</h3>
            <div className="impact-grid">
              {Object.entries(simulationResult.impact_analysis || {}).map(([metric, data]) => (
                <div key={metric} className="impact-item">
                  <div className="impact-metric">{metric.toUpperCase()}</div>
                  <div className="impact-values">
                    <span className="impact-before">{formatValue(data.before, metric)}</span>
                    <span className="impact-arrow">→</span>
                    <span className="impact-after">{formatValue(data.after, metric)}</span>
                  </div>
                  <div className={`impact-delta ${getDeltaClass(data.delta)}`}>
                    {data.delta > 0 ? "+" : ""}{data.delta?.toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Agent Activity */}
        <div className="panel agents-panel">
          <h3>🤖 Agent Activity</h3>
          <div className="agent-list">
            {simulationResult?.agent_outputs?.map((agent, idx) => (
              <div key={idx} className="agent-card">
                <div className="agent-header">
                  <span className="agent-name">{agent.agent}</span>
                  <span className="agent-confidence" style={{ color: getHealthColor(agent.confidence * 100) }}>
                    {(agent.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="agent-message">{agent.analysis || agent.recommendation}</p>
              </div>
            )) || (
                <p className="no-data">Run a simulation to see agent recommendations</p>
              )}
          </div>
        </div>
      </div>
    </>
  );

  const renderForecastTab = () => (
    <div className="forecast-container">
      <div className="panel forecast-panel">
        <h3>📈 Revenue Forecast (Multi-Horizon)</h3>
        {multiHorizonForecast ? (
          <>
            <div className="forecast-horizons">
              {Object.entries(multiHorizonForecast.horizons || {}).map(([days, data]) => (
                <div key={days} className="horizon-card">
                  <div className="horizon-days">{days} Days</div>
                  <div className="horizon-value">${Number(data.predicted_revenue).toLocaleString()}</div>
                  <div className="horizon-range">
                    ${Number(data.confidence_interval[0]).toLocaleString()} - ${Number(data.confidence_interval[1]).toLocaleString()}
                  </div>
                  <div className={`horizon-trend trend-${data.trend}`}>
                    {data.trend === "up" ? "📈" : data.trend === "down" ? "📉" : "➡️"} {data.trend}
                  </div>
                </div>
              ))}
            </div>
            <div className="forecast-chart">
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={Object.entries(multiHorizonForecast.horizons || {}).map(([days, data]) => ({
                  days: `${days}d`,
                  revenue: data.predicted_revenue,
                  low: data.confidence_interval[0],
                  high: data.confidence_interval[1],
                }))}>
                  <XAxis dataKey="days" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                  <Tooltip formatter={(v) => `$${Number(v).toLocaleString()}`} />
                  <Area type="monotone" dataKey="low" fill={COLORS.primary} fillOpacity={0.1} stroke="none" />
                  <Area type="monotone" dataKey="high" fill={COLORS.primary} fillOpacity={0.1} stroke="none" />
                  <Line type="monotone" dataKey="revenue" stroke={COLORS.primary} strokeWidth={3} dot={{ fill: COLORS.primary }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </>
        ) : (
          <p className="no-data">Upload data to see forecasts</p>
        )}
      </div>
    </div>
  );

  const renderChurnTab = () => (
    <div className="churn-container">
      <div className="panel churn-panel">
        <h3>🚨 Churn Risk Analysis</h3>
        {churnData ? (
          <>
            <div className="churn-summary">
              <div className="churn-gauge">
                <div
                  className="gauge-fill"
                  style={{
                    width: `${churnData.churn_prediction.probability * 100}%`,
                    backgroundColor: getRiskColor(churnData.churn_prediction.risk_level)
                  }}
                />
                <div className="gauge-label">
                  {(churnData.churn_prediction.probability * 100).toFixed(1)}% Churn Risk
                </div>
              </div>
              <div className={`risk-badge risk-${churnData.churn_prediction.risk_level}`}>
                {churnData.churn_prediction.risk_level.toUpperCase()}
              </div>
            </div>

            <div className="churn-factors">
              <h4>Contributing Factors</h4>
              {Object.entries(churnData.contributing_factors || {}).map(([factor, value]) => (
                <div key={factor} className="factor-item">
                  <span className="factor-name">{factor.replace(/_/g, " ")}</span>
                  <div className="factor-bar">
                    <div
                      className="factor-fill"
                      style={{
                        width: `${Math.abs(value) * 100}%`,
                        backgroundColor: value > 0 ? COLORS.danger : COLORS.success
                      }}
                    />
                  </div>
                  <span className="factor-value">{(value * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>

            <div className="churn-recommendations">
              <h4>💡 Retention Recommendations</h4>
              <ul>
                {churnData.recommendations?.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>

            <div className="revenue-at-risk">
              <span className="rar-label">Revenue at Risk:</span>
              <span className="rar-value">${Number(churnData.revenue_at_risk).toLocaleString()}</span>
            </div>
          </>
        ) : (
          <p className="no-data">Upload data to see churn predictions</p>
        )}
      </div>
    </div>
  );

  const renderMonitoringTab = () => (
    <div className="monitoring-container">
      <div className="panel monitoring-panel">
        <h3>🔍 Model & Data Monitoring</h3>
        {driftData ? (
          <>
            <div className="monitoring-summary">
              <div className="stability-gauge">
                <div className="gauge-circle" style={{ borderColor: getHealthColor((1 - driftData.drift_report.overall_score) * 100) }}>
                  <span className="gauge-value">{((1 - driftData.drift_report.overall_score) * 100).toFixed(0)}%</span>
                  <span className="gauge-label">Stable</span>
                </div>
              </div>
              <div className="monitoring-stats">
                <div className="stat-item">
                  <span className="stat-label">Severity</span>
                  <span className={`stat-value severity-${driftData.drift_report.severity}`}>
                    {driftData.drift_report.severity.toUpperCase()}
                  </span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Performance Trend</span>
                  <span className="stat-value">{driftData.drift_report.performance_trend}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Retraining Needed</span>
                  <span className={`stat-value ${driftData.drift_report.requires_retraining ? "alert" : ""}`}>
                    {driftData.drift_report.requires_retraining ? "⚠️ YES" : "✅ NO"}
                  </span>
                </div>
              </div>
            </div>

            {driftData.alerts?.length > 0 && (
              <div className="drift-alerts">
                <h4>⚠️ Active Alerts</h4>
                {driftData.alerts.map((alert, idx) => (
                  <div key={idx} className={`alert-item alert-${alert.severity}`}>
                    <span className="alert-type">{alert.type}</span>
                    <span className="alert-message">{alert.message}</span>
                    <span className="alert-action">{alert.action}</span>
                  </div>
                ))}
              </div>
            )}

            <div className="monitoring-recommendations">
              <h4>📋 Recommendations</h4>
              <ul>
                {driftData.recommendations?.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>
          </>
        ) : (
          <p className="no-data">Upload data to see monitoring status</p>
        )}
      </div>
    </div>
  );

  const renderUploadPanel = () => (
    <div className={`upload-panel ${showUploadPanel ? "" : "collapsed"}`}>
      <div className="upload-header" onClick={() => setShowUploadPanel(!showUploadPanel)}>
        <h3>📁 Data Sources</h3>
        <span className="toggle-icon">{showUploadPanel ? "▼" : "▶"}</span>
      </div>
      {showUploadPanel && (
        <div className="category-grid">
          {Object.entries(categories).map(([key, cat]) => (
            <div key={key} className={`category-card ${getCategoryStatus(key) ? "has-data" : ""}`}>
              <div className="category-icon">{CATEGORY_ICONS[key] || "📊"}</div>
              <div className="category-name">{cat.name}</div>
              <div className="category-desc">{cat.description}</div>

              {/* Show uploaded files */}
              {getCategoryStatus(key) && (
                <div className="uploaded-files">
                  {getCategoryFiles(key).map((file, idx) => (
                    <div key={idx} className="uploaded-file">
                      <span className="file-icon">📄</span>
                      <span className="file-name">{file.filename}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="category-actions">
                <input
                  type="file"
                  ref={(el) => (fileInputRefs.current[key] = el)}
                  onChange={(e) => handleFileUpload(e, key)}
                  accept=".csv,.txt,.docx"
                  style={{ display: "none" }}
                />
                <button
                  className="btn-upload"
                  onClick={() => fileInputRefs.current[key]?.click()}
                  disabled={uploadingCategory === key}
                >
                  {uploadingCategory === key ? "Uploading..." : getCategoryStatus(key) ? "✓ Replace" : "Upload"}
                </button>
                {getCategoryStatus(key) && (
                  <button className="btn-clear" onClick={() => clearCategory(key)}>✕</button>
                )}
              </div>
              {getCategoryStatus(key) && <div className="status-dot active"></div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <span className="logo-icon">🔮</span>
          <span className="logo-text">SAGE-Twin</span>
          <span className="version">v3.0</span>
        </div>

        {/* Health Score */}
        <div className="health-display">
          <div className="health-circle" style={{ borderColor: getHealthColor(healthScore) }}>
            <span className="health-value">{healthScore.toFixed(0)}</span>
          </div>
          <span className="health-label">Health</span>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button className={`tab ${activeTab === "dashboard" ? "active" : ""}`} onClick={() => setActiveTab("dashboard")}>
            📊 Dashboard
          </button>
          <button className={`tab ${activeTab === "forecasts" ? "active" : ""}`} onClick={() => setActiveTab("forecasts")} disabled={!mlEnabled}>
            📈 Forecasts
          </button>
          <button className={`tab ${activeTab === "churn" ? "active" : ""}`} onClick={() => setActiveTab("churn")} disabled={!mlEnabled}>
            🚨 Churn
          </button>
          <button className={`tab ${activeTab === "monitoring" ? "active" : ""}`} onClick={() => setActiveTab("monitoring")} disabled={!mlEnabled}>
            🔍 Monitoring
          </button>
        </nav>

        {/* Actions */}
        <div className="header-actions">
          <button className="btn-secondary" onClick={() => setShowUploadPanel(!showUploadPanel)}>
            📁 Data Sources
          </button>
          <button className="btn-reset" onClick={resetState} disabled={loading}>
            🔄 Reset
          </button>
        </div>
      </header>

      {/* Upload Panel */}
      {renderUploadPanel()}

      {/* Main Content */}
      <main className="main-content">
        {!initialized ? (
          <div className="empty-state">
            <div className="empty-icon">📁</div>
            <h2>Upload Your Business Data</h2>
            <p>Start by uploading data files to each category above to initialize your Digital Twin.</p>
          </div>
        ) : (
          <>
            {activeTab === "dashboard" && renderDashboardTab()}
            {activeTab === "forecasts" && renderForecastTab()}
            {activeTab === "churn" && renderChurnTab()}
            {activeTab === "monitoring" && renderMonitoringTab()}
          </>
        )}
      </main>

      {/* ML Status Badge */}
      {mlEnabled && (
        <div className="ml-badge">
          <span className="ml-dot"></span>
          ML Enabled
        </div>
      )}

      {/* Session Badge */}
      {sessionId && (
        <div className="session-badge" title={`Session: ${sessionId}`}>
          <span className="session-icon">👤</span>
          Your Session
        </div>
      )}
    </div>
  );
}

export default App;
