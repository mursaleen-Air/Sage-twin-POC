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
} from "recharts";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

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

  // UI State
  const [showUploadPanel, setShowUploadPanel] = useState(true);

  const fileInputRefs = useRef({});

  // Load categories on mount
  useEffect(() => {
    axios.get(`${API_URL}/data-categories`).then(res => {
      setCategories(res.data.categories || {});
    });
    loadDataSources();
  }, []);

  const loadDataSources = async () => {
    try {
      const res = await axios.get(`${API_URL}/sources`);
      setDataSources(res.data);

      // Check if initialized
      if (res.data.sources?.total_files > 0) {
        setInitialized(true);
        // Load state
        const stateRes = await axios.get(`${API_URL}/state`);
        if (!stateRes.data.error) {
          setTwinState(stateRes.data);
          setHealthScore(stateRes.data.health_score || 0);
        }
      }
    } catch (error) {
      console.error("Failed to load sources:", error);
    }
  };

  // File Upload Handler
  const handleFileUpload = async (event, category) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploadingCategory(category);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_URL}/upload/${category}`, formData);

      if (res.data.success) {
        setInitialized(true);
        setHealthScore(res.data.health_score || 0);
        await loadDataSources();

        // Refresh state
        const stateRes = await axios.get(`${API_URL}/state`);
        if (!stateRes.data.error) {
          setTwinState(stateRes.data);
        }
      } else {
        alert(`Upload failed: ${res.data.error}`);
      }
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Upload failed. Please check the file format.");
    }

    setUploadingCategory(null);
    setLoading(false);

    // Reset file input
    if (fileInputRefs.current[category]) {
      fileInputRefs.current[category].value = "";
    }
  };

  // Clear category
  const clearCategory = async (category) => {
    try {
      await axios.delete(`${API_URL}/sources/${category}`);
      await loadDataSources();

      // Refresh state
      const stateRes = await axios.get(`${API_URL}/state`);
      if (!stateRes.data.error) {
        setTwinState(stateRes.data);
        setHealthScore(stateRes.data.health_score || 0);
      } else {
        setInitialized(false);
        setTwinState(null);
      }
    } catch (error) {
      console.error("Clear failed:", error);
    }
  };

  // Run Simulation
  const runSimulation = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/simulate`, adjustments);
      setSimulationResult(res.data);
      setHealthScore(res.data.health_score);

      if (res.data.comparison) {
        setTwinState(prev => ({
          ...prev,
          current_state: res.data.comparison.projected,
          deltas: res.data.comparison.deltas
        }));
      }
    } catch (error) {
      console.error("Simulation failed:", error);
    }
    setLoading(false);
  };

  // Reset State
  const resetState = async () => {
    try {
      const res = await axios.post(`${API_URL}/reset`);
      if (res.data.success) {
        setTwinState(res.data.state);
        setSimulationResult(null);
        setAdjustments({
          price: 0,
          marketing_spend: 0,
          costs: 0,
          delivery_delay: 0,
          market_shock: false,
        });
        setHealthScore(res.data.state.health_score);
      }
    } catch (error) {
      console.error("Reset failed:", error);
    }
  };

  // Format helpers
  const formatValue = (value, metric) => {
    if (typeof value !== "number") return value;
    if (["revenue", "costs", "marketing_spend", "profit"].includes(metric)) {
      return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        minimumFractionDigits: 0,
      }).format(value);
    }
    if (["churn_rate", "risk_score", "sentiment"].includes(metric)) {
      return `${value.toFixed(1)}${metric === "churn_rate" ? "%" : ""}`;
    }
    return value.toLocaleString();
  };

  const getHealthColor = (score) => {
    if (score >= 70) return COLORS.success;
    if (score >= 50) return COLORS.warning;
    return COLORS.danger;
  };

  const getDeltaClass = (delta) => {
    if (delta > 0) return "positive";
    if (delta < 0) return "negative";
    return "neutral";
  };

  const getForecastData = () => {
    if (!simulationResult?.forecast?.projections) return [];
    const { projections } = simulationResult.forecast;
    const months = projections.revenue?.map(p => p.month) || [];

    return months.map((month, i) => ({
      month,
      revenue: projections.revenue?.[i]?.value / 1000 || 0,
      sentiment: projections.sentiment?.[i]?.value || 0,
      risk: projections.risk_score?.[i]?.value || 0,
    }));
  };

  // Get upload status for a category
  const getCategoryStatus = (category) => {
    const catData = dataSources.categories?.[category];
    return {
      uploaded: catData?.uploaded || false,
      fileCount: catData?.file_count || 0,
      files: dataSources.sources?.sources?.[category]?.files || []
    };
  };

  return (
    <div className="app-container twin-mode">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <span className="logo-icon">🏢</span>
            <div className="logo-text">
              <h1>SAGE-Twin</h1>
              <span className="logo-subtitle">Digital Twin POC</span>
            </div>
          </div>
        </div>

        <div className="header-center">
          {initialized && (
            <div className="health-indicator">
              <div
                className="health-ring"
                style={{
                  background: `conic-gradient(${getHealthColor(healthScore)} ${healthScore * 3.6}deg, rgba(255,255,255,0.1) 0deg)`
                }}
              >
                <div className="health-inner">
                  <span className="health-value">{Math.round(healthScore)}</span>
                  <span className="health-label">Health</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="header-right">
          <button
            className={`btn btn-ghost ${showUploadPanel ? "active" : ""}`}
            onClick={() => setShowUploadPanel(!showUploadPanel)}
          >
            📁 Data Sources
          </button>

          {initialized && (
            <button className="btn btn-ghost" onClick={resetState}>
              🔄 Reset
            </button>
          )}
        </div>
      </header>

      {/* Data Sources Panel */}
      {showUploadPanel && (
        <div className="data-sources-panel">
          <div className="panel-header">
            <h3>📂 Data Sources</h3>
            <span className="sources-count">
              {dataSources.sources?.total_files || 0} files uploaded
            </span>
          </div>

          <div className="categories-grid">
            {Object.entries(categories).map(([key, cat]) => {
              const status = getCategoryStatus(key);
              const isUploading = uploadingCategory === key;

              return (
                <div
                  key={key}
                  className={`category-card ${status.uploaded ? "has-data" : ""} ${isUploading ? "uploading" : ""}`}
                >
                  <div className="category-icon">{cat.icon}</div>
                  <div className="category-info">
                    <div className="category-name">{cat.name}</div>
                    <div className="category-desc">{cat.description}</div>
                    <div className="category-formats">
                      {cat.supported_formats.map(f => (
                        <span key={f} className="format-badge">.{f}</span>
                      ))}
                    </div>
                  </div>

                  <div className="category-status">
                    {status.uploaded ? (
                      <>
                        <div className="files-list">
                          {status.files.map((file, i) => (
                            <span key={i} className="file-badge">📄 {file}</span>
                          ))}
                        </div>
                        <button
                          className="btn btn-sm btn-danger"
                          onClick={() => clearCategory(key)}
                        >
                          ✕ Clear
                        </button>
                      </>
                    ) : (
                      <button
                        className="btn btn-sm btn-upload"
                        onClick={() => fileInputRefs.current[key]?.click()}
                        disabled={isUploading}
                      >
                        {isUploading ? "Uploading..." : "➕ Upload"}
                      </button>
                    )}
                  </div>

                  <input
                    type="file"
                    ref={el => fileInputRefs.current[key] = el}
                    accept={cat.supported_formats.map(f => `.${f}`).join(",")}
                    onChange={(e) => handleFileUpload(e, key)}
                    style={{ display: "none" }}
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Main Content */}
      {!initialized ? (
        <div className="welcome-screen">
          <div className="welcome-content">
            <div className="welcome-icon">🏭</div>
            <h2>Initialize Your Digital Twin</h2>
            <p>Upload your business data files to create your digital twin simulation environment. You can upload multiple files across different categories.</p>

            <div className="sample-format">
              <h4>Supported Data Types:</h4>
              <div className="data-types-list">
                <div className="data-type">💰 <strong>Revenue</strong> - Financial metrics (CSV)</div>
                <div className="data-type">👥 <strong>Customers</strong> - Customer data (CSV)</div>
                <div className="data-type">💬 <strong>Reviews</strong> - Feedback & reviews (CSV, DOCX, TXT)</div>
                <div className="data-type">📢 <strong>Marketing</strong> - Campaign data (CSV)</div>
                <div className="data-type">🚚 <strong>Operations</strong> - Delivery & logistics (CSV)</div>
              </div>
            </div>

            <button
              className="btn btn-primary btn-lg"
              onClick={() => setShowUploadPanel(true)}
            >
              📁 Upload Data Sources
            </button>
          </div>
        </div>
      ) : (
        <div className="main-layout">
          {/* Left Panel - Controls */}
          <div className="control-panel">
            <div className="panel-section">
              <h3 className="section-title">
                <span className="section-icon">⚙️</span>
                Simulation Controls
              </h3>

              {/* Sliders */}
              <div className="control-group">
                <div className="control-header">
                  <label>Price Change</label>
                  <span className={`control-value ${getDeltaClass(adjustments.price)}`}>
                    {adjustments.price > 0 ? "+" : ""}{adjustments.price}%
                  </span>
                </div>
                <input
                  type="range"
                  min="-30"
                  max="30"
                  value={adjustments.price}
                  onChange={(e) => setAdjustments(prev => ({ ...prev, price: Number(e.target.value) }))}
                  className="slider"
                />
              </div>

              <div className="control-group">
                <div className="control-header">
                  <label>Marketing Spend</label>
                  <span className={`control-value ${getDeltaClass(adjustments.marketing_spend)}`}>
                    {adjustments.marketing_spend > 0 ? "+" : ""}{adjustments.marketing_spend}%
                  </span>
                </div>
                <input
                  type="range"
                  min="-50"
                  max="100"
                  value={adjustments.marketing_spend}
                  onChange={(e) => setAdjustments(prev => ({ ...prev, marketing_spend: Number(e.target.value) }))}
                  className="slider"
                />
              </div>

              <div className="control-group">
                <div className="control-header">
                  <label>Cost Change</label>
                  <span className={`control-value ${getDeltaClass(adjustments.costs)}`}>
                    {adjustments.costs > 0 ? "+" : ""}{adjustments.costs}%
                  </span>
                </div>
                <input
                  type="range"
                  min="-30"
                  max="30"
                  value={adjustments.costs}
                  onChange={(e) => setAdjustments(prev => ({ ...prev, costs: Number(e.target.value) }))}
                  className="slider"
                />
              </div>

              <div className="control-group">
                <div className="control-header">
                  <label>Delivery Delay</label>
                  <span className={`control-value ${getDeltaClass(adjustments.delivery_delay)}`}>
                    {adjustments.delivery_delay > 0 ? "+" : ""}{adjustments.delivery_delay} days
                  </span>
                </div>
                <input
                  type="range"
                  min="-3"
                  max="7"
                  value={adjustments.delivery_delay}
                  onChange={(e) => setAdjustments(prev => ({ ...prev, delivery_delay: Number(e.target.value) }))}
                  className="slider"
                />
              </div>

              {/* Market Shock Toggle */}
              <div className="control-group toggle-group">
                <label>Market Shock</label>
                <button
                  className={`toggle-btn ${adjustments.market_shock ? "active" : ""}`}
                  onClick={() => setAdjustments(prev => ({ ...prev, market_shock: !prev.market_shock }))}
                >
                  {adjustments.market_shock ? "🔥 Active" : "💤 Off"}
                </button>
              </div>

              {/* Run Button */}
              <button
                className="btn btn-primary btn-full"
                onClick={runSimulation}
                disabled={loading}
              >
                {loading ? (
                  <><span className="spinner"></span> Simulating...</>
                ) : (
                  <>🚀 Run Simulation</>
                )}
              </button>
            </div>

            {/* Current State */}
            <div className="panel-section">
              <h3 className="section-title">
                <span className="section-icon">📊</span>
                Current State
              </h3>
              <div className="metrics-list">
                {twinState?.current_state && Object.entries(twinState.current_state).map(([key, value]) => {
                  const delta = twinState.deltas?.[key];
                  return (
                    <div key={key} className="metric-row">
                      <span className="metric-name">{key.replace(/_/g, " ")}</span>
                      <div className="metric-values">
                        <span className="metric-value">{formatValue(value, key)}</span>
                        {delta && delta.delta_previous_pct !== 0 && (
                          <span className={`metric-delta ${getDeltaClass(delta.delta_previous_pct)}`}>
                            {delta.direction === "up" ? "↑" : delta.direction === "down" ? "↓" : "→"}
                            {Math.abs(delta.delta_previous_pct).toFixed(1)}%
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Center - Results */}
          <div className="results-panel">
            {simulationResult ? (
              <>
                {/* Strategic Priority */}
                <div className={`priority-banner ${simulationResult.strategic_priority?.toLowerCase().replace(" ", "-")}`}>
                  <span className="priority-label">Strategic Priority:</span>
                  <span className="priority-value">{simulationResult.strategic_priority}</span>
                </div>

                {/* Health & Confidence */}
                <div className="scores-row">
                  <div className="score-card health">
                    <div className="score-ring" style={{
                      background: `conic-gradient(${getHealthColor(simulationResult.health_score)} ${simulationResult.health_score * 3.6}deg, rgba(255,255,255,0.1) 0deg)`
                    }}>
                      <div className="score-inner">
                        <span className="score-value">{Math.round(simulationResult.health_score)}</span>
                      </div>
                    </div>
                    <span className="score-label">Business Health</span>
                  </div>

                  <div className="score-card confidence">
                    <div className="score-ring" style={{
                      background: `conic-gradient(${COLORS.secondary} ${simulationResult.confidence * 3.6}deg, rgba(255,255,255,0.1) 0deg)`
                    }}>
                      <div className="score-inner">
                        <span className="score-value">{Math.round(simulationResult.confidence)}</span>
                      </div>
                    </div>
                    <span className="score-label">Confidence</span>
                  </div>

                  <div className="score-card rules">
                    <div className="rules-count">{simulationResult.total_rules_triggered}</div>
                    <span className="score-label">Rules Triggered</span>
                  </div>
                </div>

                {/* Before/After Comparison */}
                <div className="comparison-section">
                  <h3 className="section-title">
                    <span className="section-icon">📈</span>
                    Impact Analysis
                  </h3>
                  <div className="comparison-grid">
                    {simulationResult.comparison?.deltas &&
                      Object.entries(simulationResult.comparison.deltas).slice(0, 6).map(([key, data]) => (
                        <div key={key} className="comparison-card">
                          <div className="comparison-metric">{key.replace(/_/g, " ")}</div>
                          <div className="comparison-values">
                            <span className="before">{formatValue(data.baseline, key)}</span>
                            <span className="arrow">→</span>
                            <span className="after">{formatValue(data.current, key)}</span>
                          </div>
                          <div className={`comparison-delta ${getDeltaClass(data.delta_baseline_pct)}`}>
                            {data.delta_baseline_pct > 0 ? "+" : ""}{data.delta_baseline_pct.toFixed(1)}%
                          </div>
                        </div>
                      ))
                    }
                  </div>
                </div>

                {/* Forecast Chart */}
                {simulationResult.forecast && (
                  <div className="forecast-section">
                    <h3 className="section-title">
                      <span className="section-icon">🔮</span>
                      3-Month Forecast
                    </h3>
                    <div className="forecast-chart">
                      <ResponsiveContainer width="100%" height={200}>
                        <AreaChart data={getForecastData()}>
                          <defs>
                            <linearGradient id="revenueGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor={COLORS.success} stopOpacity={0.3} />
                              <stop offset="95%" stopColor={COLORS.success} stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor={COLORS.danger} stopOpacity={0.3} />
                              <stop offset="95%" stopColor={COLORS.danger} stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <XAxis dataKey="month" tick={{ fill: "#888", fontSize: 11 }} axisLine={false} />
                          <YAxis tick={{ fill: "#888", fontSize: 11 }} axisLine={false} />
                          <Tooltip
                            contentStyle={{
                              background: "rgba(15,15,26,0.95)",
                              border: "1px solid rgba(255,255,255,0.1)",
                              borderRadius: "8px"
                            }}
                          />
                          <Area type="monotone" dataKey="revenue" stroke={COLORS.success} fill="url(#revenueGrad)" name="Revenue (K)" />
                          <Area type="monotone" dataKey="risk" stroke={COLORS.danger} fill="url(#riskGrad)" name="Risk" />
                          <Line type="monotone" dataKey="sentiment" stroke={COLORS.secondary} name="Sentiment" strokeWidth={2} dot={false} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                    {simulationResult.forecast.summary && (
                      <div className={`forecast-summary ${simulationResult.forecast.summary.outlook}`}>
                        <strong>Outlook: {simulationResult.forecast.summary.outlook.toUpperCase()}</strong>
                        <p>{simulationResult.forecast.summary.description}</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Recommendations */}
                {simulationResult.recommendations?.length > 0 && (
                  <div className="recommendations-section">
                    <h3 className="section-title">
                      <span className="section-icon">💡</span>
                      Recommendations
                    </h3>
                    <ul className="recommendations-list">
                      {simulationResult.recommendations.map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Warnings */}
                {simulationResult.warnings?.length > 0 && (
                  <div className="warnings-section">
                    <h3 className="section-title">
                      <span className="section-icon">⚠️</span>
                      Warnings
                    </h3>
                    <ul className="warnings-list">
                      {simulationResult.warnings.map((warn, i) => (
                        <li key={i}>{warn}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Tradeoffs */}
                {simulationResult.tradeoffs?.length > 0 && (
                  <div className="tradeoffs-section">
                    <h3 className="section-title">
                      <span className="section-icon">⚖️</span>
                      Tradeoffs
                    </h3>
                    <ul className="tradeoffs-list">
                      {simulationResult.tradeoffs.map((trade, i) => (
                        <li key={i}>{trade}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <div className="empty-results">
                <div className="empty-icon">🎯</div>
                <h3>Ready to Simulate</h3>
                <p>Adjust the parameters on the left and click "Run Simulation" to see how changes propagate through your business.</p>
              </div>
            )}
          </div>

          {/* Right Panel - Agent Transparency */}
          <div className="transparency-panel">
            <h3 className="section-title">
              <span className="section-icon">🤖</span>
              Agent Activity
            </h3>

            {simulationResult?.agent_outputs ? (
              <div className="agents-list">
                {simulationResult.agent_outputs.map((agent, i) => (
                  <div key={i} className="agent-card">
                    <div className="agent-header">
                      <span className="agent-name">{agent.agent}</span>
                      <span className="agent-confidence">{agent.confidence}%</span>
                    </div>

                    {agent.rules_triggered?.length > 0 && (
                      <div className="agent-rules">
                        {agent.rules_triggered.slice(0, 3).map((rule, j) => (
                          <div key={j} className="rule-item">{rule}</div>
                        ))}
                      </div>
                    )}

                    {agent.warnings?.length > 0 && (
                      <div className="agent-warnings">
                        {agent.warnings.map((warn, j) => (
                          <div key={j} className="warning-item">⚠️ {warn}</div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="agents-placeholder">
                <p>Run a simulation to see agent activity</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
