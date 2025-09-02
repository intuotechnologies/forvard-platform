// src/pages/GraphPage.jsx

import React, { useEffect, useState, useRef, useCallback } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import financialDataService from "../services/financialDataService";
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Brush,
  Legend
} from "recharts";
import html2canvas from "html2canvas";
import "../styles/graphpage.css";
import measureLabels from "../utils/measureLabels";
import { Download, Save, RefreshCw } from "lucide-react";
import { validateAssetAccess } from '../config/assets';

const GraphPage = () => {
  const { state } = useLocation();
  const navigate = useNavigate();
  const dropdownRef = useRef(null);
  const comparisonDropdownRef = useRef(null);

  const { isAuthenticated, token } = useAuth();

  const {
    symbol: ticker,
    category,
    fromDate,
    toDate,
    measureKey,
    measureLabel
  } = state || {};

  const [data, setData] = useState([]);
  const [allData, setAllData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);
  const [showDownloadMenu, setShowDownloadMenu] = useState(false);
  const [showComparisonOptions, setShowComparisonOptions] = useState(false);
  const [comparisonMeasure, setComparisonMeasure] = useState(null);
  const [comparisonOptions, setComparisonOptions] = useState([]);
  const [isZoomed, setIsZoomed] = useState(false);

  // Verifica che i parametri necessari siano presenti
  useEffect(() => {
    if (!state || !ticker || !category || !fromDate || !toDate || !measureKey) {
      console.error('Missing required parameters for GraphPage');
      navigate('/dashboard');
      return;
    }

    // VALIDAZIONE: Solo asset base possono essere graficati
    if (!validateAssetAccess(ticker, category)) {
      console.error(`Asset ${ticker} (${category}) is not available for graphing - restricted to base assets only`);
      alert(`Access Restricted: Asset ${ticker} is not available for charting. Only base assets can be visualized in graphs.`);
      navigate('/dashboard');
      return;
    }

    if (!isAuthenticated) {
      console.log('User not authenticated, redirecting to login');
      navigate('/login');
      return;
    }
  }, [state, ticker, category, fromDate, toDate, measureKey, navigate, isAuthenticated]);

  // Funzione per calcolare le metriche di riepilogo
  const calculateMetrics = useCallback((dataToUse) => {
    if (!dataToUse || dataToUse.length === 0) return;
    
    console.log(`Calculating metrics using ${dataToUse.length} records`);
    
    // 1. Avg Vol (media della varianza annualizzata)
    const avgVol = dataToUse.reduce((sum, d) => sum + d[measureKey], 0) / dataToUse.length;
    
    // 2. Vol of Vol (volatilità della volatilità)
    const volatilities = dataToUse.map(d => d[measureKey]);
    const avgVolatility = volatilities.reduce((sum, vol) => sum + vol, 0) / volatilities.length;
    const volOfVol = Math.sqrt(
      volatilities.reduce((sum, vol) => sum + Math.pow(vol - avgVolatility, 2), 0) / volatilities.length
    );
    
    // 3. Avg Return (rendimento medio annualizzato)
{/*    const returns = [];
    for (let i = 1; i < dataToUse.length; i++) {
      const prevClose = dataToUse[i-1].close_price || dataToUse[i-1].close;
      const currentClose = dataToUse[i].close_price || dataToUse[i].close;
      if (prevClose && currentClose && prevClose > 0 && currentClose > 0) {
        returns.push(Math.log(currentClose) - Math.log(prevClose));
      }
    }
    
    const avgDailyReturn = returns.length > 0 ? returns.reduce((sum, ret) => sum + ret, 0) / returns.length : 0;
    const avgAnnualizedReturn = avgDailyReturn * 252 * 100; // Annualizzato e in percentuale
 */}
 
    // 3. Avg Return (rendimento medio del periodo)
        const returns = [];
        for (let i = 1; i < dataToUse.length; i++) {
          const prevClose = dataToUse[i-1].close_price || dataToUse[i-1].close;
          const currentClose = dataToUse[i].close_price || dataToUse[i].close;
          if (prevClose && currentClose && prevClose > 0 && currentClose > 0) {
            returns.push(Math.log(currentClose) - Math.log(prevClose));
          }
        }

        const avgDailyReturn = returns.length > 0 ? returns.reduce((sum, ret) => sum + ret, 0) / returns.length : 0;
        const avgPeriodReturn = avgDailyReturn  * 100; // Media del periodo effettivo

    // 4. Avg Volume
    const volumes = dataToUse.map(d => d.volume).filter(Boolean);
    const avgVolume = volumes.length > 0 ? 
      Math.round(volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length) : 0;
    
    const newSummary = { 
      avgVol: avgVol, 
      volOfVol: volOfVol, 
      avgReturn: avgPeriodReturn,
      avgVolume: avgVolume
    };
    
    console.log("New metrics calculated:", newSummary);
    setSummary(newSummary);
  }, [measureKey]);

  

  useEffect(() => {
    function handleClickOutside(event) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target)
      ) {
        setShowDownloadMenu(false);
      }
      if (
        comparisonDropdownRef.current &&
        !comparisonDropdownRef.current.contains(event.target)
      ) {
        setShowComparisonOptions(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Carica i dati dall'API ForVARD
  useEffect(() => {
    if (!ticker || !category || !fromDate || !toDate || !measureKey || !isAuthenticated || !token) {
      return;
    }

    loadFinancialData();
  }, [ticker, fromDate, toDate, measureKey, category, isAuthenticated, token]);

  const loadFinancialData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      console.log('Loading financial data from ForVARD API...');
      
      // Controllo autenticazione
      if (!isAuthenticated || !token) {
        console.log('User not authenticated, redirecting to login');
        navigate('/login');
        return;
      }

      // Parametri corretti secondo l'API spec
      const filters = {
        symbol: ticker,           // symbol (query)
        asset_type: category,     // asset_type (query) 
        start_date: fromDate,     // start_date (query)
        end_date: toDate,         // end_date (query)
        limit: 1000,              // limit per ottenere più dati
        page: 1                   // page default
      };

      console.log('API query filters:', filters);

      // Chiama l'API ForVARD con il token
      const response = await financialDataService.getFinancialData(token, filters);

      console.log('API response:', response);

      // Estrai i dati dalla risposta (potrebbe essere paginata)
      const apiData = response.data || response;

      if (!apiData || apiData.length === 0) {
        setError(`No data found for ${ticker} (${category}) in the selected date range: ${fromDate} to ${toDate}`);
        setLoading(false);
        return;
      }

      console.log('API data received:', apiData.length, 'records');

      // Rileva le misure disponibili nei dati (esclude campi non-volatilità)
      const detectedMeasures = Object.keys(apiData[0]).filter(
        (key) =>
          !["observation_date", "symbol", "asset_type", "volume", "trades",
            "open_price", "close_price", "high_price", "low_price"].includes(key)
      );

      console.log('Detected volatility measures:', detectedMeasures);

      const allowedKeys = Object.keys(measureLabels);

      const comparisonList = detectedMeasures
        .filter((key) => key !== measureKey && allowedKeys.includes(key))
        .map((key) => ({
          key,
          label: measureLabels[key]
        }));
      
      setComparisonOptions(comparisonList);

      // Formattazione dati per il grafico
      const formattedData = apiData.map((item) => {
        // Controlla se la misura richiesta esiste nei dati
        if (item[measureKey] === null || item[measureKey] === undefined) {
          console.warn(`Missing ${measureKey} data for ${item.symbol} on ${item.observation_date}`);
        }

        const baseData = {
          date: item.observation_date,
          [measureKey]: item[measureKey] ? Math.sqrt(item[measureKey]) * Math.sqrt(252) * 100 : 0,
          open: item.open_price,
          close: item.close_price,
          high: item.high_price,
          low: item.low_price,
          volume: item.volume,
          trades: item.trades,
          
          // Mantieni riferimenti originali per calcoli
          open_price: item.open_price,
          close_price: item.close_price,
          high_price: item.high_price,
          low_price: item.low_price
        };

        // Aggiungi tutte le misure rilevate per consentire confronti
        detectedMeasures.forEach((measure) => {
          if (item[measure] !== undefined && item[measure] !== null) {
            baseData[measure] = Math.sqrt(item[measure]) * Math.sqrt(252) * 100;
          }
        });

        return baseData;
      });

      // Filtra i dati che hanno valori validi per la misura selezionata
      const validData = formattedData.filter(item => 
        item[measureKey] && item[measureKey] > 0
      );

      if (validData.length === 0) {
        setError(`No valid ${measureKey} data found for ${ticker}`);
        setLoading(false);
        return;
      }

      // Ordina per data
      validData.sort((a, b) => new Date(a.date) - new Date(b.date));

      console.log(`Processed ${validData.length} valid data points`);
      
      setData(validData);
      setAllData(validData);
      
      // Calcola le metriche iniziali usando tutti i dati
      calculateMetrics(validData);
      
    } catch (error) {
      console.error('Failed to load financial data from ForVARD API:', error);
      
      // Gestione errori di autenticazione
      if (error.message.includes('401') || error.message.includes('Unauthorized')) {
        console.log('Authentication error, redirecting to login');
        navigate('/login');
        return;
      }

      if (error.message.includes('403') || error.message.includes('Forbidden')) {
        setError('Access denied. You may not have permission to access this data or have exceeded your limits.');
      } else if (error.message.includes('404')) {
        setError('Financial data not found. The requested symbol or date range may not be available.');
      } else if (error.message.includes('500')) {
        setError('Server error. The ForVARD API is temporarily unavailable.');
      } else if (error.message.includes('fetch')) {
        setError('Connection error. Please check if the ForVARD API is running at http://volare.unime.it:8443');
      } else {
        setError(`Failed to load data: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  }, [calculateMetrics, ticker, category, fromDate, toDate, measureKey, isAuthenticated, token, navigate]);

  const saveImage = () => {
    const chart = document.querySelector(".recharts-wrapper");
    if (chart) {
      const brush = chart.querySelector(".recharts-brush");
      if (brush) brush.style.display = "none";
      
      html2canvas(chart).then((canvas) => {
        const link = document.createElement("a");
        link.href = canvas.toDataURL("image/png");
        link.download = `forvard_chart_${ticker}_${fromDate}_${toDate}_${measureKey}${comparisonMeasure ? `_vs_${comparisonMeasure}` : ""}.png`;
        link.click();
        
        if (brush) brush.style.display = "block";
      });
    }
  };

  // Funzione per tornare alla dashboard con i parametri attuali preselezionati
  const handleUpdateGraph = () => {
    console.log('Navigating back to dashboard with current parameters for update...');
    
    // Naviga alla dashboard passando i parametri attuali come state
    navigate('/dashboard', {
      state: {
        // Parametri da preselezionare nella dashboard
        preselectedSymbol: ticker,
        preselectedCategory: category,
        preselectedFromDate: fromDate,
        preselectedToDate: toDate,
        preselectedMeasureKey: measureKey,
        preselectedMeasureLabel: measureLabel,
        // Flag per indicare che stiamo tornando per un update
        isUpdate: true,

      }
    });
  };

  const downloadData = (format) => {
    if (!isAuthenticated) {
      navigate("/login");
      return;
    }

    let content = "";
    const headers = ["Date", measureLabels[measureKey] || measureLabel];
    const comparisonLabel = measureLabels[comparisonMeasure] || comparisonMeasure;
    if (comparisonMeasure) headers.push(comparisonLabel);

    if (format === "csv") {
      content = headers.join(",") + "\n";
      content += data
        .map((d) => {
          let row = `${d.date},${d[measureKey]}`;
          if (comparisonMeasure) row += `,${d[comparisonMeasure] || "N/A"}`;
          return row;
        })
        .join("\n");
    } else if (format === "txt") {
      content = data
        .map((d) => {
          let row = `Date: ${d.date}, ${measureLabels[measureKey] || measureLabel}: ${d[measureKey]}`;
          if (comparisonMeasure) {
            row += `, ${comparisonLabel}: ${d[comparisonMeasure] || "N/A"}`;
          }
          return row;
        })
        .join("\n");
    } else if (format === "xlsx") {
      import("xlsx").then((XLSX) => {
        const ws = XLSX.utils.json_to_sheet(data);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, "ForVARD Data");
        XLSX.writeFile(
          wb,
          `forvard_data_${ticker}_${fromDate}_${toDate}_${measureKey}${
            comparisonMeasure ? `_vs_${comparisonMeasure}` : ""
          }.xlsx`
        );
      });
      return;
    }

    setShowDownloadMenu(false);
    const blob = new Blob([content], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `forvard_data_${ticker}_${fromDate}_${toDate}_${measureKey}${
      comparisonMeasure ? `_vs_${comparisonMeasure}` : ""
    }.${format}`;
    a.click();
  };

  const selectComparisonMeasure = (measureKey) => {
    const isSame = measureKey === comparisonMeasure;
    setComparisonMeasure(isSame ? null : measureKey);
    console.log("Comparison metric selected:", measureKey);
    setShowComparisonOptions(false);
  };

  // Metodo per gestire il brush
  const handleBrushChange = (brushData) => {
    console.log("Brush change event:", brushData);
    
    if (!brushData || (!brushData.startIndex && brushData.startIndex !== 0)) {
      return;
    }
    
    // Se il brush seleziona tutto il dataset, considera come vista completa
    if (brushData.startIndex === 0 && brushData.endIndex === allData.length - 1) {
      setIsZoomed(false);
      calculateMetrics(allData);
      return;
    }
    
    // Altrimenti, considera come zoom e calcola le metriche sul range selezionato
    setIsZoomed(true);
    const visibleData = allData.slice(brushData.startIndex, brushData.endIndex + 1);
    calculateMetrics(visibleData);
  };

  // Mostra loading
  if (loading) {
    return (
      <div className="loading-container" style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '400px',
        fontSize: '18px'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ marginBottom: '10px' }}>Loading financial data from ForVARD API...</div>
          <div style={{ fontSize: '14px', color: '#666' }}>
            Querying: {ticker} ({category}) from {fromDate} to {toDate}
          </div>
          <div style={{ fontSize: '12px', color: '#999', marginTop: '5px' }}>
            Endpoint: http://volare.unime.it:8443/financial-data
          </div>
        </div>
      </div>
    );
  }

  // Mostra errore
  if (error) {
    return (
      <div className="error-container" style={{ 
        display: 'flex', 
        flexDirection: 'column',
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '400px',
        padding: '20px'
      }}>
        <h2 style={{ color: '#dc3545', marginBottom: '15px' }}>ForVARD API Error</h2>
        <p style={{ textAlign: 'center', marginBottom: '10px' }}>{error}</p>
        <div style={{ fontSize: '14px', color: '#666', textAlign: 'center', marginBottom: '20px' }}>
          <p>Query attempted: <strong>{ticker}</strong> ({category})</p>
          <p>Date range: {fromDate} to {toDate}</p>
          <p>Measure: {measureLabel}</p>
          <p>API: http://volare.unime.it:8443/financial-data</p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            onClick={() => navigate('/login')}
            style={{
              padding: '10px 20px',
              background: '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Re-login
          </button>
          <button 
            onClick={handleUpdateGraph}
            style={{
              padding: '10px 20px',
              background: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Update Parameters
          </button>
        </div>
      </div>
    );
  }

  // Se non ci sono dati
  if (!data || data.length === 0) {
    return (
      <div className="no-data-container" style={{ 
        display: 'flex', 
        flexDirection: 'column',
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '400px',
        padding: '20px'
      }}>
        <h2>No Data Available</h2>
        <p>No volatility data found for <strong>{ticker}</strong> in the selected date range.</p>
        <div style={{ fontSize: '14px', color: '#666', marginBottom: '20px' }}>
          <p>Date range: {fromDate} to {toDate}</p>
          <p>Measure: {measureLabel}</p>
          <p>Category: {category}</p>
        </div>
        <button 
          onClick={handleUpdateGraph}
          style={{
            padding: '10px 20px',
            background: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Update Parameters
        </button>
      </div>
    );
  }

  return (
    <div className="graph-page">
      <section className="chart-section-centered">
        <div className="main-content">
          <div className="chart-and-controls">
            <div className="chart-container">
              <div className="chart-header">
                <h2>
                  {ticker} - {measureLabels[measureKey] || measureLabel} ({fromDate} → {toDate})
                </h2>
                <div className="data-source-indicator" style={{
                  fontSize: '12px',
                  color: '#666',
                  fontStyle: 'italic'
                }}>
                </div>
              </div>

              <div className="chart-wrapper">
                <LineChart width={900} height={450} data={data}>
                  <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
                  <XAxis dataKey="date" />
                  <YAxis unit="%" />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="linear"
                    dataKey={measureKey}
                    name={measureLabels[measureKey] || measureLabel}
                    stroke="#d62728"
                    dot={false}
                    strokeWidth={1.5}
                  />
                  {comparisonMeasure && (
                    <Line
                      type="linear"
                      dataKey={comparisonMeasure}
                      name={measureLabels[comparisonMeasure] || comparisonMeasure}
                      stroke="#2ca02c"
                      dot={false}
                      strokeWidth={1.5}
                    />
                  )}
                  <Brush 
                    dataKey="date" 
                    height={30} 
                    stroke="#8884d8" 
                    onChange={handleBrushChange}
                  />
                </LineChart>
                <div className="plot-description">
                  Visualized as annualized percentage volatility
                </div>
              </div>
            </div>

            <div className="controls-panel">
              <button className="action-btn update-btn" onClick={handleUpdateGraph}>
                <RefreshCw size={16} />
                <span>Update</span>
              </button>
              
              <div className={`dropdown-container ${showDownloadMenu ? "active" : ""}`} ref={dropdownRef}>
                <button className="action-btn download-btn" onClick={() => setShowDownloadMenu(!showDownloadMenu)}>
                  <Download size={16} />
                  <span>Download</span>
                </button>
                {showDownloadMenu && (
                  <div className="dropdown-menu">
                    <button onClick={() => downloadData("csv")}>CSV</button>
                    <button onClick={() => downloadData("txt")}>TXT</button>
                    <button onClick={() => downloadData("xlsx")}>XLSX</button>
                  </div>
                )}
              </div>

              <button className="action-btn save-img-btn" onClick={saveImage}>
                <Save size={16} />
                <span>Save image</span>
              </button>

              <div className="comparison-button-wrapper" ref={comparisonDropdownRef}>
                <button
                  className="action-btn"
                  onClick={() => setShowComparisonOptions(!showComparisonOptions)}
                >
                  {comparisonMeasure ? "Change Metric" : "Add comparison"}
                </button>
                {showComparisonOptions && (
                  <div className="comparison-options">
                    {comparisonOptions.map((option) => (
                      <div
                        key={option.key}
                        className={`option-item ${comparisonMeasure === option.key ? "selected" : ""}`}
                        onClick={() => selectComparisonMeasure(option.key)}
                      >
                        {option.label}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {comparisonMeasure && (
                <div className="active-comparison">
                  <div className="comparison-details">
                    <div className="metric-indicator primary">
                      <span className="color-dot primary"></span>
                      <span>{measureLabels[measureKey] || measureLabel}</span>
                    </div>
                    <div className="vs-label">vs</div>
                    <div className="metric-indicator secondary">
                      <span className="color-dot secondary"></span>
                      <span>{measureLabels[comparisonMeasure] || comparisonMeasure}</span>
                    </div>
                  </div>
                  <button className="remove-comparison-btn" onClick={() => setComparisonMeasure(null)}>
                    Remove comparison
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {summary && (
        <section className="summary-section">
          <div className="summary-container">
            <h2>
              Market Summary - {ticker}
              {isZoomed ? (
                <>
                  <span className="zoom-indicator"> (Zoomed View)</span>
                  
                </>
              ) : (
                <span className="full-view-indicator"></span>
              )}
            </h2>
            <div className="summary-cards">
              <div className="summary-card">
                <div className="card-title">Avg Volatility</div>
                <div className="card-value">{summary.avgVol.toFixed(2)}%</div>
              </div>
              <div className="summary-card">
                <div className="card-title">Vol-of-Vol</div>
                <div className="card-value">{summary.volOfVol.toFixed(2)}%</div>
              </div>
              <div className="summary-card">
                <div className="card-title">Avg Return</div>
                <div className="card-value">{summary.avgReturn.toFixed(2)}%</div>
              </div>
              <div className="summary-card">
                <div className="card-title">Avg Volume</div>
                <div className="card-value">{summary.avgVolume.toLocaleString()}</div>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default GraphPage;