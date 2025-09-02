import React, { useState, useEffect } from "react";
import { FaCalendarAlt, FaChevronDown, FaChartLine, FaInfoCircle } from "react-icons/fa";
import "../styles/dashboard.css";
import { useNavigate, useLocation } from "react-router-dom";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import { registerLocale, setDefaultLocale } from "react-datepicker";
import enUS from "date-fns/locale/en-US";
import { BASE_ASSETS, validateAssetAccess } from '../config/assets';

registerLocale("en", enUS);
setDefaultLocale("en");

// USO SOLO BASE_ASSETS - tutti gli utenti vedono solo questi asset per i grafici
const assetCategories = {
  stocks: {
    name: "Stocks",
    symbols: BASE_ASSETS.stocks
  },
  forex: {
    name: "Forex", 
    symbols: BASE_ASSETS.forex
  },
  futures: {
    name: "Futures",
    symbols: BASE_ASSETS.futures
  }
};

const formatDisplayedDate = (date) => {
  if (!date) return "yyyy/mm/dd";
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}/${month}/${day}`;
};

const generateYearOptions = () => {
  const currentYear = new Date().getFullYear();
  const years = [];
  for (let year = currentYear; year >= currentYear - 10; year--) {
    years.push(year);
  }
  return years;
};

// Funzione per calcolare le date di default
const getDefaultDates = () => {
  const today = new Date();
  const oneMonthAgo = new Date(today);
  oneMonthAgo.setMonth(today.getMonth() - 1);
  
  const oneYearBeforeOneMonthAgo = new Date(oneMonthAgo);
  oneYearBeforeOneMonthAgo.setFullYear(oneMonthAgo.getFullYear() - 1);
  
  return {
    fromDate: oneYearBeforeOneMonthAgo,
    toDate: oneMonthAgo
  };
};

// Funzione per convertire una stringa data in oggetto Date
const parseDate = (dateString) => {
  if (!dateString) return null;
  const [year, month, day] = dateString.split('-');
  return new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
};

const years = generateYearOptions();

const Dashboard = () => {
  const navigate = useNavigate();
  const { state } = useLocation();

  // Estrai i parametri preselezionati dallo state (se presenti)
  const {
    preselectedSymbol,
    preselectedCategory,
    preselectedFromDate,
    preselectedToDate,
    preselectedMeasureKey,
    preselectedMeasureLabel,
    isUpdate,
    updateMessage
  } = state || {};

  // Inizializza le date con i valori di default o quelli preselezionati
  const defaultDates = getDefaultDates();
  
  const [categoryMenuVisible, setCategoryMenuVisible] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState(preselectedCategory || null);
  const [datasetHovered, setDatasetHovered] = useState(false);
  const [modelMenuVisible, setModelMenuVisible] = useState(false);
  const [activeSubmenu, setActiveSubmenu] = useState(null);
  const [symbol, setSymbol] = useState(preselectedSymbol || "");
  const [suggestions, setSuggestions] = useState([]);
  const [fromDate, setFromDate] = useState(
    preselectedFromDate ? parseDate(preselectedFromDate) : defaultDates.fromDate
  );
  const [toDate, setToDate] = useState(
    preselectedToDate ? parseDate(preselectedToDate) : defaultDates.toDate
  );
  const [measureKey, setMeasureKey] = useState(preselectedMeasureKey || "");
  const [measureLabel, setMeasureLabel] = useState(preselectedMeasureLabel || "Measure");
  
  // Stato per mostrare il messaggio di update
  const [showUpdateMessage, setShowUpdateMessage] = useState(isUpdate || false);

  // Effect per gestire i parametri preselezionati e mostrare il messaggio
  useEffect(() => {
    if (isUpdate && updateMessage) {
      console.log('Dashboard loaded in update mode:', {
        symbol: preselectedSymbol,
        category: preselectedCategory,
        fromDate: preselectedFromDate,
        toDate: preselectedToDate,
        measureKey: preselectedMeasureKey
      });
      
      // Mostra il messaggio per alcuni secondi
      setTimeout(() => {
        setShowUpdateMessage(false);
      }, 5000);
    }

    // Se c'è un symbol preselezionato, attiva la modalità di input
    if (preselectedSymbol && preselectedCategory) {
      setDatasetHovered(true);
    }
  }, [isUpdate, updateMessage, preselectedSymbol, preselectedCategory, preselectedFromDate, preselectedToDate, preselectedMeasureKey]);

  const handleGo = () => {
    if (selectedCategory && symbol && fromDate && toDate && measureKey) {
      // VALIDAZIONE: Solo asset base possono essere graficati
      if (!validateAssetAccess(symbol, selectedCategory)) {
        alert(`Asset ${symbol} is restricted for graphing. Only base assets can be visualized in charts.`);
        return;
      }

      const formatDateForNavigation = (date) => {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
      };

      const formattedFromDate = formatDateForNavigation(fromDate);
      const formattedToDate = formatDateForNavigation(toDate);

      navigate("/graph", {
        state: {
          category: selectedCategory,
          symbol,
          fromDate: formattedFromDate,
          toDate: formattedToDate,
          measureKey,
          measureLabel
        },
      });
    } else {
      alert("Please select all fields!");
    }
  };

  const handleCategorySelect = (category) => {
    setSelectedCategory(category);
    setCategoryMenuVisible(false);
    
    // Se stiamo cambiando categoria e non è quella preselezionata, reset del symbol
    if (category !== preselectedCategory) {
      setSymbol("");
      setSuggestions([]);
    }
  };

  const handleSymbolChange = (e) => {
    const value = e.target.value.toUpperCase();
    setSymbol(value);

    if (selectedCategory && value) {
      // Filtra solo tra i BASE_ASSETS per la categoria selezionata
      const filteredSuggestions = assetCategories[selectedCategory].symbols.filter(symbol =>
        symbol.startsWith(value)
      );
      setSuggestions(filteredSuggestions);
    } else {
      setSuggestions([]);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setSymbol(suggestion);
    setSuggestions([]);
  };

  const handleMeasureSelect = (key, label) => {
    setMeasureKey(key);
    setMeasureLabel(label);
    setModelMenuVisible(false);
    setActiveSubmenu(null);
  };

  // Simplified submenu handlers - no more timeout system
  const handleSubmenuEnter = (submenuName) => {
    setActiveSubmenu(submenuName);
  };

  // Close menus when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      // Close category menu if clicked outside
      if (categoryMenuVisible && !event.target.closest('.dropdown-d-container')) {
        setCategoryMenuVisible(false);
      }
      
      // Close model menu if clicked outside  
      if (modelMenuVisible && !event.target.closest('.dropdown-d-container')) {
        setModelMenuVisible(false);
        setActiveSubmenu(null);
      }
      
      // Close suggestions if clicked outside
      if (suggestions.length > 0 && !event.target.closest('.dropdown-d-container')) {
        setSuggestions([]);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [categoryMenuVisible, modelMenuVisible, suggestions.length]);

  // Custom DatePicker Input Component
  const CustomDatePickerInput = React.forwardRef(({ value, onClick, label }, ref) => (
    <div className="date-input-container" ref={ref}>
      <label className="input-label">{label}</label>
      <div className="date-picker-button" onClick={onClick}>
        <FaCalendarAlt className="date-icon" />
        <span className="formatted-date">{value}</span>
      </div>
    </div>
  ));

  CustomDatePickerInput.displayName = 'CustomDatePickerInput';

  const isFormValid = selectedCategory && symbol && fromDate && toDate && measureKey;

  return (
    <div className="dashboard-page">
      <div className="dashboard-container">
        {/* Header */}
        <div className="dashboard-header">
          <div className="header-icon-title">
            <FaChartLine className="header-icon" />
            <h1 className="dashboard-title">Financial Data Explorer</h1>
          </div>
        </div>

        {/* Update Message Banner */}
        {showUpdateMessage && updateMessage && (
          <div className="update-message-banner">
            <div className="update-message-content">
              <FaInfoCircle className="update-icon" />
              <span className="update-text">{updateMessage}</span>
              <button 
                className="close-message-btn"
                onClick={() => setShowUpdateMessage(false)}
              >
                ×
              </button>
            </div>
          </div>
        )}

        {/* Main Form Card */}
        <div className="form-card">
          <div className="form-grid">
            
            {/* Left Column - Asset Selection */}
            <div className="form-column">
              <h2 className="section-title">Asset Selection</h2>
              
              {/* Asset Category */}
              <div className="input-group">
                <label className="input-label">
                  Asset Category
                  {preselectedCategory && (
                    <span className="preselected-indicator"></span>
                  )}
                </label>
                <div className="dropdown-d-container">
                  <button
                    onClick={() => setCategoryMenuVisible(!categoryMenuVisible)}
                    className={`dropdown-d-button ${preselectedCategory ? 'preselected' : ''}`}
                  >
                    <span className={selectedCategory ? "selected" : "placeholder"}>
                      {selectedCategory ? assetCategories[selectedCategory].name : "Asset"}
                    </span>
                    <FaChevronDown className={`dropdown-d-arrow ${categoryMenuVisible ? 'rotated' : ''}`} />
                  </button>
                  
                  {categoryMenuVisible && (
                    <div className="dropdown-d-menu">
                      {Object.keys(assetCategories).map((category) => (
                        <button
                          key={category}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCategorySelect(category);
                          }}
                          className={`dropdown-d-item ${category === selectedCategory ? 'selected' : ''}`}
                        >
                          {assetCategories[category].name}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Symbol Input */}
              <div className="input-group">
                <label className="input-label">
                  Symbol
                  {preselectedSymbol && (
                    <span className="preselected-indicator"></span>
                  )}
                </label>
                <div className="dropdown-d-container">
                  {(datasetHovered || symbol) && selectedCategory ? (
                    <div style={{ position: "relative", width: "100%" }}>
                      <input
                        type="text"
                        placeholder="Type a symbol"
                        value={symbol}
                        onChange={handleSymbolChange}
                        className={`text-input ${preselectedSymbol ? 'preselected' : ''}`}
                        onFocus={() => setDatasetHovered(true)}
                        onBlur={() => setTimeout(() => setDatasetHovered(false), 200)}
                      />
                      {suggestions.length > 0 && (
                        <div className="suggestions-menu">
                          {suggestions.map((suggestion, index) => (
                            <button
                              key={index}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleSuggestionClick(suggestion);
                              }}
                              className="suggestion-item"
                            >
                              {suggestion}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    <button
                      className="dropdown-d-button"
                      onMouseEnter={() => selectedCategory && setDatasetHovered(true)}
                      onClick={() => selectedCategory && setDatasetHovered(true)}
                    >
                      <span className="placeholder">
                        Dataset
                      </span>
                    </button>
                  )}
                </div>
              </div>

              {/* Date Range */}
              <div className="date-grid">
                <div className="date-input-wrapper">
                  {preselectedFromDate && (
                    <div className="preselected-date-indicator"></div>
                  )}
                  <DatePicker
                    selected={fromDate}
                    onChange={date => setFromDate(date)}
                    customInput={<CustomDatePickerInput value={formatDisplayedDate(fromDate)} label="From" />}
                    dateFormat="yyyy/MM/dd"
                    locale="en"
                    popperPlacement="bottom-start"
                    popperModifiers={{
                      preventOverflow: {
                        enabled: true,
                      },
                      flip: {
                        enabled: true,
                      },
                    }}
                    renderCustomHeader={({ date, changeYear, changeMonth, decreaseMonth, increaseMonth, prevMonthButtonDisabled, nextMonthButtonDisabled }) => (
                      <div style={{ margin: 10, display: "flex", justifyContent: "center", alignItems: "center", gap: "10px" }}>
                        <button onClick={decreaseMonth} disabled={prevMonthButtonDisabled} style={{padding: "5px 10px"}}>{"<"}</button>
                        <select value={date.getFullYear()} onChange={({ target: { value } }) => changeYear(value)} style={{padding: "5px"}}>
                          {years.map((option) => (
                            <option key={option} value={option}>{option}</option>
                          ))}
                        </select>
                        <select value={date.getMonth()} onChange={({ target: { value } }) => changeMonth(value)} style={{padding: "5px"}}>
                          {Array.from({ length: 12 }, (_, i) =>
                            new Date(2000, i).toLocaleString('en', { month: 'long' })
                          ).map((option, i) => (
                            <option key={option} value={i}>{option}</option>
                          ))}
                        </select>
                        <button onClick={increaseMonth} disabled={nextMonthButtonDisabled} style={{padding: "5px 10px"}}>{">"}</button>
                      </div>
                    )}
                  />
                </div>
                
                <div className="date-input-wrapper">
                  {preselectedToDate && (
                    <div className="preselected-date-indicator"></div>
                  )}
                  <DatePicker
                    selected={toDate}
                    onChange={date => setToDate(date)}
                    customInput={<CustomDatePickerInput value={formatDisplayedDate(toDate)} label="To" />}
                    dateFormat="yyyy/MM/dd"
                    locale="en"
                    popperPlacement="bottom-start"
                    popperModifiers={{
                      preventOverflow: {
                        enabled: true,
                      },
                      flip: {
                        enabled: true,
                      },
                    }}
                    renderCustomHeader={({ date, changeYear, changeMonth, decreaseMonth, increaseMonth, prevMonthButtonDisabled, nextMonthButtonDisabled }) => (
                      <div style={{ margin: 10, display: "flex", justifyContent: "center", alignItems: "center", gap: "10px" }}>
                        <button onClick={decreaseMonth} disabled={prevMonthButtonDisabled} style={{padding: "5px 10px"}}>{"<"}</button>
                        <select value={date.getFullYear()} onChange={({ target: { value } }) => changeYear(value)} style={{padding: "5px"}}>
                          {years.map((option) => (
                            <option key={option} value={option}>{option}</option>
                          ))}
                        </select>
                        <select value={date.getMonth()} onChange={({ target: { value } }) => changeMonth(value)} style={{padding: "5px"}}>
                          {Array.from({ length: 12 }, (_, i) =>
                            new Date(2000, i).toLocaleString('en', { month: 'long' })
                          ).map((option, i) => (
                            <option key={option} value={i}>{option}</option>
                          ))}
                        </select>
                        <button onClick={increaseMonth} disabled={nextMonthButtonDisabled} style={{padding: "5px 10px"}}>{">"}</button>
                      </div>
                    )}
                  />
                </div>
              </div>
            </div>

            {/* Right Column - Measure Selection */}
            <div className="form-column">
              <h2 className="section-title">Volatility Measure</h2>
              
              <div className="input-group">
                <label className="input-label">
                  Select Measure
                  {preselectedMeasureKey && (
                    <span className="preselected-indicator"></span>
                  )}
                </label>
                <div className="dropdown-d-container">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setModelMenuVisible(!modelMenuVisible);
                    }}
                    className={`dropdown-d-button ${preselectedMeasureKey ? 'preselected' : ''}`}
                  >
                    <span className={measureKey ? "selected" : "placeholder"}>
                      {measureLabel}
                    </span>
                    <FaChevronDown className={`dropdown-d-arrow ${modelMenuVisible ? 'rotated' : ''}`} />
                  </button>
                  
                  {modelMenuVisible && (
                    <div className="dropdown-d-menu large">
                      {/* Bipower */}
                      <div
                        className="dropdown-d-group"
                        onMouseEnter={() => handleSubmenuEnter("bipower")}
                      >
                        <div className="group-header">
                          Bipower <span className="submenu-arrow">›</span>
                        </div>
                        
                        {activeSubmenu === "bipower" && (
                          <div className="submenu">
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("bv1", "Bipower Variation (1-min)");
                            }} className="submenu-item">
                              Bipower Variation (1-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("bv5", "Bipower Variation (5-min)");
                            }} className="submenu-item">
                              Bipower Variation (5-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("bv_ss", "Bipower Variation (Sub-sampled 5-min)");
                            }} className="submenu-item">
                              Bipower Variation (Sub-sampled 5-min)
                            </button>
                          </div>
                        )}
                      </div>
                      
                      {/* Realized Variance - Basic */}
                      <div
                        className="dropdown-d-group"
                        onMouseEnter={() => handleSubmenuEnter("realized")}
                      >
                        <div className="group-header">
                          Realized Variance <span className="submenu-arrow">›</span>
                        </div>
                        
                        {activeSubmenu === "realized" && (
                          <div className="submenu">
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rv1", "Realized Variance (1-min)");
                            }} className="submenu-item">
                              Realized Variance (1-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rv5", "Realized Variance (5-min)");
                            }} className="submenu-item">
                              Realized Variance (5-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rv5_ss", "Realized Variance (Sub-sampled 5-min)");
                            }} className="submenu-item">
                              Realized Variance (Sub-sampled 5-min)
                            </button>
                          </div>
                        )}
                      </div>

                      {/* Realized Semi-Variance */}
                      <div
                        className="dropdown-d-group"
                        onMouseEnter={() => handleSubmenuEnter("semivariance")}
                      >
                        <div className="group-header">
                          Realized Semi-Variance <span className="submenu-arrow">›</span>
                        </div>
                        
                        {activeSubmenu === "semivariance" && (
                          <div className="submenu">
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rsp1", "Realized Semi-Variance (+)(1-min)");
                            }} className="submenu-item">
                              Realized Semi-Variance (+)(1-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rsp5", "Realized Semi-Variance (+)(5-min)");
                            }} className="submenu-item">
                              Realized Semi-Variance (+)(5-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rsp5_ss", "Realized Semi-Variance (+)(Sub-Sampled 5-min)");
                            }} className="submenu-item">
                              Realized Semi-Variance (+)(Sub-Sampled 5-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rsn1", "Realized Semi-Variance (-)(1-min)");
                            }} className="submenu-item">
                              Realized Semi-Variance (-)(1-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rsn5", "Realized Semi-Variance (-)(5-min)");
                            }} className="submenu-item">
                              Realized Semi-Variance (-)(5-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rsn5_ss", "Realized Semi-Variance (-)(Sub-Sampled 5-min)");
                            }} className="submenu-item">
                              Realized Semi-Variance (-)(Sub-Sampled 5-min)
                            </button>
                          </div>
                        )}
                      </div>

                      {/* Median Realized Variance */}
                      <div
                        className="dropdown-d-group"
                        onMouseEnter={() => handleSubmenuEnter("median")}
                      >
                        <div className="group-header">
                          Median Realized Variance <span className="submenu-arrow">›</span>
                        </div>
                        
                        {activeSubmenu === "median" && (
                          <div className="submenu">
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("medrv1", "Median Realized Variance (1-min)");
                            }} className="submenu-item">
                              Median Realized Variance (1-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("medrv5", "Median Realized Variance (5-min)");
                            }} className="submenu-item">
                              Median Realized Variance (5-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("medrv5_ss", "Median Realized Variance (Sub-Sampled 5-min)");
                            }} className="submenu-item">
                              Median Realized Variance (Sub-Sampled 5-min)
                            </button>
                          </div>
                        )}
                      </div>

                      {/* Minimum Realized Variance */}
                      <div
                        className="dropdown-d-group"
                        onMouseEnter={() => handleSubmenuEnter("minimum")}
                      >
                        <div className="group-header">
                          Minimum Realized Variance <span className="submenu-arrow">›</span>
                        </div>
                        
                        {activeSubmenu === "minimum" && (
                          <div className="submenu">
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("minrv1", "Minimum Realized Variance (1-min)");
                            }} className="submenu-item">
                              Minimum Realized Variance (1-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("minrv5", "Minimum Realized Variance (5-min)");
                            }} className="submenu-item">
                              Minimum Realized Variance (5-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("minrv5_ss", "Minimum Realized Variance (Sub-Sampled 5-min)");
                            }} className="submenu-item">
                              Minimum Realized Variance (Sub-Sampled 5-min)
                            </button>
                          </div>
                        )}
                      </div>

                      {/* Realized Quarticity */}
                      <div
                        className="dropdown-d-group"
                        onMouseEnter={() => handleSubmenuEnter("quarticity")}
                      >
                        <div className="group-header">
                          Realized Quarticity <span className="submenu-arrow">›</span>
                        </div>
                        
                        {activeSubmenu === "quarticity" && (
                          <div className="submenu">
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rq1", "Realized Quarticity (1-min)");
                            }} className="submenu-item">
                              Realized Quarticity (1-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rq5", "Realized Quarticity (5-min)");
                            }} className="submenu-item">
                              Realized Quarticity (5-min)
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rq5_ss", "Realized Quarticity (Sub-Sampled 5-min)");
                            }} className="submenu-item">
                              Realized Quarticity (Sub-Sampled 5-min)
                            </button>
                          </div>
                        )}
                      </div>

                      {/* Realized Kernel */}
                      <div className="dropdown-d-group">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleMeasureSelect("rk", "Realized Kernel (Parzen)");
                          }}
                          className="group-header-button"
                        >
                          Realized Kernel (Parzen)
                        </button>
                      </div>

                      {/* Other */}
                      <div
                        className="dropdown-d-group"
                        onMouseEnter={() => handleSubmenuEnter("other")}
                      >
                        <div className="group-header">
                          Other <span className="submenu-arrow">›</span>
                        </div>
                        
                        {activeSubmenu === "other" && (
                          <div className="submenu">
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("pv", "Parkinson's Range");
                            }} className="submenu-item">
                              Parkinson's Range
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("gk", "Garman-Klass Range");
                            }} className="submenu-item">
                              Garman-Klass Range
                            </button>
                            <button onClick={(e) => {
                              e.stopPropagation();
                              handleMeasureSelect("rr5", "Realized Range");
                            }} className="submenu-item">
                              Realized Range
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Action Button */}
          <div className="action-section">
            <button
              onClick={handleGo}
              disabled={!isFormValid}
              className={`analyze-button ${isFormValid ? 'enabled' : 'disabled'}`}
            >
              {isFormValid ? (isUpdate ? 'Update' : 'GO') : 'Complete All Fields'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;