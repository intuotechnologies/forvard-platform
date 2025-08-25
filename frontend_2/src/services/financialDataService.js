// src/services/financialDataService.js - conforme a OpenAPI
import { API_CONFIG } from '../config/api.js';

class FinancialDataService {
  constructor() {
    this.baseURL = API_CONFIG.BASE_URL;
    this.headers = API_CONFIG.DEFAULT_HEADERS;
  }

  // Helper per aggiungere token alle richieste
  getAuthHeaders(token) {
    if (!token) {
      token = localStorage.getItem('forvard_token');
    }
    
    if (!token) {
      throw new Error('No authentication token available');
    }

    return {
      ...this.headers,
      'Authorization': `Bearer ${token}`
    };
  }

  // Helper per gestire errori di autenticazione
  handleAuthError(error, context = '') {
    console.error(`Authentication error in ${context}:`, error);
    
    // Se è un errore 401, forza logout
    if (error.message.includes('Session expired') || error.message.includes('401')) {
      console.log('Session expired - forcing logout...');
      
      // Rimuovi token dal localStorage
      localStorage.removeItem('forvard_token');
      
      // Trigger evento personalizzato per notificare AuthContext
      window.dispatchEvent(new CustomEvent('forceLogout', { 
        detail: { reason: 'Session expired' }
      }));
      
      // Redirect immediato al login
      if (typeof window !== 'undefined') {
        window.location.href = '/login';
      }
    }
    
    throw error;
  }
  buildQueryParams(params) {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== '') {
        if (Array.isArray(value)) {
          // Per array secondo OpenAPI spec (symbols, fields)
          value.forEach(item => searchParams.append(key, item));
        } else {
          searchParams.append(key, value);
        }
      }
    });
    
    return searchParams.toString();
  }

  // Valida response secondo OpenAPI schema
  validateFinancialDataResponse(response) {
    if (!response) {
      throw new Error('Empty response from API');
    }

    // Schema OpenAPI: FinancialDataResponse
    if (!response.data || !Array.isArray(response.data)) {
      throw new Error('Invalid response format: missing or invalid data array');
    }

    // Campi obbligatori secondo OpenAPI
    const requiredFields = ['total', 'page', 'limit', 'has_more'];
    for (const field of requiredFields) {
      if (response[field] === undefined || response[field] === null) {
        console.warn(`Missing ${field} in API response`);
      }
    }

    return response;
  }

  // Valida response covariance secondo OpenAPI schema  
  validateCovarianceDataResponse(response) {
    if (!response) {
      throw new Error('Empty covariance response from API');
    }

    // Schema OpenAPI: CovarianceDataResponse
    if (!response.data || !Array.isArray(response.data)) {
      throw new Error('Invalid covariance response format: missing or invalid data array');
    }

    return response;
  }

  // GET /financial-data (conforme a OpenAPI)
  async getFinancialData(token, filters = {}) {
    try {
      const queryString = this.buildQueryParams(filters);
      const url = `${this.baseURL}${API_CONFIG.ENDPOINTS.FINANCIAL_DATA}${queryString ? `?${queryString}` : ''}`;
      
      console.log('GET /financial-data:', url);
      console.log('Filters:', filters);

      const response = await fetch(url, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });

      if (!response.ok) {
        if (response.status === 401) {
          const error = new Error('Session expired. Please login again.');
          this.handleAuthError(error, 'getFinancialData');
          return; // Non raggiungerà mai questo punto
        } else if (response.status === 403) {
          throw new Error('Forbidden - Access limit exceeded or insufficient permissions');
        } else if (response.status === 404) {
          throw new Error('Financial data not found');
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Valida response secondo OpenAPI schema
      this.validateFinancialDataResponse(result);
      
      console.log(`Received ${result.data.length} financial data points (total: ${result.total}, page: ${result.page})`);
      
      return result;
    } catch (error) {
      console.error('Get financial data error:', error);
      throw error;
    }
  }

  // GET /financial-data/covariance (secondo OpenAPI)
  async getCovarianceData(token, filters = {}) {
    try {
      // Parametri specifici per covariance secondo OpenAPI
      const validParams = {};
      
      // Parametri supportati dall'API covariance
      if (filters.asset1_symbol) validParams.asset1_symbol = filters.asset1_symbol;
      if (filters.asset2_symbol) validParams.asset2_symbol = filters.asset2_symbol;
      if (filters.symbols) validParams.symbols = filters.symbols;
      if (filters.start_date) validParams.start_date = filters.start_date;
      if (filters.end_date) validParams.end_date = filters.end_date;
      if (filters.page) validParams.page = filters.page;
      if (filters.limit) validParams.limit = filters.limit;
      if (filters.fields) validParams.fields = filters.fields;

      const queryString = this.buildQueryParams(validParams);
      const url = `${this.baseURL}${API_CONFIG.ENDPOINTS.COVARIANCE}${queryString ? `?${queryString}` : ''}`;
      
      console.log('GET /financial-data/covariance:', url);
      console.log('Covariance filters:', validParams);

      const response = await fetch(url, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });

      if (!response.ok) {
        if (response.status === 401) {
          const error = new Error('Session expired. Please login again.');
          this.handleAuthError(error, 'getCovarianceData');
          return;
        } else if (response.status === 403) {
          throw new Error('Forbidden - Access limit exceeded or insufficient permissions');
        } else if (response.status === 404) {
          throw new Error('Covariance data not found');
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Valida response covariance
      this.validateCovarianceDataResponse(result);
      
      console.log(`Received ${result.data.length} covariance data points`);
      
      return result;
    } catch (error) {
      console.error('Get covariance data error:', error);
      throw error;
    }
  }

  // GET /financial-data/limits (già corretto)
  async getAccessLimits(token) {
    try {
      console.log('Getting access limits...');
      
      const response = await fetch(`${this.baseURL}${API_CONFIG.ENDPOINTS.LIMITS}`, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });

      if (!response.ok) {
        if (response.status === 401) {
          const error = new Error('Session expired. Please login again.');
          this.handleAuthError(error, 'getAccessLimits');
          return;
        } else if (response.status === 403) {
          throw new Error('Forbidden - Admin access required');
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Access limits received:', data);
      return data;
    } catch (error) {
      console.error('Get access limits error:', error);
      throw error;
    }
  }

  // GET /financial-data/download (conforme a OpenAPI) - AGGIORNATO
  async downloadFinancialData(token, filters = {}) {
    try {
      // Parametri specifici per download secondo OpenAPI
      const validParams = {};
      
      // Parametri supportati dall'endpoint download
      if (filters.symbol) validParams.symbol = filters.symbol;
      if (filters.symbols) validParams.symbols = filters.symbols;
      if (filters.asset_type) validParams.asset_type = filters.asset_type;
      if (filters.start_date) validParams.start_date = filters.start_date;
      if (filters.end_date) validParams.end_date = filters.end_date;
      if (filters.fields) validParams.fields = filters.fields;

      const queryString = this.buildQueryParams(validParams);
      const url = `${this.baseURL}${API_CONFIG.ENDPOINTS.DOWNLOAD}${queryString ? `?${queryString}` : ''}`;
      
      console.log('GET /financial-data/download:', url);
      console.log('Download filters:', validParams);

      const response = await fetch(url, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });

      if (!response.ok) {
        if (response.status === 401) {
          const error = new Error('Session expired. Please login again.');
          this.handleAuthError(error, 'downloadFinancialData');
          return;
        } else if (response.status === 403) {
          throw new Error('Forbidden - Access limit exceeded');
        } else if (response.status === 404) {
          throw new Error('Financial data not found');
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Valida DownloadResponse secondo OpenAPI
      if (!result.download_url || !result.file_name) {
        throw new Error('Invalid download response: missing download_url or file_name');
      }
      
      console.log('Download prepared:', result.file_name);
      console.log('Fields requested:', validParams.fields);
      
      return result;
    } catch (error) {
      console.error('Download financial data error:', error);
      throw error;
    }
  }

  // GET /financial-data/covariance/download (secondo OpenAPI) - AGGIORNATO
  async downloadCovarianceData(token, filters = {}) {
    try {
      // Parametri specifici per covariance download
      const validParams = {};
      
      if (filters.asset1_symbol) validParams.asset1_symbol = filters.asset1_symbol;
      if (filters.asset2_symbol) validParams.asset2_symbol = filters.asset2_symbol;
      if (filters.symbols) validParams.symbols = filters.symbols;
      if (filters.start_date) validParams.start_date = filters.start_date;
      if (filters.end_date) validParams.end_date = filters.end_date;
      if (filters.fields) validParams.fields = filters.fields;

      const queryString = this.buildQueryParams(validParams);
      const url = `${this.baseURL}${API_CONFIG.ENDPOINTS.COVARIANCE_DOWNLOAD}${queryString ? `?${queryString}` : ''}`;
      
      console.log('GET /financial-data/covariance/download:', url);
      console.log('Covariance download filters:', validParams);

      const response = await fetch(url, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });

      if (!response.ok) {
        if (response.status === 401) {
          const error = new Error('Session expired. Please login again.');
          this.handleAuthError(error, 'downloadCovarianceData');
          return;
        } else if (response.status === 403) {
          throw new Error('Forbidden - Access limit exceeded');
        } else if (response.status === 404) {
          throw new Error('Covariance data not found');
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Valida DownloadResponse
      if (!result.download_url || !result.file_name) {
        throw new Error('Invalid covariance download response');
      }
      
      console.log('Covariance download prepared:', result.file_name);
      
      return result;
    } catch (error) {
      console.error('Download covariance data error:', error);
      throw error;
    }
  }

  // AGGIORNATO: downloadDataset con logica semplificata
  async downloadDataset(token, filters = {}) {
    console.log('Downloading dataset with filters:', filters);
    
    // Determina quale endpoint usare in base ai parametri
    if (filters.endpoint === '/financial-data/covariance/download' || 
        filters.include_covariances || 
        filters.asset1_symbol || 
        filters.asset2_symbol) {
      
      console.log('Using covariance download endpoint');
      return this.downloadCovarianceData(token, filters);
    } else {
      console.log('Using financial data download endpoint');
      
      // Converte data_types in fields per il nuovo endpoint
      if (filters.data_types && !filters.fields) {
        filters.fields = [
          'observation_date', 'symbol', 'asset_type',
          ...filters.data_types
        ];
        console.log('Converted data_types to fields:', filters.fields);
      }
      
      return this.downloadFinancialData(token, filters);
    }
  }

  // GET /financial-data/files/{filename} (secondo OpenAPI)
  async getDownloadFile(token, filename) {
    try {
      console.log('Getting download file:', filename);
      
      const response = await fetch(`${this.baseURL}${API_CONFIG.ENDPOINTS.FILES}/${filename}`, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });

      if (!response.ok) {
        if (response.status === 401) {
          const error = new Error('Session expired. Please login again.');
          this.handleAuthError(error, 'getDownloadFile');
          return;
        } else if (response.status === 403) {
          throw new Error('Forbidden - Access limit exceeded');
        } else if (response.status === 404) {
          throw new Error('File not found');
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      console.log('Download file retrieved');
      
      // Restituisce il response per gestire il blob
      return response;
    } catch (error) {
      console.error('Get download file error:', error);
      throw error;
    }
  }

  // Metodi di convenienza per query comuni
  
  // Cerca dati per simbolo specifico
  async searchBySymbol(token, symbol, dateRange = {}) {
    return this.getFinancialData(token, {
      symbol,
      ...dateRange
    });
  }

  // Cerca dati per multipli simboli  
  async searchBySymbols(token, symbols, dateRange = {}) {
    return this.getFinancialData(token, {
      symbols,
      ...dateRange
    });
  }

  // Cerca dati per tipo di asset
  async searchByAssetType(token, assetType, dateRange = {}) {
    return this.getFinancialData(token, {
      asset_type: assetType,
      ...dateRange
    });
  }

  // Covariance tra due asset specifici
  async getCovarianceBetween(token, asset1, asset2, dateRange = {}) {
    return this.getCovarianceData(token, {
      asset1_symbol: asset1,
      asset2_symbol: asset2,
      ...dateRange
    });
  }

  // MANTIENI: Metodi esistenti per backward compatibility
  async getFilteredData(token, filters = {}) {
    console.log('[LEGACY] Getting filtered data with filters:', filters);
    return this.getFinancialData(token, filters);
  }

  async downloadFileAsBlob(token, downloadUrl) {
    try {
      const fullUrl = downloadUrl.startsWith('http') 
        ? downloadUrl 
        : `${this.baseURL}${downloadUrl}`;
        
      console.log('Downloading file as blob:', fullUrl);
      
      const response = await fetch(fullUrl, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response.blob();
    } catch (error) {
      console.error('Download file as blob error:', error);
      throw error;
    }
  }

  // Test connessione
  async testAuthenticatedConnection(token) {
    try {
      const response = await fetch(`${this.baseURL}${API_CONFIG.ENDPOINTS.HEALTH}`, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });
      
      return response.ok;
    } catch (error) {
      console.error('Test authenticated connection error:', error);
      return false;
    }
  }
}

// Esporta un'istanza singleton
export const financialDataService = new FinancialDataService();
export default financialDataService;