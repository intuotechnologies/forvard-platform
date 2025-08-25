// src/config/api.js - VERSIONE FINALE CORRETTA
export const API_CONFIG = {
  // BASE_URL: Usa HTTP per evitare errori SSL
  BASE_URL: 'http://volare.unime.it:8443',
  
  ENDPOINTS: {
    // Authentication
    LOGIN: '/auth/token',
    REGISTER: '/auth/register', 
    ME: '/auth/me',
    
    // Financial Data
    FINANCIAL_DATA: '/financial-data',
    LIMITS: '/financial-data/limits',
    DOWNLOAD: '/financial-data/download',
    FILES: '/financial-data/files',
    
    // Covariance Data
    COVARIANCE: '/financial-data/covariance',
    COVARIANCE_DOWNLOAD: '/financial-data/covariance/download',
    
    // Admin
    ADMIN_USERS: '/api/admin/users',
    ADMIN_ROLES: '/api/admin/roles', 
    ADMIN_STATS: '/api/admin/stats',
    ADMIN_PERMISSIONS: '/api/admin/permissions',
    
    // System
    HEALTH: '/health',
  },
  
  // Headers di default per requests normali
  DEFAULT_HEADERS: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },

  // Headers specifici per il login (form-urlencoded)
  LOGIN_HEADERS: {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json',
  }
};

// Utility per creare headers con autenticazione Bearer
export const createAuthHeaders = (token) => ({
  ...API_CONFIG.DEFAULT_HEADERS,
  'Authorization': `Bearer ${token}`
});

// Utility per creare il corpo della richiesta di login
export const createLoginBody = (email, password) => {
  const formData = new URLSearchParams();
  formData.append('username', email);  // L'API usa 'username' ma il valore è l'email
  formData.append('password', password);
  return formData.toString();
};