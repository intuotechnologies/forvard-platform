// src/contexts/AuthContext.jsx
import React, { createContext, useContext, useState, useEffect } from 'react';
import { API_CONFIG, createAuthHeaders, createLoginBody } from '../config/api.js';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('forvard_token'));
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Debug: Log del token ogni volta che cambia
  useEffect(() => {
    console.log('Token state changed:', token ? `${token.substring(0, 20)}...` : 'null');
  }, [token]);

  // Debug: Log dello user ogni volta che cambia
  useEffect(() => {
    console.log('User state changed:', user);
  }, [user]);

  // Listener per logout forzato dal service
  useEffect(() => {
    const handleForceLogout = (event) => {
      console.log('🚪 Force logout triggered:', event.detail?.reason);
      
      // Esegui logout senza chiamare API (dato che la sessione è già scaduta)
      setUser(null);
      setToken(null);
      localStorage.removeItem('forvard_token');
      localStorage.removeItem('forvard_user');
      setError(null);
      
      // Opzionale: mostra messaggio
      if (event.detail?.reason) {
        console.log(`Logged out: ${event.detail.reason}`);
      }
    };

    // Aggiungi listener
    window.addEventListener('forceLogout', handleForceLogout);

    // Cleanup
    return () => {
      window.removeEventListener('forceLogout', handleForceLogout);
    };
  }, []);

  // Inizializzazione
  useEffect(() => {
    const initAuth = async () => {
      const savedToken = localStorage.getItem('forvard_token');
      console.log('Initializing auth, saved token:', savedToken ? `${savedToken.substring(0, 20)}...` : 'null');
      
      if (savedToken) {
        try {
          setToken(savedToken);
          await getCurrentUser(savedToken);
        } catch (error) {
          console.error('Token validation failed:', error);
          localStorage.removeItem('forvard_token');
          setToken(null);
          setUser(null);
        }
      }
      
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const testConnection = async () => {
    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.HEALTH}`, {
        method: 'GET',
        headers: API_CONFIG.DEFAULT_HEADERS,
      });
      
      if (response.ok) {
        console.log('Server connection OK');
        return true;
      } else {
        console.log('Server responded with error:', response.status);
        return false;
      }
    } catch (error) {
      console.error('Connection test failed:', error);
      try {
        const httpUrl = API_CONFIG.BASE_URL.replace('https://', 'http://');
        const response = await fetch(`${httpUrl}${API_CONFIG.ENDPOINTS.HEALTH}`);
        if (response.ok) {
          console.log('HTTP connection OK, consider updating BASE_URL');
          return true;
        }
      } catch (httpError) {
        console.error('HTTP connection also failed:', httpError);
      }
      return false;
    }
  };

  const getCurrentUser = async (authToken = token) => {
    try {
      console.log('Getting current user with token:', authToken ? `${authToken.substring(0, 20)}...` : 'null');
      
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.ME}`, {
        method: 'GET',
        headers: createAuthHeaders(authToken),
      });

      console.log('Get user response status:', response.status);

      if (!response.ok) {
        // Se 401, la sessione è scaduta - forza logout
        if (response.status === 401) {
          console.log('🚪 Session expired during getCurrentUser - forcing logout');
          window.dispatchEvent(new CustomEvent('forceLogout', { 
            detail: { reason: 'Session expired' }
          }));
          return;
        }
        
        const errorText = await response.text();
        console.error('Get user error response:', errorText);
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const userData = await response.json();
      console.log('User data received:', userData);
      setUser(userData);
      return userData;
    } catch (error) {
      console.error('Get current user error:', error);
      throw new Error(`Failed to get user data: ${error.message}`);
    }
  };

  const login = async (credentials) => {
    setError(null);
    setIsLoading(true);

    try {
      const isConnected = await testConnection();
      if (!isConnected) {
        throw new Error('Cannot connect to server. Please check your internet connection.');
      }

      console.log('Attempting login to:', `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.LOGIN}`);

      // CORREZIONE: Usa URLSearchParams invece di FormData per form-urlencoded
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.LOGIN}`, {
        method: 'POST',
        headers: API_CONFIG.LOGIN_HEADERS, // Headers specifici per login
        body: createLoginBody(credentials.email, credentials.password), // Funzione utility
      });

      console.log('Login response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Login error response:', errorText);
        
        let errorData;
        try {
          errorData = JSON.parse(errorText);
        } catch {
          errorData = { detail: `HTTP ${response.status}: ${response.statusText}` };
        }
        
        throw new Error(errorData.detail || 'Login failed');
      }

      const data = await response.json();
      console.log('Login successful');

      const { access_token } = data;
      
      // Salva il token PRIMA di chiamare getCurrentUser
      console.log('Saving token to localStorage');
      localStorage.setItem('forvard_token', access_token);
      setToken(access_token);

      // Aspetta un po' prima di chiamare getCurrentUser per assicurarsi che il token sia settato
      await new Promise(resolve => setTimeout(resolve, 100));

      // Ottieni i dati utente
      console.log('Getting user data after login');
      await getCurrentUser(access_token);

      return data;
    } catch (error) {
      console.error('Login error:', error);
      const errorMessage = error.message.includes('fetch') 
        ? 'Connection failed. Please check if the server is running and accessible.'
        : error.message;
      setError(errorMessage);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (userData) => {
    setError(null);
    setIsLoading(true);

    try {
      console.log('Attempting registration to:', `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.REGISTER}`);

      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.REGISTER}`, {
        method: 'POST',
        headers: API_CONFIG.DEFAULT_HEADERS,
        body: JSON.stringify({
          email: userData.email,
          password: userData.password,
          role_name: userData.role_name || 'base',
        }),
      });

      console.log('Registration response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Registration error response:', errorText);
        
        let errorData;
        try {
          errorData = JSON.parse(errorText);
        } catch {
          errorData = { detail: `HTTP ${response.status}: ${response.statusText}` };
        }
        
        throw new Error(errorData.detail || 'Registration failed');
      }

      const data = await response.json();
      console.log('Registration successful');
      return data;
    } catch (error) {
      console.error('Registration error:', error);
      const errorMessage = error.message.includes('fetch') 
        ? 'Connection failed. Please check if the server is running and accessible.'
        : error.message;
      setError(errorMessage);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    console.log('🚪 Manual logout');
    localStorage.removeItem('forvard_token');
    localStorage.removeItem('forvard_user');
    setToken(null);
    setUser(null);
    setError(null);
  };

  // Funzione per ottenere i limiti di accesso
  const getAccessLimits = async () => {
    const currentToken = token || localStorage.getItem('forvard_token');
    
    if (!currentToken) {
      throw new Error('No authentication token available');
    }

    try {
      console.log('Getting access limits with token:', `${currentToken.substring(0, 20)}...`);
      
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.LIMITS}`, {
        method: 'GET',
        headers: createAuthHeaders(currentToken),
      });

      console.log('Access limits response status:', response.status);

      if (!response.ok) {
        // Se 401, la sessione è scaduta - forza logout
        if (response.status === 401) {
          console.log('🚪 Session expired during getAccessLimits - forcing logout');
          window.dispatchEvent(new CustomEvent('forceLogout', { 
            detail: { reason: 'Session expired while getting access limits' }
          }));
          return;
        }
        
        const errorText = await response.text();
        console.error('Access limits error response:', errorText);
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const limitsData = await response.json();
      console.log('Access limits received:', limitsData);
      return limitsData;
    } catch (error) {
      console.error('Get access limits error:', error);
      throw error;
    }
  };

  // Funzione generica per API requests - AGGIORNATA con logout automatico
  const apiRequest = async (endpoint, options = {}) => {
    const currentToken = token || localStorage.getItem('forvard_token');
    const url = `${API_CONFIG.BASE_URL}${endpoint}`;
    
    const headers = {
      ...API_CONFIG.DEFAULT_HEADERS,
      ...options.headers,
    };

    if (currentToken) {
      headers.Authorization = `Bearer ${currentToken}`;
      console.log('Adding auth header to request:', endpoint);
    }

    try {
      console.log('API Request to:', url);
      
      const response = await fetch(url, {
        ...options,
        headers,
      });

      console.log('API Response status:', response.status);

      // Se il token è scaduto/non valido, logout automatico
      if (response.status === 401) {
        console.log('🚪 Session expired during apiRequest - forcing logout');
        window.dispatchEvent(new CustomEvent('forceLogout', { 
          detail: { reason: 'Session expired during API request' }
        }));
        throw new Error('Session expired. Please login again.');
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API request error:', error);
      throw error;
    }
  };

  const clearError = () => {
    setError(null);
  };

  const value = {
    user,
    token,
    isAuthenticated: !!token && !!user,
    isLoading,
    error,
    login,
    register,
    logout,
    getCurrentUser,
    getAccessLimits,
    apiRequest,
    clearError,
    testConnection,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};