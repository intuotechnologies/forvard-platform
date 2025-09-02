// src/App.js

import React from "react";
import { BrowserRouter as Router, Route, Routes, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./contexts/AuthContext";

// Import delle tue pagine esistenti
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import Dashboard from "./pages/Dashboard";
import GraphPage from "./pages/GraphPage";
import DownloadPage from "./pages/DownloadPage";
import AdminUserManagement from "./pages/AdminUserManagement.jsx";
import DocumentationPage from './pages/DocumentationPage';
import AboutPage from './pages/AboutPage';


// Import dei tuoi componenti esistenti
import Footer from "./components/Footer";
import Navbar from "./components/Navbar";

import "./styles/global.css";

// Componente per proteggere le route che richiedono autenticazione
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();
  
  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '50vh',
        fontSize: '18px',
        color: '#666'
      }}>
        <div>
          <div style={{ marginBottom: '10px' }}>🔄 Loading...</div>
          <div style={{ fontSize: '14px' }}>Checking authentication</div>
        </div>
      </div>
    );
  }
  
  return isAuthenticated ? children : <Navigate to="/login" replace />;
};

// Componente per route pubbliche (redirect se già autenticato)
const PublicOnlyRoute = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();
  
  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '50vh',
        fontSize: '18px',
        color: '#666'
      }}>
        <div>
          <div style={{ marginBottom: '10px' }}>🔄 Loading...</div>
          <div style={{ fontSize: '14px' }}>Checking authentication</div>
        </div>
      </div>
    );
  }
  
  return !isAuthenticated ? children : <Navigate to="/dashboard" replace />;
};

// Componente per route che possono essere accessibili sia autenticati che non
const PublicRoute = ({ children }) => {
  return children;
};

// Componente principale delle routes (deve essere all'interno di AuthProvider)
const AppRoutes = () => {
  return (
    <Router>
      <Navbar />
      <main style={{ minHeight: 'calc(100vh - 140px)' }}> {/* Spazio per navbar e footer */}
        <Routes>
          {/* Route pubbliche - accessibili a tutti */}
          <Route 
            path="/" 
            element={
              <PublicRoute>
                <HomePage />
              </PublicRoute>
            } 
          />
          

          {/* Route della documentazione - accessibile a tutti */}
          <Route 
            path="/documentation" 
            element={
              <PublicRoute>
                <DocumentationPage />
              </PublicRoute>
            } 
          />
          <Route 
            path="/about" 
            element={
              <PublicRoute>
                <AboutPage />
              </PublicRoute>
            } 
          />

          {/* Route solo per utenti NON autenticati */}
          <Route 
            path="/login" 
            element={
              <PublicOnlyRoute>
                <LoginPage />
              </PublicOnlyRoute>
            } 
          />
          

          {/* Route protette - solo per utenti autenticati */}
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } 
          />
          
          <Route 
            path="/graph" 
            element={
              <ProtectedRoute>
                <GraphPage />
              </ProtectedRoute>
            } 
          />
          
          <Route 
            path="/download" 
            element={
              <ProtectedRoute>
                <DownloadPage />
              </ProtectedRoute>
            } 
          />
          
          
          {/* Route solo per Admin - TEMPORANEAMENTE SENZA PROTEZIONE PER TEST */}
          <Route 
            path="/admin/users" 
            element={<AdminUserManagement />}
          />

          {/* Route di fallback - redirect alla home */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <Footer />
    </Router>
  );
};

// App principale con AuthProvider
function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}

export default App;