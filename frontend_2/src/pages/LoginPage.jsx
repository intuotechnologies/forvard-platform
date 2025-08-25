// src/pages/LoginPage.jsx

import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import "../styles/login.css";

const LoginPage = () => {
  const [activeTab, setActiveTab] = useState('login');
  const [loginData, setLoginData] = useState({
    email: '',
    password: ''
  });
  const [registerData, setRegisterData] = useState({
    email: '',
    password: '',
    role_name: 'base'
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [localError, setLocalError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  
  const { login, register, isAuthenticated, isLoading, error, clearError } = useAuth();
  const navigate = useNavigate();

  // Se già autenticato, reindirizza alla dashboard
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  // Pulisci errori quando l'utente cambia tab o inizia a digitare
  useEffect(() => {
    setLocalError('');
    setSuccessMessage('');
    if (error) {
      clearError();
    }
  }, [activeTab, loginData, registerData, error, clearError]);

  const handleTabSwitch = (tab) => {
    setActiveTab(tab);
    setLocalError('');
    setSuccessMessage('');
    if (error) {
      clearError();
    }
  };

  const handleLoginChange = (e) => {
    const { name, value } = e.target;
    setLoginData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleRegisterChange = (e) => {
    const { name, value } = e.target;
    setRegisterData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    
    // Validazione base
    if (!loginData.email || !loginData.password) {
      setLocalError('Please fill in all fields');
      return;
    }

    setIsSubmitting(true);
    setLocalError('');

    try {
      await login(loginData);
      console.log('Login successful, redirecting...');
      navigate('/dashboard');
    } catch (error) {
      console.error('Login failed:', error);
      setLocalError(error.message || 'Login failed. Please check your credentials.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRegisterSubmit = async (e) => {
    e.preventDefault();
    
    // Validazione base
    if (!registerData.email || !registerData.password) {
      setLocalError('Please fill in all fields');
      return;
    }

    setIsSubmitting(true);
    setLocalError('');

    try {
      await register(registerData);
      setSuccessMessage('Registration successful! You can now sign in.');
      setActiveTab('login');
      // Pre-compila email nel form di login
      setLoginData(prev => ({ ...prev, email: registerData.email }));
      // Reset form registrazione
      setRegisterData({
        email: '',
        password: '',
        role_name: 'base'
      });
    } catch (error) {
      console.error('Registration failed:', error);
      setLocalError(error.message || 'Registration failed. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Mostra loading durante l'inizializzazione
  if (isLoading) {
    return (
      <div className="login-page">
        <div className="form-container">
          <div className="loading-message">
            Initializing ForVARD...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="login-page">
      <div className="form-container">
        <div className="logo-section">
          <h1>ForVARD</h1>
        </div>

        {/* Tab Navigation */}
        <div className="form-tabs">
          <button 
            className={`tab ${activeTab === 'login' ? 'active' : ''}`}
            onClick={() => handleTabSwitch('login')}
            disabled={isSubmitting}
          >
            Sign In
          </button>
          <button 
            className={`tab ${activeTab === 'register' ? 'active' : ''}`}
            onClick={() => handleTabSwitch('register')}
            disabled={isSubmitting}
          >
            Sign Up
          </button>
        </div>

        {/* Messaggi di errore/successo */}
        {(error || localError) && (
          <div className="error-message">
            {error || localError}
          </div>
        )}

        {successMessage && (
          <div className="success-message">
            {successMessage}
          </div>
        )}

        {/* Form di Login */}
        {activeTab === 'login' && (
          <form className="form" onSubmit={handleLoginSubmit}>
            <input 
              type="email" 
              name="email"
              placeholder="Email" 
              value={loginData.email}
              onChange={handleLoginChange}
              required 
              disabled={isSubmitting}
            />
            <input 
              type="password" 
              name="password"
              placeholder="Password" 
              value={loginData.password}
              onChange={handleLoginChange}
              required 
              disabled={isSubmitting}
            />

            <Link to="/reset-password" className="forgot-password">
              Forgot your password?
            </Link>

            <button 
              type="submit" 
              className="submit-btn"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'SIGNING IN...' : 'SIGN IN'}
            </button>
          </form>
        )}

        {/* Form di Registrazione */}
        {activeTab === 'register' && (
          <form className="form" onSubmit={handleRegisterSubmit}>
            <input 
              type="email" 
              name="email"
              placeholder="Email" 
              value={registerData.email}
              onChange={handleRegisterChange}
              required 
              disabled={isSubmitting}
            />
            <input 
              type="password" 
              name="password"
              placeholder="Password" 
              value={registerData.password}
              onChange={handleRegisterChange}
              required 
              disabled={isSubmitting}
              minLength="6"
            />
            <select
              name="role_name"
              value={registerData.role_name}
              onChange={handleRegisterChange}
              disabled={isSubmitting}
              className="role-select"
            >
              <option value="base">Base User</option>
              <option value="premium">Premium User</option>
              <option value="admin">Admin</option>
            </select>

            <button 
              type="submit" 
              className="submit-btn"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'CREATING ACCOUNT...' : 'CREATE ACCOUNT'}
            </button>
          </form>
        )}

        {activeTab === 'login' && (
          <p className="form-footer">
            Need an account? <button 
              className="link-button" 
              onClick={() => handleTabSwitch('register')}
              disabled={isSubmitting}
            >
              Sign Up
            </button>
          </p>
        )}

        {activeTab === 'register' && (
          <p className="form-footer">
            Already have an account? <button 
              className="link-button" 
              onClick={() => handleTabSwitch('login')}
              disabled={isSubmitting}
            >
              Sign In
            </button>
          </p>
        )}
      </div>
    </div>
  );
};

export default LoginPage;