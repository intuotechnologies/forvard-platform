// src/services/adminService.js
import { API_CONFIG } from '../config/api.js';

class AdminService {
  constructor() {
    this.baseURL = API_CONFIG.BASE_URL;
    this.headers = API_CONFIG.DEFAULT_HEADERS;
  }

  // Helper per aggiungere token alle richieste
  getAuthHeaders(token) {
    return {
      ...this.headers,
      'Authorization': `Bearer ${token}`
    };
  }

  // Ottieni tutti gli utenti (admin only)
  async getAllUsers(token) {
    try {
      const response = await fetch(`${this.baseURL}/api/admin/users`, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get all users error:', error);
      throw error;
    }
  }

  // Ottieni tutti i ruoli (admin only)
  async getAllRoles(token) {
    try {
      const response = await fetch(`${this.baseURL}/api/admin/roles`, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get all roles error:', error);
      throw error;
    }
  }

  // Ottieni statistiche admin (admin only)
  async getAdminStats(token) {
    try {
      const response = await fetch(`${this.baseURL}/api/admin/stats`, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get admin stats error:', error);
      throw error;
    }
  }

  // Ottieni permessi di un utente specifico (admin only)
  async getUserPermissions(token, userId) {
    try {
      const response = await fetch(`${this.baseURL}/api/admin/permissions/${userId}`, {
        method: 'GET',
        headers: this.getAuthHeaders(token)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get user permissions error:', error);
      throw error;
    }
  }
}

// Esporta un'istanza singleton
export const adminService = new AdminService();
export default adminService;