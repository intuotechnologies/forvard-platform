// src/pages/AdminUserManagement.jsx

import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import adminService from '../services/adminService';
import { Users, Crown, User, Shield, ExternalLink, AlertCircle } from 'lucide-react';

const AdminUserManagement = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [updating, setUpdating] = useState({});
  const { userType, isAuthenticated } = useAuth();

  // IMPORTANTE: useEffect DEVE essere chiamato prima di qualsiasi return condizionale
  useEffect(() => {
    // Solo carica i dati se l'utente è admin
    if (isAuthenticated && userType === 'admin') {
      loadUsers();
    } else {
      setLoading(false);
    }
  }, [isAuthenticated, userType]);

  const loadUsers = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const userData = await adminService.getAllUsers();
      setUsers(userData);
      console.log('Loaded ForVARD users:', userData);
    } catch (error) {
      console.error('Failed to load ForVARD users:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Ora i return condizionali vengono DOPO gli hooks
  
  // Solo admin può accedere a questa pagina
  if (!isAuthenticated || userType !== 'admin') {
    return (
      <div style={{ 
        textAlign: 'center', 
        padding: '50px',
        color: 'red' 
      }}>
        <h2>Access Denied</h2>
        <p>Only administrators can access user management.</p>
      </div>
    );
  }

  const updateUserRole = async (userId, newRole) => {
    setUpdating(prev => ({ ...prev, [userId]: true }));

    try {
      await adminService.updateUserRole(userId, newRole);

      // Aggiorna lo stato locale
      setUsers(prevUsers => 
        prevUsers.map(user => 
          user.id === userId 
            ? { ...user, role: newRole }
            : user
        )
      );

      console.log(`Updated ForVARD user ${userId} to ${newRole}`);
      
    } catch (error) {
      console.error('Failed to update ForVARD user role:', error);
      alert('Failed to update user: ' + error.message);
    } finally {
      setUpdating(prev => ({ ...prev, [userId]: false }));
    }
  };

  const getUserIcon = (role) => {
    switch (role) {
      case 'admin':
        return <Shield size={16} color="#dc3545" />;
      case 'senator':
        return <Crown size={16} color="#ffc107" />;
      default:
        return <User size={16} color="#6c757d" />;
    }
  };

  const getUserBadgeColor = (role) => {
    switch (role) {
      case 'admin':
        return '#dc3545';
      case 'senator':
        return '#ffc107';
      default:
        return '#6c757d';
    }
  };

  if (loading) {
    return (
      <div style={{ 
        textAlign: 'center', 
        padding: '50px' 
      }}>
        <div>Loading ForVARD users...</div>
      </div>
    );
  }

  // Se c'è un errore, mostra info sull'admin panel
  if (error) {
    return (
      <div style={{ 
        maxWidth: '800px', 
        margin: '0 auto', 
        padding: '20px' 
      }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          marginBottom: '30px' 
        }}>
          <Users size={24} style={{ marginRight: '10px' }} />
          <h1>ForVARD User Management</h1>
        </div>

        <div style={{
          background: '#fff3cd',
          padding: '20px',
          borderRadius: '8px',
          border: '1px solid #ffeaa7',
          marginBottom: '20px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '15px' }}>
            <AlertCircle size={20} color="#856404" style={{ marginRight: '10px' }} />
            <h3 style={{ margin: '0', color: '#856404' }}>API Endpoint Not Available</h3>
          </div>
          <p style={{ marginBottom: '15px' }}>
            The user management API endpoints are not yet implemented in the ForVARD backend. 
            However, you can manage users through the built-in admin panel.
          </p>
          <button
            onClick={adminService.redirectToAdminPanel}
            style={{
              display: 'flex',
              alignItems: 'center',
              padding: '12px 20px',
              background: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '16px',
              textDecoration: 'none'
            }}
          >
            <ExternalLink size={16} style={{ marginRight: '8px' }} />
            Open ForVARD Admin Panel
          </button>
        </div>

        <div style={{
          background: '#f8f9fa',
          padding: '20px',
          borderRadius: '8px',
          border: '1px solid #dee2e6'
        }}>
          <h3>ForVARD Admin Panel Features:</h3>
          <ul style={{ marginBottom: '15px' }}>
            <li><strong>User Management:</strong> View, edit, and manage user accounts</li>
            <li><strong>Role Management:</strong> Assign Base, Senator, or Admin roles</li>
            <li><strong>Access Control:</strong> Configure role-based permissions</li>
            <li><strong>Financial Data Management:</strong> Manage volatility data</li>
          </ul>
          
          <div style={{
            background: '#e7f3ff',
            padding: '15px',
            borderRadius: '6px',
            border: '1px solid #b8e0ff'
          }}>
            <strong>Default Admin Credentials:</strong><br/>
            Email: admin@example.com<br/>
            Password: adminpass
          </div>
        </div>

        <div style={{
          marginTop: '20px',
          padding: '15px',
          background: '#f8f9fa',
          borderRadius: '8px',
          border: '1px solid #dee2e6'
        }}>
          <h4>📋 For Developers:</h4>
          <p style={{ marginBottom: '10px' }}>
            To enable user management via API, implement these endpoints in the ForVARD backend:
          </p>
          <ul style={{ 
            fontFamily: 'monospace', 
            fontSize: '14px',
            background: '#fff',
            padding: '10px',
            borderRadius: '4px',
            marginBottom: '0'
          }}>
            <li>GET /admin/users - List all users</li>
            <li>PUT /admin/users/{"{user_id}"}/role - Update user role</li>
            <li>GET /admin/stats - Get admin statistics</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div style={{ 
      maxWidth: '1200px', 
      margin: '0 auto', 
      padding: '20px' 
    }}>
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        marginBottom: '30px' 
      }}>
        <Users size={24} style={{ marginRight: '10px' }} />
        <h1>ForVARD User Management</h1>
      </div>

      <div style={{
        background: '#f8f9fa',
        padding: '15px',
        borderRadius: '8px',
        marginBottom: '20px',
        border: '1px solid #dee2e6'
      }}>
        <h3>Manage User Access Levels</h3>
        <p>
          <strong>Base:</strong> Limited access to basic financial data<br/>
          <strong>Senator:</strong> Full access to premium financial data and analytics<br/>
          <strong>Admin:</strong> Full administrative access (use carefully!)
        </p>
      </div>

      <div style={{
        background: 'white',
        borderRadius: '8px',
        overflow: 'hidden',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <table style={{ 
          width: '100%', 
          borderCollapse: 'collapse' 
        }}>
          <thead style={{ 
            background: '#f8f9fa',
            borderBottom: '2px solid #dee2e6'
          }}>
            <tr>
              <th style={{ padding: '15px', textAlign: 'left' }}>User</th>
              <th style={{ padding: '15px', textAlign: 'left' }}>Email</th>
              <th style={{ padding: '15px', textAlign: 'center' }}>Current Role</th>
              <th style={{ padding: '15px', textAlign: 'center' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {users.map(user => (
              <tr key={user.id} style={{ 
                borderBottom: '1px solid #dee2e6'
              }}>
                <td style={{ padding: '15px' }}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    {getUserIcon(user.role)}
                    <span style={{ marginLeft: '8px', fontWeight: '500' }}>
                      {user.full_name || user.email}
                    </span>
                  </div>
                </td>
                <td style={{ padding: '15px', color: '#666' }}>
                  {user.email}
                </td>
                <td style={{ padding: '15px', textAlign: 'center' }}>
                  <span style={{
                    background: getUserBadgeColor(user.role),
                    color: 'white',
                    padding: '4px 12px',
                    borderRadius: '12px',
                    fontSize: '12px',
                    fontWeight: 'bold',
                    textTransform: 'uppercase'
                  }}>
                    {user.role}
                  </span>
                </td>
                <td style={{ padding: '15px', textAlign: 'center' }}>
                  <div style={{ display: 'flex', gap: '8px', justifyContent: 'center' }}>
                    {user.role !== 'base' && (
                      <button
                        onClick={() => updateUserRole(user.id, 'base')}
                        disabled={updating[user.id]}
                        style={{
                          padding: '6px 12px',
                          background: '#6c757d',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '12px'
                        }}
                      >
                        {updating[user.id] ? '...' : 'Make Base'}
                      </button>
                    )}
                    
                    {user.role !== 'senator' && (
                      <button
                        onClick={() => updateUserRole(user.id, 'senator')}
                        disabled={updating[user.id]}
                        style={{
                          padding: '6px 12px',
                          background: '#ffc107',
                          color: 'black',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '12px'
                        }}
                      >
                        {updating[user.id] ? '...' : 'Make Senator'}
                      </button>
                    )}
                    
                    {user.role !== 'admin' && (
                      <button
                        onClick={() => {
                          if (window.confirm('Are you sure you want to make this user an admin? This gives them full access.')) {
                            updateUserRole(user.id, 'admin');
                          }
                        }}
                        disabled={updating[user.id]}
                        style={{
                          padding: '6px 12px',
                          background: '#dc3545',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '12px'
                        }}
                      >
                        {updating[user.id] ? '...' : 'Make Admin'}
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {users.length === 0 && (
          <div style={{ 
            textAlign: 'center', 
            padding: '40px',
            color: '#666'
          }}>
            No users found.
          </div>
        )}
      </div>

      <div style={{
        marginTop: '20px',
        padding: '15px',
        background: '#fff3cd',
        borderRadius: '8px',
        border: '1px solid #ffeaa7'
      }}>
        <h4>Instructions:</h4>
        <ul style={{ marginBottom: '0' }}>
          <li>New users are automatically assigned <strong>Base</strong> access</li>
          <li>Click <strong>Make Senator</strong> to give premium access to trusted users</li>
          <li>Only make users <strong>Admin</strong> if they need to manage other users</li>
          <li>Changes take effect immediately</li>
        </ul>
      </div>
    </div>
  );
};

export default AdminUserManagement;