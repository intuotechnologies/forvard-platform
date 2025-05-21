import axios from 'axios';

// Determine baseURL based on environment
const getBaseUrl = () => {
  if (typeof window !== 'undefined') {
    // Client-side: route through Next.js API proxy
    return '/api';
  }
  // Server-side (though this instance is primarily for client, good to be explicit)
  // Or if direct backend calls are ever made from server-side Next.js contexts using this instance
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8443';
};

const api = axios.create({
  baseURL: getBaseUrl(),
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor to include the auth token in requests
api.interceptors.request.use(
  (config) => {
    // Get token from local storage only on client-side
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    // If server-side and need auth, it must be handled differently (e.g., passed in)
    return config;
  },
  (error) => Promise.reject(error)
);

export default api; 