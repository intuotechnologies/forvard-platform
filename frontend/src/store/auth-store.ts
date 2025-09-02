import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import api from '@/lib/axios';
import { jwtDecode } from 'jwt-decode';

interface User {
  user_id: string;
  email: string;
  role_name: string;
  created_at?: string;
}

interface TokenData {
  sub: string;
  role: string;
  exp: number;
}

interface AuthState {
  token: string | null;
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  getUserInfo: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        
        try {
          // Format data for FastAPI OAuth2 endpoint
          const formData = new URLSearchParams();
          formData.append('username', email); // OAuth2 uses username field
          formData.append('password', password);
          
          const response = await api.post('/auth/token', formData.toString(), {
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
            },
          });
          
          const { access_token } = response.data;
          
          // Store token
          localStorage.setItem('token', access_token);
          
          // Decode token to get user info
          const decoded = jwtDecode<TokenData>(access_token);
          
          set({
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
          });
          
          // Get additional user info
          await get().getUserInfo();
        } catch (error: any) {
          set({
            isLoading: false,
            error: error.response?.data?.detail || 'Authentication failed',
          });
        }
      },

      logout: () => {
        localStorage.removeItem('token');
        set({
          token: null,
          user: null,
          isAuthenticated: false,
          error: null,
        });
      },

      getUserInfo: async () => {
        try {
          const response = await api.get('/auth/me');
          set({
            user: response.data,
          });
        } catch (error) {
          console.error('Failed to get user info', error);
          // If this fails, we should log the user out
          get().logout();
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ token: state.token, user: state.user, isAuthenticated: state.isAuthenticated }),
    }
  )
); 