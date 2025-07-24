import { create } from 'zustand';
import api from '@/lib/axios';
import { toast } from 'react-hot-toast';

// Define a simple logger
const logger = {
  error: (...args: any[]) => console.error(...args),
  // Add other methods like log, warn, info if needed
};

interface DataPoint {
  observation_date: string;
  symbol: string;
  asset_type: string;
  volume: number;
  trades: number;
  open_price: number;
  close_price: number;
  high_price: number;
  low_price: number;
  rv5: number;
  [key: string]: any; // For additional fields
}

interface DataResponse {
  data: DataPoint[];
  total_count: number;
  page: number;
  limit: number;
  effective_limit: number;
}

interface FilterOptions {
  symbol?: string;
  symbols?: string[];
  asset_type?: string;
  start_date?: string;
  end_date?: string;
  page?: number;
  limit?: number;
  fields?: string[];
}

interface DownloadResponse {
  download_url: string;
  file_name: string;
  expires_at: string;
}

interface FinancialDataState {
  data: DataPoint[];
  isLoading: boolean;
  isDownloading: boolean;
  error: string | null;
  totalCount: number;
  page: number;
  limit: number;
  effectiveLimit: number;
  fetchData: (options: FilterOptions) => Promise<void>;
  downloadData: (options: FilterOptions) => Promise<boolean>;
  accessLimits: Record<string, number> | null;
  fetchAccessLimits: () => Promise<void>;
}

export const useFinancialDataStore = create<FinancialDataState>((set, get) => ({
  data: [],
  isLoading: false,
  isDownloading: false,
  error: null,
  totalCount: 0,
  page: 1,
  limit: 100,
  effectiveLimit: 0,
  accessLimits: null,

  fetchData: async (options: FilterOptions) => {
    set({ isLoading: true, error: null });
    try {
      const fieldsToFetch = Array.from(new Set([
        ...(options.fields || []),
        'observation_date', 
        'symbol', 
        'asset_type' 
      ]));

      console.log('🔍 DEBUG: Fields being sent to API:', fieldsToFetch);
      console.log('🔍 DEBUG: Full options being sent:', { ...options, fields: fieldsToFetch });

      const response = await api.get('/financial-data', { params: { ...options, fields: fieldsToFetch } });
      const result: DataResponse = response.data;
      
      console.log('🔍 DEBUG: API Response received:', result);
      console.log('🔍 DEBUG: First data point:', result.data[0]);
      
      set({
        data: result.data.sort((a, b) => new Date(a.observation_date).getTime() - new Date(b.observation_date).getTime()),
        totalCount: result.total_count,
        page: result.page,
        limit: result.limit,
        effectiveLimit: result.effective_limit,
        isLoading: false,
      });
    } catch (error: any) {
      console.error('🔍 DEBUG: API Error:', error);
      set({ isLoading: false, error: error.response?.data?.detail || 'Failed to fetch data' });
    }
  },

  downloadData: async (options: FilterOptions) => {
    set({ isDownloading: true, error: null });
    const { useAuthStore } = await import('@/store/auth-store');
    const { token } = useAuthStore.getState();
    if (!token) {
      set({ error: 'User not authenticated', isDownloading: false });
      toast.error('Authentication required for download.');
      return false;
    }

    // Construct query parameters from filters
    const queryParams = new URLSearchParams();
    if (options.symbols && options.symbols.length > 0) {
      options.symbols.forEach(symbol => queryParams.append('symbols', symbol));
    }
    if (options.asset_type) {
      queryParams.append('asset_type', options.asset_type);
    }
    if (options.start_date) {
      queryParams.append('start_date', options.start_date);
    }
    if (options.end_date) {
      queryParams.append('end_date', options.end_date);
    }
    if (options.fields && options.fields.length > 0) {
        options.fields.forEach(field => queryParams.append('fields', field));
    }
    queryParams.append('format', 'csv'); // Ensure CSV format

    try {
      // Step 1: Request the file generation and get the download URL
      const prepareResponse = await api.get<DownloadResponse>(
        `/financial-data/download?${queryParams.toString()}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );

      if (prepareResponse.status !== 200 || !prepareResponse.data.download_url) {
        throw new Error('Failed to prepare file for download.');
      }

      const { download_url, file_name } = prepareResponse.data;

      // Step 2: Download the actual file using the provided URL
      // The download_url from backend is like /financial-data/files/filename.csv
      // Our axiosInstance has baseURL: '/api', so this will become /api/financial-data/files/filename.csv
      const fileResponse = await api.get(download_url, {
        headers: { Authorization: `Bearer ${token}` },
        responseType: 'blob', // Important for file downloads
      });

      if (fileResponse.status !== 200) {
        throw new Error(`Error downloading file: ${fileResponse.statusText}`);
      }

      // Create a link and trigger the download
      const blob = new Blob([fileResponse.data], { type: 'text/csv' });
      const link = document.createElement('a');
      link.href = window.URL.createObjectURL(blob);
      link.download = file_name || `financial_data_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(link.href);

      toast.success('Data downloaded successfully!');
      set({ isDownloading: false });
      return true;

    } catch (err: any) {
      logger.error('Error downloading financial data:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to download data.';
      set({ error: errorMessage, isDownloading: false });
      toast.error(`Download failed: ${errorMessage}`);
      return false;
    }
  },

  fetchAccessLimits: async () => {
    if (get().accessLimits && Object.keys(get().accessLimits!).length > 0) return;
    set({ isLoading: true }); 
    try {
      const response = await api.get('/financial-data/limits');
      set({ accessLimits: response.data.limits, isLoading: false });
    } catch (error: any) {
      set({ isLoading: false, error: error.response?.data?.detail || 'Failed to fetch access limits' });
    }
  },
})); 