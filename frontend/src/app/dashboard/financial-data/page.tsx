"use client"

import { useState, useEffect, useCallback } from 'react'
import { useFinancialDataStore } from '@/store/financial-data-store'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { 
  LineChart, ResponsiveContainer, Line, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, Brush
} from 'recharts'
import { Download, Filter, RefreshCw } from 'lucide-react'
import { formatDate } from '@/lib/utils'
import { useAuthStore } from '@/store/auth-store'

export default function FinancialDataPage() {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  const { 
    data, 
    isLoading, 
    error, 
    totalCount, 
    page, 
    limit, 
    fetchData, 
    downloadData,
    accessLimits,
    fetchAccessLimits,
    isDownloading
  } = useFinancialDataStore()
  const { isAuthenticated, user } = useAuthStore()

  // Filter state
  const [filters, setFilters] = useState({
    symbol: '',
    asset_type: '',
    start_date: '',
    end_date: '',
    page: 1,
    limit: 100,
    fields: ['close_price', 'rv5'] // Default fields to fetch and potentially plot
  })

  // Available fields for the chart
  const availableFields = [
    { value: 'open_price', label: 'Open Price' },
    { value: 'close_price', label: 'Close Price' },
    { value: 'high_price', label: 'High Price' },
    { value: 'low_price', label: 'Low Price' },
    { value: 'volume', label: 'Volume' },
    { value: 'trades', label: 'Trades' },
    { value: 'pv', label: 'PV (Parkinson Volatility)' },
    { value: 'gk', label: 'GK (Garman-Klass Volatility)' },
    { value: 'rr5', label: 'RR5 (5-day Realized Range)' }, // Assuming RR stands for Realized Range
    { value: 'rv1', label: 'RV1 (1-day Realized Volatility)' },
    { value: 'rv5', label: 'RV5 (5-day Realized Volatility)' },
    { value: 'rv5_ss', label: 'RV5_SS (5-day Realized Volatility Sub-Sampled)' },
    { value: 'bv1', label: 'BV1 (1-day Bipower Variation)' },
    { value: 'bv5', label: 'BV5 (5-day Bipower Variation)' },
    { value: 'bv5_ss', label: 'BV5_SS (5-day Bipower Variation Sub-Sampled)' },
    { value: 'rsp1', label: 'RSP1 (1-day Realized Semi-variance Positive)' },
    { value: 'rsn1', label: 'RSN1 (1-day Realized Semi-variance Negative)' },
    { value: 'rsp5', label: 'RSP5 (5-day Realized Semi-variance Positive)' },
    { value: 'rsn5', label: 'RSN5 (5-day Realized Semi-variance Negative)' },
    { value: 'rsp5_ss', label: 'RSP5_SS (5-day Realized Semi-variance Positive Sub-Sampled)' },
    { value: 'rsn5_ss', label: 'RSN5_SS (5-day Realized Semi-variance Negative Sub-Sampled)' },
    { value: 'medrv1', label: 'MedRV1 (1-day Median Realized Volatility)' },
    { value: 'medrv5', label: 'MedRV5 (5-day Median Realized Volatility)' },
    { value: 'medrv5_ss', label: 'MedRV5_SS (5-day Median Realized Volatility Sub-Sampled)' },
    { value: 'minrv1', label: 'MinRV1 (1-day Minimum Realized Volatility)' }, // Assuming MinRV
    { value: 'minrv5', label: 'MinRV5 (5-day Minimum Realized Volatility)' }, // Assuming MinRV
    { value: 'minrv5_ss', label: 'MinRV5_SS (5-day Minimum Realized Volatility Sub-Sampled)' }, // Assuming MinRV
    { value: 'rk', label: 'RK (Realized Kurtosis)' },
  ]

  // Chart config now focuses on which of the fetched `fields` to display prominently
  const [chartDisplayConfig, setChartDisplayConfig] = useState({
    primaryMetric: 'close_price', // Main Y-axis
    secondaryMetric: 'rv5',     // Optional second Y-axis
  })

  // Load data on initial render
  useEffect(() => {
    if (isAuthenticated && user) {
      fetchAccessLimits()
    }
  }, [isAuthenticated, user, fetchAccessLimits])

  // Load data when filters or chartConfig change
  const loadData = useCallback(() => {
    const coreTableFields = ['open_price', 'close_price', 'high_price', 'low_price', 'volume', 'trades'];
    // Ensure essential metrics for display and table are included in fields to fetch
    const fieldsToFetch = Array.from(new Set([
      chartDisplayConfig.primaryMetric,
      chartDisplayConfig.secondaryMetric,
      ...coreTableFields, // Aggiungiamo campi core per la tabella
      'observation_date', 'symbol', 'asset_type' // Always fetch these for table and context
    ].filter(Boolean))); // .filter(Boolean) rimuove eventuali valori null o undefined (es. se secondaryMetric è vuoto)

    fetchData({ 
      ...filters, // symbol, asset_type, start_date, end_date, page, limit
      page: 1, // Reset to page 1 on new filter/field application
      fields: fieldsToFetch, // Passiamo i campi costruiti
    });
  }, [fetchData, filters, chartDisplayConfig]); // Includiamo chartDisplayConfig per triggerare il fetch quando le metriche del grafico cambiano

  // Initialize with some data
  useEffect(() => {
    loadData()
  }, [loadData])

  // Handle filter changes
  const handleFilterChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFilters(prev => ({ ...prev, [name]: value }))
  }

  const handleMultiSelectFieldsChange = (selectedFields: string[]) => {
    setFilters(prev => ({ ...prev, fields: selectedFields }));
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    loadData()
  }

  // Handle chart metric selection for prominent display
  const handleChartDisplayConfigChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const { name, value } = e.target
    setChartDisplayConfig(prev => ({ ...prev, [name]: value }))
  }

  // Handle data download
  const handleDownload = async () => {
    const success = await downloadData(filters);
    if (success) {
      console.log("Download initiated successfully.");
    } else {
      console.error("Download failed. Check console for errors.");
    }
  }

  // Format data for chart, plotting all fields in `filters.fields` apart from date/symbol/type
  const chartData = data.map(item => {
    const point: any = {
      date: formatDate(item.observation_date),
      symbol: item.symbol,
      // asset_type: item.asset_type, // Not usually plotted directly
    };
    filters.fields.forEach(field => {
      if (item[field] !== undefined && typeof item[field] === 'number') {
        point[field] = item[field];
      }
    });
    return point;
  });

  const lineColors = ["#2563eb", "#f97316", "#10b981", "#8b5cf6", "#ec4899"];

  // Ricalcoliamo i campi che dovrebbero essere stati fetchati per la tabella
  const coreTableFieldsForRender = ['open_price', 'close_price', 'high_price', 'low_price', 'volume', 'trades'];
  const tableColumns = Array.from(new Set([
    chartDisplayConfig.primaryMetric,
    chartDisplayConfig.secondaryMetric,
    ...coreTableFieldsForRender,
    // Aggiungiamo manualmente i campi non numerici se vogliamo vederli esplicitamente e non sono già inclusi
    // 'symbol', 'asset_type' // observation_date è già la prima colonna dedicata
  ].filter(Boolean)));

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Volatility Data Analysis</h1>
        <p className="text-gray-500 mt-1">
          Filter, visualize, and download volatility estimators and financial data
        </p>
      </div>

      {/* Filters */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="text-lg flex items-center">
            <Filter className="mr-2 h-5 w-5" />
            Data Filters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <label htmlFor="symbol" className="block text-sm font-medium text-gray-700 mb-1">
                  Symbol
                </label>
                <Input
                  id="symbol"
                  name="symbol"
                  placeholder="e.g. AAPL, BTC"
                  value={filters.symbol}
                  onChange={handleFilterChange}
                />
              </div>
              
              <div>
                <label htmlFor="asset_type" className="block text-sm font-medium text-gray-700 mb-1">
                  Asset Type
                </label>
                <Select
                  id="asset_type"
                  name="asset_type"
                  value={filters.asset_type}
                  onChange={handleFilterChange}
                >
                  <option value="">All Types</option>
                  <option value="equity">Equity</option>
                  <option value="fx">Foreign Exchange</option>
                  <option value="crypto">Cryptocurrency</option>
                  <option value="future">Future</option>
                  <option value="bond">Bond</option>
                </Select>
              </div>
              
              <div>
                <label htmlFor="start_date" className="block text-sm font-medium text-gray-700 mb-1">
                  Start Date
                </label>
                <Input
                  id="start_date"
                  name="start_date"
                  type="date"
                  value={filters.start_date}
                  onChange={handleFilterChange}
                  className="dark:bg-grins-blue-night"
                />
              </div>
              
              <div>
                <label htmlFor="end_date" className="block text-sm font-medium text-gray-700 mb-1">
                  End Date
                </label>
                <Input
                  id="end_date"
                  name="end_date"
                  type="date"
                  value={filters.end_date}
                  onChange={handleFilterChange}
                  className="dark:bg-grins-blue-night"
                />
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="primaryMetric" className="block text-sm font-medium text-gray-700 mb-1">
                  Primary Chart Metric
                </label>
                <Select
                  id="primaryMetric"
                  name="primaryMetric"
                  value={chartDisplayConfig.primaryMetric}
                  onChange={handleChartDisplayConfigChange}
                >
                  {availableFields.map(field => (
                    <option key={field.value} value={field.value}>
                      {field.label}
                    </option>
                  ))}
                </Select>
              </div>
              
              <div>
                <label htmlFor="secondaryMetric" className="block text-sm font-medium text-gray-700 mb-1">
                  Secondary Chart Metric
                </label>
                <Select
                  id="secondaryMetric"
                  name="secondaryMetric"
                  value={chartDisplayConfig.secondaryMetric}
                  onChange={handleChartDisplayConfigChange}
                >
                  <option value="">None</option>
                  {availableFields.map(field => (
                    <option key={field.value} value={field.value}>
                      {field.label}
                    </option>
                  ))}
                </Select>
              </div>
            </div>
            
            <div className="flex flex-wrap justify-between">
              <div className="flex items-center text-sm text-gray-500">
                {isLoading ? (
                  <span className="flex items-center">
                    <RefreshCw className="animate-spin h-4 w-4 mr-2" />
                    Loading data...
                  </span>
                ) : (
                  <>
                    {totalCount > 0 && (
                      <span>Showing {data.length} of {totalCount} records</span>
                    )}
                  </>
                )}
              </div>
              
              <div className="flex gap-3">
                <Button type="submit" disabled={isLoading}>
                  Apply Filters
                </Button>
                <Button 
                  type="button" 
                  variant="outline" 
                  onClick={handleDownload}
                  disabled={isLoading || isDownloading || data.length === 0}
                >
                  {isDownloading ? (
                    <><RefreshCw className="animate-spin h-4 w-4 mr-2" /> Exporting...</>
                  ) : (
                    <><Download className="h-4 w-4 mr-2" /> Export Data</>
                  )}
                </Button>
              </div>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Data Visualization */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Data Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          {error ? (
            <div className="text-red-500 p-4 bg-red-50 rounded-md">
              {error}
            </div>
          ) : data.length > 0 ? (
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chartData}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" />
                  {chartDisplayConfig.secondaryMetric && <YAxis yAxisId="right" orientation="right" />}
                  <Tooltip />
                  <Legend />
                  {/* Line for Primary Metric */}
                  {chartDisplayConfig.primaryMetric && (
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey={chartDisplayConfig.primaryMetric}
                      stroke={lineColors[0 % lineColors.length]}
                      name={availableFields.find(f => f.value === chartDisplayConfig.primaryMetric)?.label || chartDisplayConfig.primaryMetric}
                      dot={false}
                    />
                  )}
                  {/* Line for Secondary Metric if selected */}
                  {chartDisplayConfig.secondaryMetric && (
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey={chartDisplayConfig.secondaryMetric}
                      stroke={lineColors[1 % lineColors.length]}
                      name={availableFields.find(f => f.value === chartDisplayConfig.secondaryMetric)?.label || chartDisplayConfig.secondaryMetric}
                      dot={false}
                    />
                  )}
                  <Brush dataKey="date" height={30} stroke="#8884d8" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="text-center py-10">
              <p className="text-gray-500">No data available. Please adjust your filters and try again.</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Data Table */}
      {isClient && data.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Data Table</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Date
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Symbol
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    {tableColumns.filter(key => key !== 'observation_date' && key !== 'symbol' && key !== 'asset_type') // Escludiamo quelli già dedicati
                      .map(key => (
                      <th key={key} scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        {availableFields.find(f => f.value === key)?.label || key.replace(/_/g, ' ')}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {isClient && data.map((item, i) => (
                    <tr key={i}> {/* Considerare una chiave più robusta se possibile */}
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatDate(item.observation_date)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {item.symbol}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 capitalize">
                        {item.asset_type}
                      </td>
                      {tableColumns.filter(key => key !== 'observation_date' && key !== 'symbol' && key !== 'asset_type')
                        .map(key => (
                        <td key={key} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {typeof item[key] === 'number' ? parseFloat(item[key].toFixed(4)) : item[key]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 