"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useAuthStore } from "@/store/auth-store"
import { useFinancialDataStore } from "@/store/financial-data-store"
import { formatDate } from "@/lib/utils"
import Link from "next/link"
import { ArrowRight, DatabaseIcon, LineChart, ShieldIcon, LayersIcon } from "lucide-react"

export default function Dashboard() {
  const { user, isAuthenticated } = useAuthStore()
  const { accessLimits, fetchAccessLimits, data: financialData, isLoading: dataLoading, fetchData } = useFinancialDataStore()
  const [latestData, setLatestData] = useState<any[]>([])
  const [totalSymbols, setTotalSymbols] = useState<number>(0);

  useEffect(() => {
    if (isAuthenticated && user && !accessLimits) {
      fetchAccessLimits();
    }
    
    // Fetch some sample data for the dashboard
    const fetchSampleData = async () => {
      try {
        // Using the store's fetchData now for consistency
        await fetchData({ limit: 5, page: 1, fields: ['close_price', 'rv5'] }); 
        // The data is already in the store, but we might want to update latestData separately if needed
        // For now, let's assume the store update triggers a re-render that uses financialData
      } catch (error) {
        console.error('Failed to fetch sample data', error)
      }
    }
    
    fetchSampleData();

    // Fetch total distinct symbols as a quick metric
    const fetchTotalSymbols = async () => {
        try {
            const response = await fetch(`/api/financial-data/symbols/count`, { // Assuming an endpoint like this exists or can be made
                headers: {
                    Authorization: `Bearer ${localStorage.getItem('token')}`,
                },
            });
            if (response.ok) {
                const countData = await response.json();
                setTotalSymbols(countData.total_symbols || 0);
            }
        } catch (error) {
            console.error("Failed to fetch total symbols count", error);
        }
    };
    // fetchTotalSymbols(); // Commented out until endpoint exists

  }, [isAuthenticated, user, accessLimits, fetchAccessLimits, fetchData]) // Added isAuthenticated, user, accessLimits to dependencies

  // Use financialData from the store directly for the latest data table
  const displayData = financialData.slice(0, 5);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Welcome back{user ? `, ${user.email.split('@')[0]}` : ''}</h1>
        <p className="text-gray-500 mt-1">
          Here's an overview of your financial data access
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-500">YOUR ROLE</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center">
              <ShieldIcon className="h-8 w-8 text-blue-500 mr-3" />
              <div>
                <div className="text-2xl font-bold text-gray-900 capitalize">
                  {user?.role_name || 'Loading...'}
                </div>
                <p className="text-gray-500 text-sm">Account type</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-500">DEFINED ACCESS CATEGORIES</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center">
              <LayersIcon className="h-8 w-8 text-green-500 mr-3" />
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {accessLimits ? Object.keys(accessLimits).length : 'Loading...'}
                </div>
                <p className="text-gray-500 text-sm">Asset types with limits</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-500">ACCESS LIMITS</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {accessLimits ? (
                Object.entries(accessLimits).map(([type, limit]) => (
                  <div key={type} className="flex justify-between items-center">
                    <span className="text-gray-600 capitalize">{type}</span>
                    <span className="font-medium">{limit} items</span>
                  </div>
                ))
              ) : (
                <div className="text-gray-500">Loading limits...</div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-500">DATA ACCESS</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center">
              <DatabaseIcon className="h-8 w-8 text-blue-500 mr-3" />
              <div>
                <Link 
                  href="/dashboard/financial-data" 
                  className="text-xl font-bold text-blue-600 hover:text-blue-800 transition-colors flex items-center"
                >
                  Access Financial Data
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
                <p className="text-gray-500 text-sm">View and filter data</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Example: Card for Total Symbols - Requires backend endpoint */}
        {/* <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-500">TOTAL SYMBOLS AVAILABLE</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center">
              <BarChartIcon className="h-8 w-8 text-purple-500 mr-3" /> 
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {totalSymbols || 'N/A'} 
                </div>
                <p className="text-gray-500 text-sm">Distinct financial symbols</p>
              </div>
            </div>
          </CardContent>
        </Card> */}
      </div>

      <div className="mb-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Recent Financial Data Activity</h2>
        
        <div className="bg-white shadow rounded-lg overflow-hidden">
          {dataLoading && <p className="p-4 text-sm text-gray-500">Loading recent data...</p>}
          {!dataLoading && displayData.length > 0 ? (
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
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Close Price
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      RV5
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {displayData.map((item, i) => (
                    <tr key={`${item.symbol}-${item.observation_date}-${i}`}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatDate(item.observation_date)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {item.symbol}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 capitalize">
                        {item.asset_type}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {item.close_price?.toFixed(2) ?? 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {item.rv5?.toFixed(4) ?? 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-10 px-6 text-center">
              <LineChart className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No recent data activity</h3>
              <p className="mt-1 text-sm text-gray-500">Get started by accessing the financial data section.</p>
              <div className="mt-6">
                <Link
                  href="/dashboard/financial-data"
                  className="inline-flex items-center rounded-md bg-blue-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-500"
                >
                  <LineChart className="-ml-0.5 mr-1.5 h-5 w-5" aria-hidden="true" />
                  View Financial Data
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 