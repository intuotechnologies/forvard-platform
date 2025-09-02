"use client"

import { useEffect } from 'react'
import { useAuthStore } from '@/store/auth-store'
import { useFinancialDataStore } from '@/store/financial-data-store'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function ProfilePage() {
  const { user, token } = useAuthStore()
  const { accessLimits, fetchAccessLimits, isLoading } = useFinancialDataStore()

  useEffect(() => {
    if (token) {
      fetchAccessLimits()
    }
  }, [token, fetchAccessLimits])

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">User Profile</h1>
        <p className="text-gray-500 mt-1">Manage your account and view access permissions</p>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Account Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Email</label>
              <p className="text-sm text-gray-900">{user?.email || 'Not available'}</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Role</label>
              <p className="text-sm text-gray-900 capitalize">{user?.role || 'Not available'}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Access Limits</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p className="text-sm text-gray-500">Loading access limits...</p>
          ) : (
            <>
              {Object.keys(accessLimits || {}).length > 0 ? (
                <div className="space-y-2">
                  {Object.entries(accessLimits || {}).map(([category, limit]) => (
                    <div key={category} className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-700 capitalize">
                        {category.replace(/_/g, ' ')}
                      </span>
                      <span className="text-sm text-gray-900">{limit} items</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500">No specific access limits defined for your role.</p>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
} 