"use client"

import { useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { useAuthStore } from "@/store/auth-store"
import { useFinancialDataStore } from "@/store/financial-data-store"
import { ShieldCheck, BarChartBig, UserCircle } from "lucide-react"

export default function ProfilePage() {
  const { user, isAuthenticated } = useAuthStore()
  const { accessLimits, fetchAccessLimits } = useFinancialDataStore()

  useEffect(() => {
    if (isAuthenticated && user && !accessLimits) {
      fetchAccessLimits()
    }
  }, [isAuthenticated, user, accessLimits, fetchAccessLimits])

  if (!user) {
    return (
      <div className="flex items-center justify-center h-full">
        <p>Loading user profile...</p>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 flex items-center">
          <UserCircle className="w-8 h-8 mr-3 text-blue-600" />
          User Profile
        </h1>
        <p className="text-gray-500 mt-1">
          View your account details and access permissions.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-xl">Account Information</CardTitle>
          <CardDescription>Your personal and role-based details.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center">
            <p className="w-1/3 text-sm font-medium text-gray-600">Email:</p>
            <p className="w-2/3 text-sm text-gray-800">{user.email}</p>
          </div>
          <div className="flex items-center">
            <p className="w-1/3 text-sm font-medium text-gray-600">User ID:</p>
            <p className="w-2/3 text-sm text-gray-800">{user.user_id}</p>
          </div>
          <div className="flex items-center">
            <p className="w-1/3 text-sm font-medium text-gray-600">Role:</p>
            <p className="w-2/3 text-sm text-gray-800 capitalize flex items-center">
              <ShieldCheck className="w-5 h-5 mr-2 text-green-500" />
              {user.role_name}
            </p>
          </div>
          {user.created_at && (
            <div className="flex items-center">
              <p className="w-1/3 text-sm font-medium text-gray-600">Member Since:</p>
              <p className="w-2/3 text-sm text-gray-800">
                {new Date(user.created_at).toLocaleDateString()}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-xl">Access Limits</CardTitle>
          <CardDescription>Your current access limits per asset category.</CardDescription>
        </CardHeader>
        <CardContent>
          {accessLimits ? (
            <ul className="space-y-3">
              {Object.entries(accessLimits).map(([category, limit]) => (
                <li key={category} className="flex justify-between items-center p-3 bg-gray-50 rounded-md">
                  <span className="text-sm font-medium text-gray-700 capitalize flex items-center">
                    <BarChartBig className="w-5 h-5 mr-2 text-indigo-500" />
                    {category.replace(/_/g, ' ')}
                  </span>
                  <span className="text-sm font-semibold text-gray-900">{limit} items</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-500">Loading access limits...</p>
          )}
          {Object.keys(accessLimits || {}).length === 0 && !isLoading && (
             <p className="text-sm text-gray-500">No specific access limits defined for your role, or still loading.</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
} 