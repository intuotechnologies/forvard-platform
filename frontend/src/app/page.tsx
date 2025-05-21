"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuthStore } from "@/store/auth-store"

export default function Home() {
  const router = useRouter()
  const { isAuthenticated } = useAuthStore()

  useEffect(() => {
    if (isAuthenticated) {
      router.push("/dashboard")
    } else {
      router.push("/login")
    }
  }, [router, isAuthenticated])

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="flex flex-col items-center">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-600 mb-4"></div>
        <h2 className="text-xl font-semibold text-gray-700">Loading...</h2>
      </div>
    </div>
  )
}
