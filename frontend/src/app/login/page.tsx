"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { useForm } from "react-hook-form"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { useAuthStore } from "@/store/auth-store"
import Image from "next/image"
import { AlertCircle } from "lucide-react"
import { RefreshCw } from "lucide-react"

interface LoginFormValues {
  email: string
  password: string
}

export default function LoginPage() {
  const router = useRouter()
  const { login, isLoading, error: authError, isAuthenticated } = useAuthStore()
  const [formError, setFormError] = useState<string | null>(null)

  const { register, handleSubmit, formState: { errors } } = useForm<LoginFormValues>()

  const onSubmit = async (data: LoginFormValues) => {
    setFormError(null);
    // The login function in authStore already sets its own error state.
    // We rely on that for displaying auth-related errors.
    await login(data.email, data.password);
    // Navigation will be handled by authStore or middleware upon successful login / isAuthenticated change
  };

  // Watch for authStore error changes to display them
  useEffect(() => {
    if (authError) {
      setFormError(authError);
    }
  }, [authError]);

  // Watch for isAuthenticated to redirect after successful login
   useEffect(() => {
    if (isAuthenticated) {
      router.push("/dashboard");
    }
  }, [isAuthenticated, router]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-grins-ivory dark:bg-grins-blue-night p-6 sm:p-8">
      <div className="w-full max-w-lg p-4">
        <div className="text-center mb-10">
          <Image src="/grins-logo.png" alt="GRINS Logo" width={240} height={72} className="mx-auto mb-6" />
          <h1 className="text-4xl font-extrabold text-grins-blue-deep dark:text-grins-ivory tracking-tight">Welcome to GRINS</h1>
          <p className="text-lg text-grins-blue-gray dark:text-grins-gray-light mt-2">Your Financial Data Platform</p>
        </div>
        <Card className="bg-white dark:bg-grins-blue-deep shadow-2xl rounded-xl overflow-hidden border-grins-petrol/20 dark:border-grins-blue-light/20 border">
          <CardHeader className="space-y-2 p-8 bg-grins-petrol text-grins-ivory text-center">
            <CardTitle className="text-3xl font-bold">Login</CardTitle>
            <CardDescription className="text-grins-ivory/90 text-base">
              Access your financial dashboard
            </CardDescription>
          </CardHeader>
          <CardContent className="p-8 space-y-6">
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <div className="space-y-2">
                <label htmlFor="email" className="block text-sm font-semibold text-grins-blue-night dark:text-grins-ivory">Email Address</label>
                <Input
                  id="email"
                  type="email"
                  placeholder="name@example.com"
                  className="w-full px-4 py-3 rounded-lg bg-grins-ivory/50 dark:bg-grins-blue-night/50 border-grins-gray-light dark:border-grins-blue-gray focus:border-grins-petrol dark:focus:border-grins-yellow-light focus:ring-grins-petrol dark:focus:ring-grins-yellow-light text-grins-blue-night dark:text-grins-ivory placeholder:text-grins-blue-gray/70 dark:placeholder:text-grins-gray-light/70"
                  {...register("email", {
                    required: "Email is required",
                    pattern: {
                      value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                      message: "Invalid email address format",
                    },
                  })}
                />
                {errors.email && (
                  <p className="text-xs text-red-500 dark:text-red-400 flex items-center pt-1"><AlertCircle className="w-3.5 h-3.5 mr-1.5 flex-shrink-0"/>{errors.email.message}</p>
                )}
              </div>
              <div className="space-y-2">
                <label htmlFor="password" className="block text-sm font-semibold text-grins-blue-night dark:text-grins-ivory">Password</label>
                <Input
                  id="password"
                  type="password"
                  placeholder="Enter your password"
                  className="w-full px-4 py-3 rounded-lg bg-grins-ivory/50 dark:bg-grins-blue-night/50 border-grins-gray-light dark:border-grins-blue-gray focus:border-grins-petrol dark:focus:border-grins-yellow-light focus:ring-grins-petrol dark:focus:ring-grins-yellow-light text-grins-blue-night dark:text-grins-ivory placeholder:text-grins-blue-gray/70 dark:placeholder:text-grins-gray-light/70"
                  {...register("password", {
                    required: "Password is required",
                  })}
                />
                {errors.password && (
                  <p className="text-xs text-red-500 dark:text-red-400 flex items-center pt-1"><AlertCircle className="w-3.5 h-3.5 mr-1.5 flex-shrink-0"/>{errors.password.message}</p>
                )}
              </div>
              {(formError) && (
                <div className="p-4 rounded-lg bg-red-100 dark:bg-red-900/50 border border-red-400 dark:border-red-700 text-red-700 dark:text-red-300 text-sm flex items-start">
                  <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0 mt-0.5" />
                  <span className="flex-1">{formError}</span>
                </div>
              )}
              <Button 
                type="submit" 
                className="w-full py-3 px-4 rounded-lg font-semibold text-lg
                           bg-grins-yellow-light text-grins-blue-night 
                           hover:bg-yellow-400 dark:hover:bg-grins-yellow-light/90 
                           focus:outline-none focus:ring-4 focus:ring-yellow-500/50 dark:focus:ring-grins-yellow-light/50
                           transition-all duration-150 ease-in-out shadow-lg hover:shadow-xl 
                           transform hover:-translate-y-1 disabled:opacity-70 disabled:cursor-not-allowed"
                disabled={isLoading}
              >
                {isLoading ? (
                  <span className="flex items-center justify-center">
                    <RefreshCw className="animate-spin h-5 w-5 mr-3" /> 
                    Authenticating...
                  </span>
                ) : "Sign In"}
              </Button>
            </form>
          </CardContent>
          <CardFooter className="flex justify-center p-6 bg-grins-ivory/50 dark:bg-grins-blue-deep/50 border-t border-grins-gray-light/50 dark:border-grins-blue-gray/50">
            <p className="text-xs text-grins-blue-gray dark:text-grins-gray-light">
              &copy; {new Date().getFullYear()} GRINS - All Rights Reserved
            </p>
          </CardFooter>
        </Card>
      </div>
    </div>
  )
} 