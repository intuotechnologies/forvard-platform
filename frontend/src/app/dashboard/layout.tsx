"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { useAuthStore } from "@/store/auth-store"
import { 
  LineChart, 
  LayoutDashboard, 
  LogOut, 
  Menu, 
  X,
  UserCircle,
  Briefcase,
  TestTubeDiagonal,
  Wand,
  ListFilter,
  Newspaper
} from "lucide-react"
import { Button } from "@/components/ui/button"
import Image from "next/image"

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const pathname = usePathname()
  const router = useRouter()
  const { user, logout, isAuthenticated } = useAuthStore()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    if (isAuthenticated === false) {
      router.push("/login")
    }
  }, [isAuthenticated, router])

  const handleLogout = () => {
    logout()
    router.push("/login")
  }

  const navigation = [
    { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
    { name: "Volatility Data", href: "/dashboard/financial-data", icon: LineChart },
    { name: "Profile", href: "/dashboard/profile", icon: UserCircle },
  ]

  const filteredNavigation = navigation.filter(item => {
    return true
  })

  if (!isAuthenticated || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <p className="text-foreground">Loading dashboard...</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="lg:hidden">
        <div className="fixed inset-0 flex z-40">
          <div
            className={`fixed inset-0 bg-grins-blue-night bg-opacity-75 transition-opacity ${
              isMobileMenuOpen ? "opacity-100" : "opacity-0 pointer-events-none"
            }`}
            aria-hidden="true"
            onClick={() => setIsMobileMenuOpen(false)}
          />

          <div
            className={`relative flex-1 flex flex-col max-w-xs w-full bg-grins-ivory dark:bg-grins-blue-deep transition ease-in-out duration-300 transform ${
              isMobileMenuOpen ? "translate-x-0" : "-translate-x-full"
            }`}
          >
            <div className="absolute top-0 right-0 -mr-12 pt-2">
              {isMobileMenuOpen && (
                <button
                  type="button"
                  className="ml-1 flex items-center justify-center h-10 w-10 rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-grins-yellow-light text-grins-ivory"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  <span className="sr-only">Close sidebar</span>
                  <X className="h-6 w-6" />
                </button>
              )}
            </div>

            <div className="flex-1 h-0 pt-5 pb-4 overflow-y-auto">
              <div className="flex-shrink-0 flex items-center px-4">
                <Image src="/grins-logo.png" alt="GRINS Logo" width={120} height={30} />
              </div>
              <nav className="mt-8 px-2 space-y-1">
                {filteredNavigation.map((item) => (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={`group flex items-center px-3 py-2 text-base font-medium rounded-md transition-colors duration-150 ease-in-out ${
                      pathname === item.href
                        ? "bg-grins-yellow-light text-grins-blue-night"
                        : "text-grins-blue-gray dark:text-grins-gray-light hover:bg-grins-yellow-hover dark:hover:bg-grins-blue-night hover:text-grins-blue-night dark:hover:text-grins-yellow-light"
                    }`}
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    <item.icon
                      className={`mr-4 flex-shrink-0 h-6 w-6 transition-colors duration-150 ease-in-out ${
                        pathname === item.href ? "text-grins-blue-night" : "text-grins-blue-gray group-hover:text-grins-blue-night"
                      }`}
                    />
                    {item.name}
                  </Link>
                ))}
              </nav>
            </div>
            <div className="flex-shrink-0 flex border-t border-grins-gray-light p-4">
              <div className="flex-shrink-0 group block">
                <div className="flex items-center">
                  <div className="h-10 w-10 rounded-full bg-grins-petrol flex items-center justify-center text-grins-ivory font-semibold text-lg">
                    {user.email.charAt(0).toUpperCase()}
                  </div>
                  <div className="ml-3">
                    <p className="text-sm font-medium text-grins-blue-night">{user.email}</p>
                    <p className="text-xs font-medium text-grins-blue-gray capitalize">{user.role_name}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="flex-shrink-0 w-14" aria-hidden="true"></div>
        </div>
      </div>

      <div className="hidden lg:flex lg:w-64 lg:flex-col lg:fixed lg:inset-y-0">
        <div className="flex-1 flex flex-col min-h-0 border-r border-grins-gray-light bg-grins-ivory dark:bg-grins-blue-deep">
          <div className="flex-1 flex flex-col pt-5 pb-4 overflow-y-auto">
            <div className="flex items-center flex-shrink-0 px-4 mb-5">
              <Image src="/grins-logo.png" alt="GRINS Logo" width={150} height={40} />
            </div>
            <nav className="mt-5 flex-1 px-2 space-y-1">
              {filteredNavigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors duration-150 ease-in-out ${
                    pathname === item.href
                      ? "bg-grins-yellow-light text-grins-blue-night shadow-sm"
                      : "text-grins-blue-gray dark:text-grins-gray-light hover:bg-grins-yellow-hover dark:hover:bg-grins-blue-night hover:text-grins-blue-night dark:hover:text-grins-yellow-light"
                  }`}
                >
                  <item.icon
                    className={`mr-3 flex-shrink-0 h-5 w-5 transition-colors duration-150 ease-in-out ${
                      pathname === item.href ? "text-grins-blue-night" : "text-grins-blue-gray group-hover:text-grins-blue-night"
                    }`}
                  />
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>
          <div className="flex-shrink-0 flex flex-col border-t border-grins-gray-light p-4 space-y-2">
            <div className="flex-shrink-0 w-full group block">
              <div className="flex items-center">
                <div className="h-9 w-9 rounded-full bg-grins-petrol flex items-center justify-center text-grins-ivory font-semibold">
                  {user.email.charAt(0).toUpperCase()}
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-grins-blue-night dark:text-grins-ivory">{user.email}</p>
                  <p className="text-xs font-medium text-grins-blue-gray dark:text-grins-gray-light capitalize">{user.role_name}</p>
                </div>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleLogout}
              className="w-full flex items-center justify-start text-grins-blue-gray dark:text-grins-gray-light group"
            >
              <LogOut className="mr-3 h-5 w-5 text-grins-blue-gray dark:text-grins-gray-light group-hover:text-grins-blue-night dark:group-hover:text-grins-yellow-light" />
              Logout
            </Button>
          </div>
        </div>
      </div>

      <div className="lg:pl-64 flex flex-col flex-1">
        <div className="sticky top-0 z-30 flex-shrink-0 flex h-16 bg-grins-ivory dark:bg-grins-blue-deep border-b border-grins-gray-light lg:hidden">
          <button
            type="button"
            className="px-4 border-r border-grins-gray-light text-grins-blue-gray focus:outline-none focus:ring-2 focus:ring-inset focus:ring-grins-petrol lg:hidden"
            onClick={() => setIsMobileMenuOpen(true)}
          >
            <span className="sr-only">Open sidebar</span>
            <Menu className="h-6 w-6" />
          </button>
          <div className="flex-1 px-4 flex justify-between items-center">
            <div className="flex-1 flex">
            </div>
            <div className="ml-4 flex items-center md:ml-6">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleLogout}
                className="text-grins-blue-gray dark:text-grins-gray-light group"
              >
                <LogOut className="h-5 w-5 mr-1 md:mr-2 text-grins-blue-gray dark:text-grins-gray-light group-hover:text-grins-blue-night dark:group-hover:text-grins-yellow-light" />
                <span className="hidden sm:inline">Logout</span>
              </Button>
            </div>
          </div>
        </div>

        <main className="flex-1 pb-8">
          <div className="py-6">
            <div className="max-w-full mx-auto px-4 sm:px-6 md:px-8">
              <div className="py-4">{children}</div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
} 