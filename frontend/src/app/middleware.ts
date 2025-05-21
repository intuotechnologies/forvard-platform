import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'
import { jwtDecode } from 'jwt-decode' // Assicurati che sia installato e importato correttamente

interface TokenData {
  sub: string;
  role: string;
  exp: number;
}

const ADMIN_ROUTES = ['/admin']; // Definisci qui le tue route admin
const USER_DASHBOARD = '/dashboard';
const ADMIN_DASHBOARD = '/admin/dashboard'; // Esempio, se hai una dashboard admin separata
const LOGIN_URL = '/login';

// This function can be marked `async` if using `await` inside
export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl
  const authCookie = request.cookies.get('auth-storage')
  
  let isAuthenticated = false
  let userRole: string | null = null
  
  if (authCookie) {
    try {
      const authState = JSON.parse(authCookie.value)
      if (authState.state?.token) {
        const decodedToken = jwtDecode<TokenData>(authState.state.token)
        const currentTime = Math.floor(Date.now() / 1000)
        if (decodedToken.exp > currentTime) {
          isAuthenticated = true
          userRole = decodedToken.role
        }
      }
    } catch (e) {
      console.error("Error parsing auth cookie or decoding token:", e)
      // Consider a corrupted cookie/token as unauthenticated
      // Potresti voler eliminare il cookie qui
      const response = NextResponse.redirect(new URL(LOGIN_URL, request.url))
      response.cookies.delete('auth-storage')
      return response
    }
  }
  
  // Redirect to login if not authenticated and accessing protected routes
  if (!isAuthenticated && (pathname.startsWith('/dashboard') || pathname.startsWith('/admin') || pathname === '/')) {
    if (pathname === LOGIN_URL) return NextResponse.next() // Allow access to login page itself
    return NextResponse.redirect(new URL(LOGIN_URL, request.url))
  }
  
  // If authenticated
  if (isAuthenticated) {
    // Redirect from login page if already authenticated
    if (pathname === LOGIN_URL) {
        // Potresti reindirizzare a dashboard diverse in base al ruolo
        // if (userRole === 'admin') return NextResponse.redirect(new URL(ADMIN_DASHBOARD, request.url))
        return NextResponse.redirect(new URL(USER_DASHBOARD, request.url))
    }

    // Root path redirect
    if (pathname === '/') {
        // if (userRole === 'admin') return NextResponse.redirect(new URL(ADMIN_DASHBOARD, request.url))
        return NextResponse.redirect(new URL(USER_DASHBOARD, request.url))
    }

    // Role-based access control (esempio)
    // Se un utente non admin tenta di accedere a una route admin
    if (userRole !== 'admin' && ADMIN_ROUTES.some(route => pathname.startsWith(route))) {
      console.warn(`User with role '${userRole}' attempted to access admin route: ${pathname}`)
      // Reindirizza a una pagina di "accesso negato" o alla dashboard utente
      return NextResponse.redirect(new URL(USER_DASHBOARD, request.url)) 
    }
    
    // Se un admin accede a una route non admin, permettilo o reindirizza come preferisci
    // if (userRole === 'admin' && !ADMIN_ROUTES.some(route => pathname.startsWith(route)) && pathname !== USER_DASHBOARD) {
    //   return NextResponse.redirect(new URL(ADMIN_DASHBOARD, request.url))
    // }
  }
  
  return NextResponse.next()
}

// Add paths that you want to match
export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
} 