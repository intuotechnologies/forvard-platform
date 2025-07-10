import { NextRequest, NextResponse } from 'next/server'

// API URLs - Use a server-side runtime environment variable
const API_URL = process.env.API_URL || 'http://localhost:8443'

export async function GET(
  request: NextRequest,
  context: { params: { path: string[] } }
) {
  const routeParams = await context.params;
  const pathProperty = routeParams.path; // Access .path separately
  
  // Ensure params is properly handled
  const pathArray = Array.isArray(pathProperty) ? pathProperty : [pathProperty].filter(Boolean);
  const path = pathArray.join('/')
  const { searchParams } = new URL(request.url)
  
  // Add searchParams to URL
  const queryString = searchParams.toString()
  const url = `${API_URL}/${path}${queryString ? `?${queryString}` : ''}`

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  // Forward authorization header if present
  const authHeader = request.headers.get('Authorization')
  if (authHeader) {
    headers['Authorization'] = authHeader
  }

  try {
    console.log(`[API PROXY] GET: ${url}`)
    const response = await fetch(url, {
      method: 'GET',
      headers,
    })

    // Handle different content types (especially for file downloads)
    const contentType = response.headers.get("content-type");
    if (contentType && (contentType.includes("application/octet-stream") || contentType.includes("text/csv") || contentType.includes("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))) {
      console.log(`[API PROXY] Streaming file response from: ${url} with status: ${response.status}`)
      const blob = await response.blob();
      return new NextResponse(blob, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    const data = await response.json()
    console.log(`[API PROXY] GET response from: ${url} with status: ${response.status}`)

    return NextResponse.json(data, {
      status: response.status,
    })
  } catch (error) {
    console.error(`[API PROXY] Error proxying GET request to ${url}:`, error)
    return NextResponse.json(
      { error: 'Failed to fetch data from API' },
      { status: 500 }
    )
  }
}

export async function POST(
  request: NextRequest,
  context: { params: { path: string[] } }
) {
  const routeParams = await context.params;
  const pathProperty = routeParams.path; // Access .path separately

  // Ensure params is properly handled
  const pathArray = Array.isArray(pathProperty) ? pathProperty : [pathProperty].filter(Boolean);
  const path = pathArray.join('/')
  const url = `${API_URL}/${path}`

  const headers: HeadersInit = {}

  // Forward content type header
  const contentTypeHeader = request.headers.get('Content-Type')
  if (contentTypeHeader) {
    headers['Content-Type'] = contentTypeHeader
  }

  // Forward authorization header if present
  const authHeader = request.headers.get('Authorization')
  if (authHeader) {
    headers['Authorization'] = authHeader
  }

  try {
    console.log(`[API PROXY] POST: ${url}`)
    const body = await request.text()
    
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body,
    })

    const data = await response.json()
    console.log(`[API PROXY] POST response from: ${url} with status: ${response.status}`)

    return NextResponse.json(data, {
      status: response.status,
    })
  } catch (error) {
    console.error(`[API PROXY] Error proxying POST request to ${url}:`, error)
    return NextResponse.json(
      { error: 'Failed to send data to API' },
      { status: 500 }
    )
  }
} 