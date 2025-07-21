from fastapi import FastAPI, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import time
from dotenv import load_dotenv
from loguru import logger
from starlette.middleware.sessions import SessionMiddleware

from .core.logging import setup_logging
from .core.database import check_db_connection
from .core.exceptions import setup_exception_handlers
from .routers import auth, financial_data, admin
from .admin.panel import setup_admin

# Load environment variables
load_dotenv()

# Initialize logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="ForVARD Financial Data API",
    description="REST API for financial data access with user authentication and role-based access control",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Setup sessions for admin authentication
app.add_middleware(
    SessionMiddleware, 
    secret_key=os.getenv("SESSION_SECRET_KEY", "your-super-secret-key-change-this-in-production")
)

# Setup CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup exception handlers
setup_exception_handlers(app)

# Setup admin panel
setup_admin(app)

# Add request processing time middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to add processing time to response headers
    """
    start_time = time.time()
    
    # Log incoming request
    logger.info(
        f"Request received: {request.method} {request.url.path}",
        client=request.client.host if request.client else "unknown",
        path=request.url.path,
        method=request.method,
        query_params=dict(request.query_params)
    )
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f} sec"
    
    # Log response
    logger.info(
        f"Response sent: {response.status_code}",
        status_code=response.status_code,
        path=request.url.path,
        method=request.method,
        process_time=f"{process_time:.4f} sec"
    )
    
    return response


# Include routers
app.include_router(auth.router)
app.include_router(financial_data.router)
app.include_router(admin.router)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API info"""
    return {
        "app": "ForVARD Financial Data API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "admin": "/admin"
    }


@app.get("/health", tags=["system"])
async def health_check():
    """
    Health check endpoint
    """
    # Check database connection
    db_healthy = await check_db_connection()
    
    if not db_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "message": "Database connection failed"}
        )
    
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    # This will run when file is executed directly (not when imported)
    PORT = int(os.getenv("API_PORT", "8443"))
    
    logger.info(f"Starting API server on port {PORT}")
    
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=PORT,
        reload=True,  # Set to False in production
        log_level="info"
    ) 