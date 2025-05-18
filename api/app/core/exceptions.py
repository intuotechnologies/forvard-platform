from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger
from uuid import uuid4
import traceback


class AppException(Exception):
    """Base exception for application-specific exceptions"""
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code or "general_error"


class AccessLimitExceeded(AppException):
    """Exception raised when a user exceeds their access limits"""
    def __init__(self, detail: str = "Access limit exceeded"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="access_limit_exceeded"
        )


class ResourceNotFound(AppException):
    """Exception raised when a requested resource is not found"""
    def __init__(self, resource: str, detail: str = None):
        message = detail or f"Resource not found: {resource}"
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message,
            error_code="resource_not_found"
        )


def setup_exception_handlers(app: FastAPI):
    """Register exception handlers with the FastAPI app"""
    
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        """Handler for application-specific exceptions"""
        error_id = str(uuid4())
        logger.error(
            f"Application exception: {exc.detail}",
            error_id=error_id,
            status_code=exc.status_code,
            error_code=exc.error_code,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_id": error_id,
                "error_code": exc.error_code,
                "detail": exc.detail
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handler for request validation errors"""
        error_id = str(uuid4())
        errors = exc.errors()
        
        logger.error(
            "Request validation error",
            error_id=error_id,
            errors=errors,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error_id": error_id,
                "error_code": "validation_error",
                "detail": "Invalid request data",
                "errors": errors
            }
        )
    
    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
        """Handler for database errors"""
        error_id = str(uuid4())
        
        logger.error(
            f"Database error: {str(exc)}",
            error_id=error_id,
            error_type=type(exc).__name__,
            path=request.url.path,
            traceback=traceback.format_exc()
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_id": error_id,
                "error_code": "database_error",
                "detail": "A database error occurred"
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handler for all other exceptions"""
        error_id = str(uuid4())
        
        logger.error(
            f"Unhandled exception: {str(exc)}",
            error_id=error_id,
            error_type=type(exc).__name__,
            path=request.url.path,
            traceback=traceback.format_exc()
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_id": error_id,
                "error_code": "server_error",
                "detail": "An unexpected error occurred"
            }
        ) 