# ForVARD Platform API

A financial data REST API service with authentication, role-based access control, and admin panel.

## Overview

ForVARD Platform API provides secure access to financial volatility data with comprehensive user management and role-based permissions. Built on FastAPI, it offers excellent performance, automatic OpenAPI documentation, and a modern admin interface.

## Technology Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Database**: PostgreSQL
- **Authentication**: JWT (JSON Web Tokens)
- **Admin Panel**: SQLAdmin
- **Dependency Management**: Poetry

## Features

- **Secure Authentication**: JWT-based authentication system
- **Role-Based Access Control**: Base, Senator, and Admin user roles
- **Admin Panel**: Web interface for user and data management
- **API Documentation**: Auto-generated OpenAPI/Swagger docs
- **Financial Data Storage**: Database structure for volatility metrics
- **Logging System**: Advanced logging with rotation and JSON formatting

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 13+
- Poetry (Python dependency manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/forvard-platform.git
   cd forvard-platform/api
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Configure environment variables:
   ```bash
   cp setup_env.sh.example setup_env.sh
   # Edit setup_env.sh with your database credentials and settings
   ```

4. Setup database:
   ```bash
   source setup_env.sh
   poetry run python app/src/init_users.py
   ```

5. Run the application:
   ```bash
   source setup_env.sh
   poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload-dir app
   ```

## Environment Variables

Key environment variables are set in `setup_env.sh`:

- `API_PORT`: API server port (default: 8000)
- `DATABASE_URL_API`: PostgreSQL connection string
- `JWT_SECRET_KEY`: Secret key for JWT token generation
- `SESSION_SECRET_KEY`: Secret key for admin session
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `LOG_FILE`: Path to log file

## API Endpoints

### Authentication

- `POST /auth/register`: Register a new user
- `POST /auth/token`: Login and get access token

### Financial Data

- `GET /financial-data/`: List available financial data
- `GET /financial-data/{id}`: Get specific financial data item

### System

- `GET /health`: Health check endpoint
- `GET /docs`: Swagger UI API documentation
- `GET /redoc`: ReDoc API documentation

## Admin Panel

The admin interface is accessible at `/admin` and provides:

- User management
- Role management
- Access control configuration
- Financial data management

Default admin credentials:
- Email: admin@example.com
- Password: adminpass

## User Roles

- **Base**: Standard access to limited financial data
- **Senator**: Extended access to financial data and analytics
- **Admin**: Full system access including admin panel

## Database Schema

### Core Tables

- `users`: User accounts and authentication
- `roles`: User role definitions
- `asset_access_limits`: Role-based access control settings
- `realized_volatility_data`: Financial volatility metrics

## Data Loading

To load volatility data from CSV:

```bash
source setup_env.sh
poetry run python app/src/init_rv_data.py --file path/to/your/data.csv
```

## Development

### Running in Development Mode

```bash
source setup_env.sh
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload-dir app
```

The `--reload-dir app` flag ensures that only changes to your application code trigger reloading, not changes to dependencies.

### Code Structure

- `app/main.py`: Application entry point
- `app/routers/`: API route definitions
- `app/models/`: Pydantic models for requests/responses
- `app/core/`: Core functionality (auth, database, etc.)
- `app/admin/`: Admin panel configuration

## Deployment

### Docker Deployment

```bash
docker build -t forvard-api .
docker run -p 8000:8000 --env-file .env forvard-api
```

### Docker Compose

```bash
docker-compose up -d
```

## Troubleshooting

### Common Issues

- **Database Connection Errors**: Verify PostgreSQL credentials in setup_env.sh
- **JWT Authentication Issues**: Check that JWT_SECRET_KEY is properly set
- **Admin Panel Login Failures**: Ensure admin user exists in database

## License

MIT

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLAdmin](https://aminalaee.dev/sqladmin/)
- [Pydantic](https://pydantic-docs.helpmanual.io/)
- [SQLAlchemy](https://www.sqlalchemy.org/) 