#!/bin/bash
# Script per impostare le variabili d'ambiente per l'API

# API server configuration
export API_PORT=8000

# Database connection
export DATABASE_URL_API="postgresql://admin:admin@79.72.44.95:5432/forvard"
export POSTGRES_APP_USER="admin"
export POSTGRES_APP_PASSWORD="admin"
export POSTGRES_APP_DB="forvard"
export POSTGRES_DB_PORT_INTERNAL=5432
export POSTGRES_DB_HOST="79.72.44.95"

# JWT configuration
export JWT_SECRET_KEY="forvard_secret_key_change_me_in_production"
export ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS configuration
export CORS_ORIGINS="http://localhost,http://localhost:3000"

# Logging configuration
export LOG_LEVEL="INFO"
export LOG_FILE="./logs/api.log"

# Admin panel
export SESSION_SECRET_KEY="forvard_session_key_change_me_in_production"

echo "Variabili d'ambiente configurate correttamente."
echo "Ricorda di eseguire questo script con source: source setup_env.sh" 