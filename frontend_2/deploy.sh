#!/bin/bash

# Script per deploy del frontend Forvard

set -e  # Exit on any error

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configurazione
IMAGE_NAME="forvard-frontend"
DOCKERHUB_USERNAME="salvini"
VERSION="latest"

echo -e "${GREEN}🚀 Avvio deploy del frontend Forvard${NC}"

# Funzione per stampare messaggi
print_step() {
    echo -e "${YELLOW}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Verifica che Docker sia running
print_step "Verifico che Docker sia in esecuzione..."
if ! docker info > /dev/null 2>&1; then
    print_error "Docker non è in esecuzione. Avvia Docker e riprova."
    exit 1
fi
print_success "Docker è in esecuzione"

# Step 2: Build dell'immagine (multi-architettura)
print_step "Setup buildx per multi-architettura..."
docker buildx create --use --name multiarch-builder 2>/dev/null || docker buildx use multiarch-builder

print_step "Building dell'immagine Docker per AMD64 e ARM64..."
docker buildx build --platform linux/amd64,linux/arm64 -t $IMAGE_NAME:$VERSION --load .
print_success "Build completato"

# Step 3: Test locale (opzionale)
read -p "Vuoi testare l'immagine localmente? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Avvio test locale..."
    docker-compose -f docker-compose.test.yml up -d
    print_success "Container avviato. Verifica su:"
    echo "  - HTTP:  http://localhost"
    echo "  - HTTPS: https://localhost (certificato self-signed)"
    echo ""
    read -p "Premi Enter quando hai finito i test per continuare..."
    docker-compose -f docker-compose.test.yml down
    print_success "Test locale completato"
fi

# Step 4: Tag per DockerHub
if [ -z "$DOCKERHUB_USERNAME" ]; then
    DOCKERHUB_USERNAME="salvini"
fi

print_step "Tagging dell'immagine per DockerHub..."
docker tag $IMAGE_NAME:$VERSION $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION
docker tag $IMAGE_NAME:$VERSION $DOCKERHUB_USERNAME/$IMAGE_NAME:$(date +%Y%m%d-%H%M%S)
print_success "Tagging completato"

# Step 5: Login a DockerHub
print_step "Login a DockerHub..."
docker login
print_success "Login completato"

# Step 6: Push dell'immagine (multi-architettura)
print_step "Push dell'immagine multi-architettura su DockerHub..."
docker buildx build --platform linux/amd64,linux/arm64 \
  -t $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION \
  -t $DOCKERHUB_USERNAME/$IMAGE_NAME:$(date +%Y%m%d-%H%M%S) \
  --push .
print_success "Push completato"

# Step 7: Cleanup locale
read -p "Vuoi rimuovere le immagini locali per liberare spazio? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Pulizia immagini locali..."
    docker rmi $IMAGE_NAME:$VERSION
    print_success "Pulizia completata"
fi

echo ""
print_success "🎉 Deploy completato!"
echo "Immagine disponibile su: $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION"
echo ""
echo "Per avviare in produzione:"
echo "docker run -d -p 80:80 -p 443:443 --name forvard-frontend $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION"
