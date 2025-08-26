#!/bin/bash

# Script per configurare Let's Encrypt per il frontend Forvard

set -e

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configurazione
DOMAIN="volare.unime.it"
EMAIL="your-email@unime.it"  # CAMBIA QUESTO!

echo -e "${GREEN}🔐 Setup Let's Encrypt per $DOMAIN${NC}"

print_step() {
    echo -e "${YELLOW}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Verifica che l'email sia stata cambiata
if [[ "$EMAIL" == "your-email@unime.it" ]]; then
    print_error "Devi cambiare l'email nel script prima di eseguirlo!"
    exit 1
fi

# Verifica che il dominio sia raggiungibile
print_step "Verifico che il dominio $DOMAIN sia raggiungibile..."
if ! ping -c 1 $DOMAIN &> /dev/null; then
    print_error "Il dominio $DOMAIN non è raggiungibile!"
    exit 1
fi
print_success "Dominio raggiungibile"

# Crea le directory necessarie
print_step "Creo le directory necessarie..."
mkdir -p ssl-certs www-certbot
print_success "Directory create"

# Step 1: Avvia il container temporaneo per il challenge
print_step "Avvio container temporaneo per il challenge HTTP..."

# Crea una configurazione nginx temporanea solo per il challenge
cat > nginx-temp.conf << 'EOF'
server {
    listen 80;
    server_name volare.unime.it;
    
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    
    location / {
        return 200 'Let\'s Encrypt setup in progress...';
        add_header Content-Type text/plain;
    }
}
EOF

# Avvia container temporaneo
docker run -d \
    --name temp-nginx \
    -p 80:80 \
    -v $(pwd)/www-certbot:/var/www/certbot:ro \
    -v $(pwd)/nginx-temp.conf:/etc/nginx/conf.d/default.conf:ro \
    nginx:alpine

print_success "Container temporaneo avviato"

# Step 2: Ottieni il certificato
print_step "Ottengo il certificato Let's Encrypt..."
docker run --rm \
    -v $(pwd)/ssl-certs:/etc/letsencrypt \
    -v $(pwd)/www-certbot:/var/www/certbot \
    certbot/certbot:latest \
    certonly --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN

# Verifica che il certificato sia stato creato
if [[ -f "ssl-certs/live/$DOMAIN/fullchain.pem" ]]; then
    print_success "Certificato ottenuto con successo!"
else
    print_error "Errore nell'ottenere il certificato"
    docker stop temp-nginx && docker rm temp-nginx
    exit 1
fi

# Step 3: Ferma il container temporaneo
print_step "Ferma container temporaneo..."
docker stop temp-nginx && docker rm temp-nginx
rm nginx-temp.conf
print_success "Container temporaneo rimosso"

# Step 4: Avvia il servizio con Let's Encrypt
print_step "Avvio servizio con certificati Let's Encrypt..."
docker-compose -f docker-compose.letsencrypt.yml up -d
print_success "Servizio avviato"

# Step 5: Setup cron per rinnovo automatico
print_step "Setup rinnovo automatico..."
echo "0 12 * * * docker run --rm -v $(pwd)/ssl-certs:/etc/letsencrypt -v $(pwd)/www-certbot:/var/www/certbot certbot/certbot:latest renew --quiet && docker-compose -f $(pwd)/docker-compose.letsencrypt.yml restart frontend" > /tmp/certbot-cron
crontab /tmp/certbot-cron
print_success "Rinnovo automatico configurato"

echo ""
print_success "🎉 Let's Encrypt configurato con successo!"
echo ""
echo "Il tuo sito ora è disponibile su:"
echo "  🌐 https://$DOMAIN (con certificato valido)"
echo ""
echo "Il certificato si rinnoverà automaticamente ogni giorno alle 12:00"
echo ""
print_warning "Ricorda di aprire le porte 80 e 443 sul firewall se necessario"
