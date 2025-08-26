# Multi-stage Dockerfile per React frontend in produzione

# Stage 1: Build dell'applicazione React
FROM node:20-alpine AS builder

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file di configurazione delle dipendenze
COPY package*.json ./

# Installa le dipendenze (incluse quelle di dev per il build)
RUN npm config set registry https://registry.npmjs.org/ && \
    npm install --no-audit --no-fund --timeout=600000

# Copia il codice sorgente
COPY . .

# Build dell'applicazione per produzione
RUN npm run build

# Stage 2: Nginx per servire l'applicazione
FROM nginx:alpine

# Installa openssl per generare certificati self-signed se necessario
RUN apk add --no-cache openssl

# Rimuovi la configurazione di default di nginx
RUN rm /etc/nginx/conf.d/default.conf

# Copia la configurazione personalizzata di nginx
COPY nginx.conf /etc/nginx/conf.d/

# Copia i file buildati da React
COPY --from=builder /app/build /usr/share/nginx/html

# Crea directory per i certificati SSL
RUN mkdir -p /etc/ssl/certs

# Genera certificato self-signed (per test - sostituire con certificati reali in produzione)
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/nginx-selfsigned.key \
    -out /etc/ssl/certs/nginx-selfsigned.crt \
    -subj "/C=IT/ST=Sicily/L=Messina/O=Unime/OU=IT/CN=volare.unime.it"

# Esponi le porte 80 (HTTP) e 443 (HTTPS)
EXPOSE 80 443

# Avvia nginx
CMD ["nginx", "-g", "daemon off;"]
