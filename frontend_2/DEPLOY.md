# Deploy Guide - Forvard Frontend

## Configurazione

Il frontend è configurato per comunicare con l'API su `http://volare.unime.it:8443` come specificato in `src/config/api.js`.

## Build e Deploy

### 1. Build locale

```bash
# Build dell'immagine Docker
docker build -t forvard-frontend .

# Test locale
docker run -d -p 80:80 -p 443:443 --name forvard-frontend-test forvard-frontend
```

### 2. Script automatico

Utilizza lo script `deploy.sh` per un deploy automatizzato:

```bash
./deploy.sh
```

Lo script:
- Fa il build dell'immagine
- Permette di testare localmente
- Fa il tag per DockerHub
- Esegue il push
- Pulisce le immagini locali

### 3. Push manuale su DockerHub

```bash
# Tag per DockerHub
docker tag forvard-frontend:latest salvini/forvard-frontend:latest

# Login a DockerHub
docker login

# Push dell'immagine
docker push salvini/forvard-frontend:latest
```

### 4. Deploy in produzione

```bash
# Opzione 1: Docker run diretto
docker run -d \
  -p 80:80 \
  -p 443:443 \
  --name forvard-frontend \
  --restart unless-stopped \
  salvini/forvard-frontend:latest

# Opzione 2: Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

## Configurazione HTTPS

Il container è configurato per:
- **Porta 80**: Redirect automatico a HTTPS
- **Porta 443**: Servizio HTTPS con certificato self-signed

### Certificati in produzione

Per sostituire il certificato self-signed con uno reale:

1. Monta i certificati reali nel container:
```bash
docker run -d \
  -p 80:80 \
  -p 443:443 \
  -v /path/to/your/cert.crt:/etc/ssl/certs/nginx-selfsigned.crt \
  -v /path/to/your/private.key:/etc/ssl/private/nginx-selfsigned.key \
  --name forvard-frontend \
  salvini/forvard-frontend:latest
```

2. Oppure crea una nuova immagine con i certificati:
   - Modifica il Dockerfile
   - Copia i certificati invece di generarli
   - Rebuilda l'immagine

## Monitoraggio

### Health Check

Il container include un health check che verifica ogni 30 secondi:
```bash
docker ps  # Mostra lo stato di salute
```

### Logs

```bash
# Logs del container
docker logs forvard-frontend

# Logs in tempo reale
docker logs -f forvard-frontend
```

### Metriche di sistema

```bash
# Utilizzo risorse
docker stats forvard-frontend
```

## Configurazione Nginx

La configurazione Nginx include:
- Redirect HTTP → HTTPS
- Security headers
- Cache per file statici
- Supporto per React Router (SPA)
- Proxy opzionale per API

## Risoluzione problemi

### Problemi comuni

1. **Porta già in uso**:
   ```bash
   # Trova processo che usa la porta
   sudo lsof -i :80
   sudo lsof -i :443
   ```

2. **Certificato SSL**:
   - Il browser mostrerà un warning per il certificato self-signed
   - Puoi accettare l'eccezione per i test
   - In produzione usa certificati reali (Let's Encrypt)

3. **API non raggiungibile**:
   - Verifica che l'API sia in esecuzione su `volare.unime.it:8443`
   - Controlla i CORS se necessario

### Test di funzionamento

```bash
# Test HTTP (dovrebbe dare redirect 301)
curl -I http://localhost

# Test HTTPS (ignora certificato self-signed)
curl -k -I https://localhost

# Test contenuto
curl -k https://localhost
```

## Aggiornamenti

Per aggiornare il frontend:

1. Modifica il codice
2. Rebuilda l'immagine
3. Fai il push su DockerHub
4. Riavvia il container:
   ```bash
   docker pull salvini/forvard-frontend:latest
   docker stop forvard-frontend
   docker rm forvard-frontend
   docker run -d -p 80:80 -p 443:443 --name forvard-frontend salvini/forvard-frontend:latest
   ```

## Sicurezza

Il container è configurato con:
- User non-root
- Security headers HTTP
- SSL/TLS configurato
- No new privileges
- Limiti di risorse

Per ulteriori sicurezze in produzione, considera:
- Firewall configurato
- Certificati SSL validi
- Reverse proxy (Traefik, nginx-proxy)
- Monitoraggio e alerting
