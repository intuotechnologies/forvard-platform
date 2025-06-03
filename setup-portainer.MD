# Configurazione Server Completa: Utente, Docker, Nginx Proxy Manager & Portainer

Questa guida ti accompagnerà attraverso i passaggi per configurare un nuovo server Linux da zero, includendo:
1.  Creazione di un nuovo utente con privilegi `sudo`.
2.  Installazione di Docker Engine.
3.  Creazione di una rete Docker dedicata.
4.  Setup di Nginx Proxy Manager (NPM) e Portainer utilizzando Docker Compose.
5.  Configurazione base di NPM per l'accesso a Portainer con SSL (linee guida).

---

## Prerequisiti

* **Accesso root:** Devi poter accedere al tuo server come utente `root` o con un utente con privilegi `sudo` per i comandi iniziali.
* **DNS Configurato (Consigliato):** Se intendi usare Nginx Proxy Manager per esporre servizi con nomi di dominio e SSL (come Portainer), assicurati che il tuo nome di dominio (es. `portainer.tuodominio.com`) punti all'indirizzo IP pubblico del tuo server. Questo passaggio è cruciale per la generazione dei certificati SSL.

---

## 1. Creazione Nuovo Utente e Concessione Privilegi Sudo

Accedi al tuo server come utente **root**.

Sostituisci `nuovoutente` con il nome utente desiderato:

```bash
adduser nuovoutente
```

Ti verranno chieste alcune informazioni, inclusa una nuova password per l'utente.

Ora, concedi i privilegi `sudo` al nuovo utente.

**Per Debian/Ubuntu:**

```bash
usermod -aG sudo nuovoutente
```

**Per RHEL/CentOS (e derivate come AlmaLinux, Rocky Linux):**

```bash
usermod -aG wheel nuovoutente
```

Verifica passando al nuovo utente:

```bash
su - nuovoutente
```

D'ora in poi, esegui tutti i comandi successivi come `nuovoutente`, utilizzando `sudo` quando necessario. Prova un comando:

```bash
sudo apt update # Per Debian/Ubuntu
# Oppure
sudo dnf check-update # Per RHEL/CentOS
```

Se ti viene chiesta la password di `nuovoutente` e il comando ha successo, sei pronto.

---

## 2. Installazione di Docker Engine

Questi passaggi ti aiuteranno a installare l'ultima versione di Docker Engine.

**a. Rimuovi vecchie installazioni Docker (se presenti):**
Questo comando è più completo per assicurare una pulizia efficace.

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove -y $pkg; done
# Per sistemi basati su dnf/yum, adatta il comando di rimozione:
# for pkg in docker docker-client docker-client-latest docker-common docker-latest docker-latest-logrotate docker-logrotate docker-engine podman-docker containerd runc; do sudo dnf remove -y $pkg; done
```

**b. Installa Docker usando lo script ufficiale:**
Questo è il metodo suggerito dal tuo collega e spesso il più semplice.

```bash
curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
sudo sh get-docker.sh
```

**c. Aggiungi il tuo utente al gruppo `docker`:**
Questo ti permette di eseguire comandi `docker` senza `sudo`.

```bash
sudo groupadd docker # Questo comando potrebbe restituire un errore se il gruppo esiste già, è normale.
sudo usermod -aG docker ${USER} # ${USER} è una variabile che contiene il tuo nome utente corrente.
```

**Importante:** Devi effettuare il **logout e poi di nuovo il login** affinché le modifiche all'appartenenza al gruppo abbiano effetto. In alternativa, per la sessione corrente (ma il logout/login è più pulito):

```bash
newgrp docker
```

**d. Verifica l'installazione di Docker:**

```bash
docker run hello-world
```

Dovresti vedere un messaggio "Hello from Docker."

**e. Abilita Docker all'avvio:**

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

---

## 3. Creazione Rete Docker Dedicata

Creiamo una rete Docker personalizzata che verrà usata da NPM e Portainer.

```bash
sudo docker network create mycloud
```

---

## 4. Setup Nginx Proxy Manager & Portainer con Docker Compose

Useremo Docker Compose per gestire Nginx Proxy Manager (NPM) e Portainer.

**a. Installa Docker Compose (se non già installato con lo script Docker):**
Lo script `get.docker.com` dovrebbe installare anche Docker Compose. Verifica con `docker compose version`. Se non è presente:

```bash
sudo apt-get update
sudo apt-get install docker-compose-plugin # O la versione standalone se preferisci
```
Per altre distribuzioni, consulta la documentazione ufficiale di Docker.

**b. Crea un file `docker-compose.yml`:**
Crea una cartella per la tua configurazione, ad esempio `mkdir ~/docker_setup && cd ~/docker_setup`. Poi, crea un file chiamato `npm.yml` (come nel suggerimento del collega) o `docker-compose.yml` al suo interno con il seguente contenuto:

```yaml
# ~/docker_setup/npm.yml o ~/docker_setup/docker-compose.yml
version: '3.8' # Specifica una versione di compose file syntax

services:
  npm:
    image: 'jc21/nginx-proxy-manager:latest'
    container_name: npm
    restart: unless-stopped
    ports:
      # Queste sono le porte pubbliche HTTP/S
      - '80:80'
      - '443:443'
      # Questa è la porta per l'interfaccia Admin di NPM
      - '81:81'
      # Queste porte addizionali erano nel tuo esempio, potrebbero servire per usi futuri (es. specifici servizi proxati)
      - '21115:21115'
      - '21116:21116'
      - '21116:21116/udp'
      - '21117:21117'
      - '21118:21118'
      - '21119:21119'
    environment:
      DISABLE_IPV6: 'true' # Impostazione del tuo collega, disabilita IPv6 per NPM
    volumes:
      # Abbiamo usato sottocartelle per evitare conflitti tra i dati di npm e portainer
      - ./data_npm:/data 
      - ./data_letsencrypt:/etc/letsencrypt
    networks:
      - mycloud_network # Usa il nome della rete definito sotto

  portainer:
    image: 'portainer/portainer-ce:latest' # Il tuo collega usa 'latest', considera 'lts' per maggiore stabilità
    container_name: portainer
    restart: unless-stopped
    # privileged: true # ATTENZIONE: Vedi nota di sicurezza sotto!
    volumes:
      - ./data_portainer:/data # Dati persistenti di Portainer
      - '/var/run/docker.sock:/var/run/docker.sock' # Permette a Portainer di gestire Docker
    networks:
      - mycloud_network # Usa il nome della rete definito sotto

networks:
  mycloud_network: # Definizione interna al compose file della rete
    external: true # Specifica che la rete è stata creata esternamente
    name: mycloud # Nome della rete Docker creata al Passaggio 3

```

**Note importanti sul file `docker-compose.yml`:**

* **Volumi:** Ho modificato i percorsi dei volumi da `./data` a `./data_npm`, `./data_letsencrypt` e `./data_portainer`. Questo per assicurare che NPM e Portainer salvino i loro dati in sottocartelle distinte all'interno della directory dove si trova il file `docker-compose.yml` (es. `~/docker_setup/data_npm/`), evitando conflitti. Quando avvii docker-compose, queste cartelle verranno create se non esistono.
* **`privileged: true` per Portainer (Commentato):** La configurazione del tuo collega includeva `privileged: true` per Portainer. L'ho commentato di default perché **concede al container Portainer accesso completo e illimitato al tuo sistema host, il che rappresenta un rischio significativo per la sicurezza.** Va usato solo se strettamente necessario e comprendendone appieno le implicazioni. Per la maggior parte degli usi, l'accesso al socket Docker (`/var/run/docker.sock`) è sufficiente. Se decidi di usarlo, rimuovi `#` dalla riga `privileged: true`.
* **Immagine Portainer:** Il tuo collega usa `portainer/portainer-ce:latest`. Per ambienti di produzione, potresti considerare `portainer/portainer-ce:lts` (Long Term Support) per una maggiore stabilità.

**c. Avvia i servizi con Docker Compose:**
Assicurati di essere nella stessa cartella del file `npm.yml` (o `docker-compose.yml`).

```bash
docker compose -f npm.yml up -d 
# Se hai chiamato il file docker-compose.yml, puoi omettere -f npm.yml
# docker compose up -d 
```

I container verranno scaricati ed avviati in background.

---

## 5. Configurazione di Nginx Proxy Manager (NPM)

Ora che NPM è in esecuzione, puoi configurarlo.

**a. Accedi all'interfaccia di Amministrazione di NPM:**
Apri il tuo browser e vai a:

```
http://INDIRIZZO_IP_DEL_TUO_SERVER:81
```

Le credenziali di default sono:
* Email: `admin@example.com`
* Password: `changeme`

Ti verrà chiesto di cambiare la password al primo accesso.

**b. Configura SSL (Let's Encrypt):**
NPM semplifica l'ottenimento di certificati SSL.
1.  Vai nella sezione "SSL Certificates".
2.  Clicca su "Add SSL Certificate" e scegli "Let's Encrypt".
3.  Inserisci il tuo dominio/sottodominio (es. `portainer.tuodominio.com` o un wildcard come `*.tuodominio.com`).
4.  Per la validazione DNS (necessaria per i certificati wildcard), dovrai fornire le credenziali API del tuo provider DNS (es. Cloudflare, come menzionato dal tuo collega). Segui le istruzioni di NPM e la documentazione del tuo provider DNS. Il link fornito dal tuo collega per Cloudflare può essere una buona risorsa: `https://blog.jverkamp.com/2023/03/27/wildcard-lets-encrypt-certificates-with-nginx-proxy-manager-and-cloudflare/`
5.  Abilita "Force SSL" e "HTTP/2 Support" se desiderato.

**c. Crea un Proxy Host per Portainer:**
Una volta ottenuto il certificato SSL:
1.  Vai su "Hosts" -> "Proxy Hosts".
2.  Clicca su "Add Proxy Host".
3.  **Details Tab:**
    * **Domain Names:** Inserisci il dominio che vuoi usare per Portainer (es. `portainer.tuodominio.com`).
    * **Scheme:** `http` (NPM gestirà l'HTTPS esternamente).
    * **Forward Hostname / IP:** Inserisci il nome del container Portainer come definito nel `docker-compose.yml`, ovvero `portainer`.
    * **Forward Port:** `9000` (Portainer di default ascolta sulla porta 9000 HTTP e 9443 HTTPS all'interno della rete Docker; NPM comunicherà con Portainer sulla sua porta HTTP interna 9000, o 9443 se configuri Portainer per usare HTTPS internamente e scegli HTTPS come scheme qui). Per semplicità con il proxy, `http` e `portainer` con porta `9000` è comune. *Nota: Portainer CE dalla versione 2.19 usa la porta 9000 per HTTP e 9443 per HTTPS (con un certificato autofirmato di default)*. Dato che Portainer nel compose non espone porte all'host, ci riferiamo alla porta *interna* del container Portainer, che è 9000 (HTTP) o 9443 (HTTPS).
4.  **SSL Tab:**
    * Seleziona il certificato SSL che hai creato nel menu a tendina "SSL Certificate".
    * Abilita "Force SSL".
    * (Opzionale) Abilita "HTTP/2 Support".
5.  Salva.

---

## 6. Accesso a Portainer

Ora dovresti essere in grado di accedere a Portainer tramite l'URL HTTPS che hai configurato in NPM:

```
[https://portainer.tuodominio.com](https://portainer.tuodominio.com)
```

Al primo accesso, Portainer ti chiederà di creare un utente amministratore.

Se riscontri problemi di timeout o Portainer non è raggiungibile, verifica i log dei container:
```bash
docker logs npm
docker logs portainer
```
E assicurati che le impostazioni del Proxy Host in NPM siano corrette (specialmente il Forward Hostname/IP `portainer` e la Forward Port `9000` o `9443` a seconda della tua configurazione e di come vuoi che NPM comunichi con Portainer). Se usi `portainer` e `9443` come forward, potresti dover abilitare "Accept self-signed certificates" in NPM se Portainer usa il suo certificato autofirmato internamente. Usare `http` e `9000` per la comunicazione interna tra NPM e Portainer è spesso più semplice.

---