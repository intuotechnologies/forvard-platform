# Configurazione Portainer su Server Linux

Questa guida ti accompagnerà attraverso i passaggi per:
1.  Creare un nuovo utente.
2.  Concedere i privilegi `sudo` al nuovo utente.
3.  Installare Docker Engine.
4.  Installare Portainer.

---

## 1. Creazione Nuovo Utente

Accedi al tuo server come utente **root**.

Sostituisci `nuovoutente` con il nome utente desiderato:

```bash
adduser nuovoutente
```

Ti verranno chieste alcune informazioni, inclusa una nuova password per l'utente.

---

## 2. Concessione Privilegi Sudo

Per aggiungere il nuovo utente al gruppo `sudo` (comune su distribuzioni Debian/Ubuntu) o `wheel` (comune su distribuzioni RHEL/CentOS), esegui il comando appropriato:

**Per Debian/Ubuntu:**

```bash
usermod -aG sudo nuovoutente
```

**Per RHEL/CentOS:**

```bash
usermod -aG wheel nuovoutente
```

Ora verifica passando al nuovo utente:

```bash
su - nuovoutente
```

D'ora in poi, esegui tutti i comandi successivi come `nuovoutente` utilizzando `sudo` quando necessario. Prova ad eseguire un comando con `sudo`, ad esempio:

```bash
sudo apt update  # Per Debian/Ubuntu
```

Se ti viene chiesta la password di `nuovoutente` e il comando viene eseguito, i privilegi `sudo` sono stati configurati correttamente.

---

## 3. Installazione di Docker Engine

Puoi installare Docker Engine utilizzando il repository ufficiale (metodo raccomandato per maggiore controllo) oppure tramite lo script di convenienza. Entrambi i metodi evitano l'uso di Snap.

### Metodo 1: Installazione tramite Repository Docker (Raccomandato)

Questi passaggi si basano sulle istruzioni ufficiali di Docker per Ubuntu. Adatta i comandi se utilizzi una distribuzione Linux differente.

**a. Disinstalla vecchie versioni (se presenti):**

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

**b. Configura il repository di Docker:**

```bash
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL [https://download.docker.com/linux/ubuntu/gpg](https://download.docker.com/linux/ubuntu/gpg) | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] [https://download.docker.com/linux/ubuntu](https://download.docker.com/linux/ubuntu) \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

**c. Installa Docker Engine:**

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Metodo 2: Installazione tramite Script di Convenienza

Questo script è fornito da Docker per semplificare l'installazione.

**a. Assicurati che `curl` sia installato:**

```bash
which curl
```
Se non è installato (il comando non restituisce un percorso), installalo:
```bash
sudo apt-get update # Per Debian/Ubuntu
sudo apt-get install curl # Per Debian/Ubuntu
# Adatta per altre distribuzioni, es. sudo dnf install curl per RHEL/CentOS
```

**b. Esegui lo script di installazione di Docker:**

```bash
curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
sudo sh get-docker.sh
```
Oppure, come da tua documentazione:
```bash
curl -fsSL [https://get.docker.com/](https://get.docker.com/) | sudo sh
```

### Passaggi Post-Installazione (per entrambi i metodi)

**d. Aggiungi il tuo utente al gruppo `docker` (Fortemente Raccomandato):**

Questo ti permette di eseguire comandi `docker` senza `sudo`. Sostituisci `nuovoutente` con il tuo nome utente effettivo se diverso.

```bash
sudo usermod -aG docker ${USER}
# O specifica l'utente:
# sudo usermod -aG docker nuovoutente
```

**Importante:** Dopo aver eseguito questo comando, devi **effettuare il logout e poi di nuovo il login** affinché le modifiche all'appartenenza al gruppo abbiano effetto. In alternativa, puoi attivare le modifiche per la sessione corrente con (ma il logout/login è più pulito):

```bash
newgrp docker
```
Tuttavia, `newgrp docker` apre una nuova shell. È generalmente meglio effettuare un logout completo e un nuovo login.

**e. Verifica l'installazione di Docker:**

```bash
docker run hello-world
```

Questo comando scaricherà un'immagine di test e la eseguirà in un container. Se tutto è corretto, vedrai un messaggio di conferma "Hello from Docker."

Se vedi un errore come "Cannot connect to the Docker daemon", prova a riavviare il servizio Docker (anche se lo script di installazione o l'installazione da repository dovrebbero avviarlo automaticamente):
```bash
sudo systemctl start docker
sudo systemctl enable docker # Per avviare Docker al boot
# Oppure
# sudo service docker restart
```

---

## 4. Installazione di Portainer

Portainer viene eseguito come un container Docker. Utilizzeremo il tag `lts` (Long Term Support) per l'immagine Portainer CE come suggerito dalla documentazione per una maggiore stabilità.

**a. Crea un volume per i dati persistenti di Portainer:**

```bash
docker volume create portainer_data
```

**b. Scarica ed esegui il container di Portainer Server:**

```bash
docker run -d -p 8000:8000 -p 9443:9443 --name portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce:lts
```

Spiegazione dei flag:
* `-d`: Esegue il container in background (detached mode).
* `-p 8000:8000`: Mappa la porta 8000 dell'host alla porta 8000 del container (per tunnel Edge Agent, se necessario).
* `-p 9443:9443`: Mappa la porta 9443 dell'host alla porta 9443 del container (per l'interfaccia utente di Portainer HTTPS). Per impostazione predefinita, Portainer genera e utilizza un certificato SSL autofirmato.
* `--name portainer`: Assegna un nome al container.
* `--restart=always`: Riavvia automaticamente il container se si arresta.
* `-v /var/run/docker.sock:/var/run/docker.sock`: Monta il socket Docker per permettere a Portainer di gestire Docker.
* `-v portainer_data:/data`: Monta il volume creato in precedenza per i dati persistenti di Portainer.
* `portainer/portainer-ce:lts`: Specifica l'immagine Docker di Portainer Community Edition (versione Long Term Support).

**Nota sull'HTTP (Porta 9000):**
Se per motivi di legacy hai bisogno che Portainer sia accessibile anche tramite HTTP sulla porta 9000, aggiungi `-p 9000:9000` al comando `docker run`. Ad esempio:
```bash
# Esempio con porta 9000 HTTP aggiuntiva
docker run -d -p 8000:8000 -p 9000:9000 -p 9443:9443 --name portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce:lts
```
Tuttavia, l'uso di HTTPS (porta 9443) è fortemente raccomandato per la sicurezza.

**c. Verifica che il container Portainer sia in esecuzione:**

```bash
docker ps
```
Dovresti vedere `portainer/portainer-ce:lts` nella lista dei container in esecuzione.

**d. Accedi all'interfaccia utente di Portainer:**

Apri il tuo browser web e naviga a:

```
https://INDIRIZZO_IP_DEL_TUO_SERVER:9443
```

Sostituisci `INDIRIZZO_IP_DEL_TUO_SERVER` con l'indirizzo IP effettivo del tuo server (o `localhost` se stai accedendo dalla macchina stessa).

La prima volta che accedi, ti verrà chiesto di creare un utente amministratore per Portainer. Segui le istruzioni a schermo.

---