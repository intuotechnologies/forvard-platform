# Guida all'Installazione di Docker e Portainer CE su Linux

Questa guida ti accompagnerà attraverso i passaggi essenziali per configurare un server Linux con Docker e un'istanza di Portainer Community Edition (CE) per gestire i tuoi container.

Questa è una versione semplificata e mirata, che include:
1.  Creazione di un nuovo utente con privilegi `sudo` (consigliato per la sicurezza).
2.  Installazione di Docker Engine.
3.  Installazione e avvio del container di Portainer CE.

---

## Prerequisiti

* Un server con una distribuzione Linux recente (es. Ubuntu, Debian, CentOS).
* Accesso al server come utente `root` o con un utente che ha già i privilegi `sudo`.

---

## Passaggio 1: Creazione di un Nuovo Utente (Consigliato)

Per motivi di sicurezza, è sconsigliato operare come utente `root`. Creiamo un nuovo utente e concediamogli i privilegi `sudo`.

Accedi al tuo server e, come **root**, esegui i seguenti comandi. Sostituisci `nuovoutente` con il nome che preferisci.

```bash
adduser nuovoutente
```
Segui le istruzioni per impostare la password e le altre informazioni.

Ora, aggiungi il nuovo utente al gruppo `sudo` per conferirgli i privilegi di amministratore.

**Per Debian/Ubuntu:**
```bash
usermod -aG sudo nuovoutente
```

**Per RHEL/CentOS (e derivate come AlmaLinux, Rocky Linux):**
```bash
usermod -aG wheel nuovoutente
```

Ora disconnettiti dal server e accedi di nuovo con le credenziali del `nuovoutente` appena creato. D'ora in poi, tutti i comandi che richiedono privilegi elevati andranno preceduti da `sudo`.

---

## Passaggio 2: Installazione di Docker Engine

Questi passaggi installeranno l'ultima versione di Docker Engine utilizzando lo script ufficiale, che è il metodo più semplice e veloce.

**a. Rimuovi eventuali vecchie versioni di Docker**
Questo comando previene possibili conflitti con pacchetti Docker preesistenti.

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
# Per sistemi basati su yum/dnf, il comando è simile:
# sudo dnf remove docker docker-client docker-client-latest docker-common docker-latest docker-latest-logrotate docker-logrotate docker-engine
```

**b. Installa Docker usando lo script di convenienza**
Questo script rileva la tua distribuzione e installa Docker in modo automatico.

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

**c. Permetti al tuo utente di eseguire Docker senza `sudo` (Post-installazione)**
Per evitare di dover scrivere `sudo` ogni volta che usi il comando `docker`, aggiungi il tuo utente al gruppo `docker`.

```bash
sudo usermod -aG docker ${USER}
```
**IMPORTANTE:** Affinché questa modifica abbia effetto, devi effettuare il **logout completo dal server e poi eseguire nuovamente il login**.

**d. Abilita e avvia il servizio Docker**
Assicurati che Docker parta automaticamente all'avvio del server.

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

**e. Verifica l'installazione**
Controlla che Docker funzioni correttamente eseguendo il container di test.

```bash
docker run hello-world
```
Se vedi un messaggio di benvenuto da Docker, l'installazione è andata a buon fine.

---

## Passaggio 3: Installazione di Portainer CE

Ora che Docker è operativo, possiamo installare Portainer. Useremo il metodo "Docker Standalone" raccomandato dalla documentazione ufficiale.

**a. Crea un volume Docker per i dati di Portainer**
Questo passaggio assicura che la configurazione di Portainer (utenti, endpoint, ecc.) sia persistente e non venga persa se il container viene rimosso e ricreato.

```bash
docker volume create portainer_data
```

**b. Scarica e avvia il container di Portainer Server**
Esegui questo comando per avviare Portainer. Stiamo usando il tag `:lts` (Long-Term Support) che garantisce maggiore stabilità rispetto a `:latest`.

```bash
docker run -d -p 8000:8000 -p 9443:9443 --name portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce:lts
```

Analizziamo il comando:
* `-d`: Esegue il container in background (detached mode).
* `-p 8000:8000`: Mappa la porta 8000 del server, usata da Portainer per la funzionalità "Edge Agent" (opzionale).
* `-p 9443:9443`: Mappa la porta 9443, che è la porta principale per accedere all'interfaccia web di Portainer tramite HTTPS.
* `--name portainer`: Assegna un nome facile da ricordare al container.
* `--restart=always`: Riavvia automaticamente il container se si ferma o al riavvio del server.
* `-v /var/run/docker.sock:/var/run/docker.sock`: Collega il socket Docker dell'host al container. Questo è il passaggio cruciale che permette a Portainer di gestire Docker.
* `-v portainer_data:/data`: Monta il volume che abbiamo creato prima nella directory `/data` del container, dove Portainer salva i suoi dati.
* `portainer/portainer-ce:lts`: Specifica l'immagine Docker da usare.

**c. Verifica che Portainer sia in esecuzione**
Puoi controllare lo stato dei tuoi container con il comando:

```bash
docker ps
```
Dovresti vedere il container `portainer` nella lista con lo stato `Up`.

---

## Passaggio 4: Accesso e Configurazione Iniziale

L'installazione è completa. Ora puoi accedere all'interfaccia web di Portainer.

1.  Apri il tuo browser e vai all'indirizzo del tuo server sulla porta 9443:
    ```
    https://INDIRIZZO_IP_DEL_SERVER:9443
    ```
    Sostituisci `INDIRIZZO_IP_DEL_SERVER` con l'IP pubblico o locale della tua macchina.

2.  **Avviso di Sicurezza del Browser:** Vedrai un avviso che indica che la connessione non è privata. Questo è normale. Portainer genera un certificato SSL autofirmato (self-signed) per proteggere la connessione. Clicca su "Avanzate" e poi su "Procedi" (o una dicitura simile) per continuare.

3.  **Creazione Utente Amministratore:** La prima schermata ti chiederà di creare il tuo utente amministratore. Scegli un nome utente e una password sicura, quindi clicca su "Create user".

Una volta creato l'utente, verrai portato alla dashboard di Portainer. L'installazione è conclusa e sei pronto a gestire il tuo ambiente Docker!