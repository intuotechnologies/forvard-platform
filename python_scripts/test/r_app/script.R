# script.R
# Un semplice script R per il test

# Stampa un messaggio di saluto
print("Ciao da R dentro un container Docker, avviato tramite Portainer!")

# Stampa la data e l'ora corrente
current_time <- Sys.time()
print(paste("L'ora corrente è:", format(current_time, "%Y-%m-%d %H:%M:%S")))

# Esempio di creazione di un piccolo output (opzionale, più utile se si montano volumi)
# df <- data.frame(
#   ID = 1:3,
#   Valore = c("A", "B", "C")
# )
# print("DataFrame creato:")
# print(df)

# Messaggio di fine script
print("Lo script R è stato eseguito con successo.")
