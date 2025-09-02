# Imposta la directory di lavoro all'interno del container
WORKDIR /usr/src/app

# Copia lo script R dalla directory locale (r_app) alla directory di lavoro nel container
COPY script.R .

# Comando che verrà eseguito quando il container parte
# Esegue lo script R utilizzando Rscript
CMD ["Rscript", "./script.R"]
