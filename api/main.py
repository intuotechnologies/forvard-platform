import os
from dotenv import load_dotenv

# Carica variabili d'ambiente prima di importare l'app
load_dotenv()

# Importa l'app dopo aver caricato l'ambiente
from app.main import app

# Questo blocco verrà eseguito solo se il file viene eseguito direttamente
if __name__ == "__main__":
    import uvicorn
    
    # Porta configurabile tramite variabile d'ambiente
    PORT = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=PORT,
        reload=True,  # Imposta a False in produzione
        log_level="info"
    )
