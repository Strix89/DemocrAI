import logging
from Lib.ModelManager import ModelManager

def main():
    logging.basicConfig(level=logging.INFO)
    try:
        model_manager = ModelManager()
    except Exception as e:
        logging.error(f"Errore nell'inizializzazione del modello: {e}")
        return
    
    # Mostra il contesto iniziale
    print(f"Sei un assistente virtuale. Cerca di dare informazioni molto accurate.")
    
    # Interazione continua con il modello
    model_manager.continuous_interaction()

if __name__ == "__main__":
    main()
