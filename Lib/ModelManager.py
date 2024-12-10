from langchain_ollama import OllamaLLM
import os
import logging


class ModelManager:
    def __init__(self, model_name: str = None, initial_prompt: str = None):
        """
        Inizializza il gestore del modello Ollama con la possibilità di configurare un prompt iniziale.

        :param model_name: Il nome del modello da utilizzare. Se None, verrà preso dal file .env.
        :param initial_prompt: Un prompt iniziale da usare come contesto per il modello.
        """
        self.model_name = model_name
        self.model = None
        self.context = initial_prompt or "Sei un assistente virtuale molto competente. Devi rispondere in italiano"
        self._initialize_model()

    def _initialize_model(self):
        """Inizializza il modello Ollama."""
        if not self.model_name:
            raise ValueError("Il nome del modello non è stato specificato.")
        
        try:
            self.model = OllamaLLM(model=self.model_name)
            logging.info(f"Modello '{self.model_name}' caricato con successo.")
        except Exception as e:
            logging.error(f"Errore durante il caricamento del modello: {e}")
            raise

    def set_context(self, new_context: str):
        """
        Modifica il contesto attuale del modello.

        :param new_context: Nuovo contesto o prompt da fornire al modello.
        """
        self.context = new_context
        logging.info(f"Contesto aggiornato: {self.context}")

    def invoke_model(self, input_text: str) -> str:
        """
        Invoca il modello Ollama con l'input fornito, includendo il contesto.

        :param input_text: Il testo da inviare al modello.
        :return: La risposta del modello.
        """
        if not self.model:
            raise RuntimeError("Il modello non è stato inizializzato correttamente.")
        
        input_with_context = f"{self.context}\n{input_text}"

        try:
            result = self.model.invoke(input=input_with_context)
            logging.info("Richiesta al modello eseguita con successo.")
            return result
        except Exception as e:
            logging.error(f"Errore durante l'invocazione del modello: {e}")
            raise

    def continuous_interaction(self):
        """Gestisce un'interazione continua con l'utente."""
        print("Avvio della sessione di interazione continua. Digita 'exit' per terminare.")
        while True:
            user_input = input("Tu: ")
            if user_input.lower() == "exit":
                print("Sessione terminata.")
                break
            response = self.invoke_model(user_input)
            print(f"{self.model_name.capitalize()}: {response}")
    
    def add_context(self, new_context: str):
        """
        Aggiunge un contesto al contesto attuale.

        :param new_context: Contesto da aggiungere.
        """
        self.context += f"\n{new_context}"
        logging.info(f"Contesto aggiornato: {self.context}")

    def get_context(self) -> str:
        """Restituisce il contesto attuale."""
        return self.context