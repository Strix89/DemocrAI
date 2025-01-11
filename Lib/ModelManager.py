import logging
from langchain_ollama import OllamaLLM

def _read_prompt_from_file(prompt_file: str) -> str:
    """
    Legge un prompt da un file e lo restituisce come stringa.

    :param prompt_file: Il percorso del file da cui leggere il prompt.
    :return: Il prompt letto dal file.
    """
    try:
        text = ""
        with open(prompt_file, "rb") as f:
            text = f.read().decode("utf-8")
            logging.info(f"Lettura del prompt dal file '{prompt_file}' eseguita con successo.")
            logging.info(f"Prompt letto: \n{text}\n")
            return text
    except Exception as e:
        logging.error(f"Errore durante la lettura del prompt dal file '{prompt_file}': {e}")
        return ""

class ModelManager:
    def __init__(self, model_name: str = None, path_prompt: str = None):
        """
        Inizializza il gestore del modello Ollama con la possibilità di configurare un prompt iniziale.

        :param model_name: Il nome del modello da utilizzare. Se None, verrà preso dal file .env.
        :param initial_prompt: Un prompt iniziale da usare come contesto per il modello.
        :param path_prompt: Il percorso del file da cui leggere il prompt iniziale.
        """
        self.model_name = model_name
        self.model = None
        self.context = ""
        self.initial_prompt = _read_prompt_from_file(path_prompt)
        self._initialize_model()
        self.prompt_template = """Regole: {prompt}\n---\nContesto fornito: {context}\n---\nDomanda: {query}"""
        

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

    async def invoke_model(self, input_text: str, context: str = None) -> str:
        """
        Invoca il modello Ollama con l'input fornito, includendo il contesto.

        :param input_text: Il testo da inviare al modello.
        :param context: Contesto da fornire al modello. Se None, verrà usato il contesto dell'oggetto.
        :return: La risposta del modello.
        """
        if not self.model:
            raise RuntimeError("Il modello non è stato inizializzato correttamente.")

        prompt = self.prompt_template.format(
            prompt=self.initial_prompt,
            context=self.context if context is None else context,
            query=input_text
        )

        logging.info(f"Prompt inviato al modello: \n{prompt}\n")

        try:
            logging.info(f"Prompt inviato al modello")
            result = self.model.invoke(input=prompt)
            logging.info("Richiesta al modello eseguita con successo.")
            return result
        except Exception as e:
            logging.error(f"Errore durante l'invocazione del modello: {e}")
            raise
    
    def add_context(self, new_context: str):
        """
        Aggiunge un contesto al contesto attuale.

        :param new_context: Contesto da aggiungere.
        """
        self.context += f"\n{new_context}"

    def get_context(self) -> str:
        """Restituisce il contesto attuale."""
        return self.context
    
    def get_initial_prompt(self) -> str:
        """Restituisce il prompt iniziale."""
        return self.initial_prompt