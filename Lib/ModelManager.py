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
        # Apertura del file in modalità binaria
        with open(prompt_file, "rb") as f:
            text = f.read().decode("utf-8")
            logging.info(f"Lettura del prompt dal file '{prompt_file}' eseguita con successo.")
            logging.info(f"Prompt letto: \n{text}\n")
            return text
    except Exception as e:
        logging.error(f"Errore durante la lettura del prompt dal file '{prompt_file}': {e}")
        return ""

class ModelManager:
    """
    Gestore del modello Ollama per fornire funzionalità AI avanzate.
    Gestisce l'inizializzazione, i prompt e il contesto per le richieste.
    """

    def __init__(self, model_name: str = None, path_prompt: str = None):
        """
        Inizializza il gestore del modello Ollama con la possibilità di configurare un prompt iniziale.

        :param model_name: Il nome del modello da utilizzare. Se None, verrà preso dal file .env.
        :param path_prompt: Il percorso del file da cui leggere il prompt iniziale.
        """
        self.model_name = model_name  # Nome del modello
        self.model = None  # Oggetto modello, inizialmente None
        self.context = ""  # Contesto da utilizzare con il modello
        self.initial_prompt = _read_prompt_from_file(path_prompt)  # Lettura del prompt iniziale
        self._initialize_model()  # Inizializza il modello

        # Template per il prompt con contesto
        self.prompt_template = """Regole: {prompt}\n---\nContesto fornito: {context}\n---\nDomanda: {query}"""

    def _initialize_model(self):
        """
        Inizializza il modello Ollama specificato.
        """
        if not self.model_name:
            raise ValueError("Il nome del modello non è stato specificato.")

        try:
            self.model = OllamaLLM(model=self.model_name)  # Inizializza il modello
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

        # Prepara il prompt combinando contesto e query
        prompt = self.prompt_template.format(
            prompt=self.initial_prompt,
            context=self.context if context is None else context,
            query=input_text
        )

        logging.info(f"Prompt inviato al modello: \n{prompt}\n")

        try:
            # Invio del prompt al modello
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
        """
        Restituisce il contesto attuale.
        
        :return: Contesto corrente del modello.
        """
        return self.context

    def get_initial_prompt(self) -> str:
        """
        Restituisce il prompt iniziale.

        :return: Il prompt iniziale letto dal file.
        """
        return self.initial_prompt