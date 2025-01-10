import os
import shutil
import sys
import spacy
import numpy as np
import re
import logging
from typing import List
from ChunkType import ChunkType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma as constructorChroma
from sklearn.metrics.pairwise import cosine_similarity

def _get_env_bool(value: str):
    return str(value).lower() == "true"

def _compute_sentiment(text: str) -> dict:
    """
    Calcola il sentiment di un testo utilizzando VADER.
    """
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment = sentiment_analyzer.polarity_scores(text)
    return sentiment

class OllamaEmbedder:
    """
    Classe che:
      1) Carica PDF da una cartella (DirectoryLoader + PyPDFLoader)
      2) Suddivide i PDF in chunk con RecursiveCharacterTextSplitter
      3) Calcola embedding con Ollama (modello 'snowflake-arctic-embed2')
      4) Salva tutto in un database Chroma (opzionalmente ricreandolo da zero)
    """

    def __init__(
        self,
        docs_folder: str,
        model_name: str = "snowflake-arctic-embed2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        persist_directory: str = "chroma_db",
        pipeline_spacy = "it_core_news_lg",
        chunking_type: str = "semantic"
    ):
        """
        :param docs_folder:    Cartella da cui caricare i PDF e file di testo.
        :param model_name:    Nome del modello Ollama per gli embedding. Default: 'snowflake-arctic-embed2'.
        :param chunk_size:    Numero massimo di caratteri per chunk.
        :param chunk_overlap: Overlap (in caratteri) fra chunk consecutivi.
        :param persist_directory: Directory in cui salvare il DB di Chroma.
        :param glob_pattern:  Pattern di ricerca dei file PDF, default "**/*.pdf, "**/*.txt".
        :param pipeline_spacy: Nome del modello di Spacy da utilizzare. Default: 'it_core_news_lg'.
        :param chunking_type: Tipo di suddivisione del testo. Default: 'semantic'.
        """
        self.docs_folder = docs_folder
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.chunking_type = ChunkType.from_string(chunking_type)

        # Istanzia il loader per i PDF
        self.pdfloader = DirectoryLoader(
            self.docs_folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        self.textloader = DirectoryLoader(
            self.docs_folder,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )

        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.nlp = spacy.load(pipeline_spacy)

        # Strategia di suddivisione testo
        if self.chunking_type == ChunkType.SEMANTIC:
            self.text_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile", #standard_deviatio, interquartile
            )
        elif self.chunking_type == ChunkType.RECURSIVE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                add_start_index=True,
                separators=["\n\n", "\n", " ", ""]  # Delimitatori naturali (paragrafi, frasi, spazi)
            )
        #else:

    def _preprocess_text(self, text: str) -> str:
        """
        Rimuove stopwords e trasforma le parole in lemma.
        Ritorna la stringa pre-processata.
        """
        doc = self.nlp(text.lower())
        tokens = []
        for token in doc:
            if not token.is_stop and not token.is_space:
                tokens.append(token.lemma_)
        return " ".join(tokens)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        - Riduce a uno solo i multipli 'a capo'
        - Riduce a uno solo i multipli spazi
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\n+', '\n', text)
        text = text.strip()
        return text

    def _pipelineSpacy(self, text: str) -> str:
        """
        Esegue il pipeline di Spacy su un testo.
        """
        doc = self.nlp(text)
        return doc
    
    def load_documents(self) -> List[Document]:
        """
        Carica tutti i PDF dalla cartella specificata e li restituisce
        come lista di Document (uno per PDF, ma con tutto il testo concatenato).
        """
        pdf_documents = self.pdfloader.load()
        text_documents = self.textloader.load()
        return pdf_documents + text_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Suddivide i PDF in chunk in base alla strategia definita da
        self.chunking_type, restituendo una lista di Document.
        """
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def save_to_chroma(self, chunks: List[Document]) -> Chroma:
        """
        Crea un database Chroma a partire dalla lista di Document chunked.
        Per default cancella la cartella esistente e ne crea una nuova.
        """
        # Se vuoi ricreare da zero il DB, cancella prima la cartella
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        # Creiamo un nuovo DB con i documenti chunked
        db = constructorChroma.from_documents(
            chunks, embedding=self.embeddings, persist_directory=self.persist_directory
        )
        return db

    def process_folder_and_store(self, preprocess: bool = False) -> Chroma:
        """
        Metodo principale che:
          1) Carica i PDF
          2) Li divide in chunk
          3) Crea il DB Chroma
          4) Stampa qualche informazione di debug
        """
        documents = self.load_documents()

        if preprocess: #and self.chunking_type not in [ChunkType.SEMANTIC] ChunkType.AGENTIC #TODO
            for doc in documents:
                doc.page_content = self.normalize_whitespace(doc.page_content)
                original_text = doc.page_content
                doc.metadata["original_text"] = original_text
                preprocessed = self._preprocess_text(original_text)
                doc.page_content = preprocessed

        chunks = self.split_documents(documents)

        logging.info(f"Caricati {len(documents)} Documents.")
        logging.info(f"Suddivisi in {len(chunks)} chunk totali.\n")

        db = self.save_to_chroma(chunks)
        logging.info(f"Salvati {len(chunks)} chunk su Chroma in '{self.persist_directory}'.\n")
        return db

    def _get_embeddings_of_phrase(self, phrase: str, preprocess: bool = False) -> List[float]:
        """
        Restituisce gli embedding di una frase usando OllamaEmbeddings.
        """
        return self.embeddings.embed_query(phrase) if not preprocess else self.embeddings.embed_query(self._preprocess_text(phrase))

    def _compute_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calcola la similarità coseno tra due embedding.
        :param embedding1: Primo embedding.
        :param embedding2: Secondo embedding.
        :return: Valore di similarità coseno.
        """
        vector1 = np.array(embedding1).reshape(1, -1)
        vector2 = np.array(embedding2).reshape(1, -1)
        similarity = cosine_similarity(vector1, vector2)[0][0]
        return similarity

class OllamaDBRetriever:
    def __init__(self, persist_directory: str, model_name: str = "snowflake-arctic-embed2", pipeline_spacy = "it_core_news_lg"):
        """
        Inizializza un retriever per il DB di Chroma.
        :param persist_directory: Directory in cui è salvato il DB di Chroma.
        :param model_name: Nome del modello Ollama per gli embedding. Default: 'snowflake-arctic-embed2'.
        :param pipeline_spacy: Nome del modello di Spacy da utilizzare. Default: 'it_core_news_lg'.
        """
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.chromadb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        self.nlp = spacy.load(pipeline_spacy)
    
    def query(self, query_text: str, preprocess: bool = False, k: int = 5, similarity_threshold: float = 0.2):
        """
        Esegue una query sul DB di Chroma e restituisce i risultati.
        :param query_text: Testo da cercare.
        :param preprocess: Se True, pre-elabora il testo con _preprocess_text.
        :param k: Numero di risultati da restituire.
        :param similarity_threshold: Soglia di similarità minima per considerare un risultato.
        """
        if preprocess:
            doc = self.nlp(query_text.lower())
            tokens = []
            for token in doc:
                if not token.is_stop and not token.is_space:
                    tokens.append(token.lemma_)
            query_text = " ".join(tokens)
        results = self.chromadb.similarity_search_with_relevance_scores(query_text, k=k)
        if len(results) == 0 or results[0][1] <= similarity_threshold:
            return []
        return results

# Esempio di utilizzo
if __name__ == "__main__":
    # Specifica la cartella dei PDF (può essere assoluta o relativa)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DOCS_FOLDER = os.path.join(script_dir, "documents")

    embedder = OllamaEmbedder(
        docs_folder=DOCS_FOLDER,
        model_name="snowflake-arctic-embed2",
        chunk_size=512,
        chunk_overlap=50,
        persist_directory="chroma_db",
        chunking_type="semantic"
    )

    # 1) Carichiamo e suddividiamo i PDF
    # 2) Creiamo/ricreiamo il DB di Chroma
    #chroma_db = embedder.process_folder_and_store(True)

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHROMA_FOLDER = os.path.join(script_dir, "chroma_db")
    embeddingFunction = OllamaEmbeddings(model = "snowflake-arctic-embed2")
    db = Chroma(persist_directory=CHROMA_FOLDER, embedding_function=embeddingFunction)
    # Search the DB.
    query_text = "art. 2 costituzione"
    query_text = embedder._preprocess_text(query_text)
    results = db.similarity_search_with_relevance_scores(query_text, k=4)
    if len(results) == 0 or results[0][1] < 0.1:
        print(f"Unable to find matching results.")
        sys.exit(1)
    for result in results:
        print(result[0])
        print(f"Relevance Score: {result[1]}")
        print(f"Document Text: {result[0].page_content}")
        print(f"Original Document Text: {result[0].metadata['original_text']}")
        print(f"Document ID: {result[0].metadata["source"]}")
        print()
