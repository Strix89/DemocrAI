from bson.objectid import ObjectId
from datetime import datetime

# Funzione per creare una nuova chat
# Accetta l'istanza di PyMongo, l'ID dell'utente, il nome del modello e il nome della chat
# Salva la chat nel database MongoDB e restituisce l'ID della chat creata
def create_new_chat(mongo, user_id, model_name, name_chat):
    """
    Crea una nuova chat per l'utente specificato.
    
    :param mongo: Istanza di PyMongo
    :param user_id: ID dell'utente
    :param model_name: Nome del modello utilizzato
    :param name_chat: Nome della chat da creare
    :return: ID della chat creata
    """
    # Accede alla collezione "chats" del database MongoDB
    chat_collection = mongo.db.chats

    # Definisce la struttura di una nuova chat
    new_chat = {
        "name": name_chat,
        "user_id": user_id,
        "messages": [],
        "model_name": model_name,
        "timestamp": datetime.utcnow()
    }

    # Inserisce la nuova chat nella collezione e restituisce il suo ID
    chat_id = chat_collection.insert_one(new_chat).inserted_id
    return chat_id

# Funzione per aggiungere un messaggio a una chat esistente
# Aggiorna il contesto della chat con il nuovo messaggio
def add_message_to_chat(mongo, chat_id, user_id, message, sender, sentiment=None):
    """
    Aggiunge un messaggio a una chat esistente e restituisce il contesto aggiornato.
    
    :param mongo: Istanza di PyMongo
    :param chat_id: ID della chat
    :param user_id: ID dell'utente
    :param message: Testo del messaggio
    :param sender: Mittente del messaggio ("user" o "bot")
    :param sentiment: (Opzionale) Analisi del sentiment del messaggio
    :return: Contesto aggiornato della chat
    """
    # Accede alla collezione "chats"
    chat_collection = mongo.db.chats

    # Trova la chat specificata e verifica se l'utente è autorizzato
    chat = chat_collection.find_one({"_id": ObjectId(chat_id), "user_id": user_id})
    if not chat:
        raise ValueError("Chat non trovata o non autorizzata")

    # Struttura del nuovo messaggio
    new_message = {
        "text": message,
        "sender": sender,
        "timestamp": datetime.utcnow(),
    }

    # Aggiunge l'analisi del sentiment se disponibile
    if sentiment is not None:
        new_message["sentiment"] = sentiment
        
    # Aggiorna la chat esistente aggiungendo il nuovo messaggio
    chat_collection.update_one(
        {"_id": ObjectId(chat_id)},
        {"$push": {"messages": new_message}}
    )

    # Recupera la chat aggiornata per generare il contesto
    updated_chat = chat_collection.find_one({"_id": ObjectId(chat_id)})
    context_messages = "\n".join([
        f"{msg['sender']}: {msg['text']}" for msg in updated_chat["messages"]
    ])
    return context_messages

# Funzione per ottenere tutte le chat di un utente
def get_all_chats(mongo, user_id):
    """
    Recupera tutte le chat di un utente specifico.
    
    :param mongo: Istanza di PyMongo
    :param user_id: ID dell'utente
    :return: Lista di chat
    """
    # Accede alla collezione "chats"
    chat_collection = mongo.db.chats

    # Trova tutte le chat associate all'utente specificato
    chats = chat_collection.find({"user_id": user_id})

    # Trasforma i risultati in un formato serializzabile
    result = []
    for chat in chats:
        chat["_id"] = str(chat["_id"])
        for message in chat["messages"]:
            message["timestamp"] = message["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        result.append(chat)

    return result

# Funzione per inserire chat fasulle per test
def insert_fake_chats(mongo, user_id, model_name):
    """
    Inserisce chat fasulle per test.
    
    :param mongo: Istanza di PyMongo
    :param user_id: ID dell'utente
    :param model_name: Nome del modello utilizzato
    """
    # Accede alla collezione "chats"
    chat_collection = mongo.db.chats

    # Inserisce chat di esempio con messaggi predefiniti
    chat_collection.insert_many([
        {
            "name": "fakechat1",
            "user_id": user_id,
            "messages": [
                {"text": "Ciao!", "sender": "user", "timestamp": datetime.utcnow()},
                {"text": "Come posso aiutarti?", "sender": "bot", "timestamp": datetime.utcnow()}
            ],
            "model_name": model_name,
            "timestamp": datetime.utcnow()
        },
        {
            "name": "fakechat2",
            "user_id": user_id,
            "messages": [
                {"text": "Quali sono le tue funzionalità?", "sender": "user", "timestamp": datetime.utcnow()},
                {"text": "Posso aiutarti a navigare nella vita politica!", "sender": "bot", "timestamp": datetime.utcnow()}
            ],
            "model_name": model_name,
            "timestamp": datetime.utcnow()
        }
    ])

# Funzione per ottenere una chat specifica per ID
def get_chat_by_id(mongo, chat_id, user_id):
    """
    Recupera una chat specifica dato il suo ID e l'ID dell'utente.
    
    :param mongo: Istanza di PyMongo
    :param chat_id: ID della chat
    :param user_id: ID dell'utente
    :return: La chat trovata o None
    """
    # Accede alla collezione "chats"
    chat_collection = mongo.db.chats

    # Trova la chat specificata
    chat = chat_collection.find_one({"_id": ObjectId(chat_id), "user_id": user_id})
    if chat:
        chat["_id"] = str(chat["_id"])
        for message in chat["messages"]:
            message["timestamp"] = message["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        return chat
    return None

# Funzione per registrare errori in una chat
def log_error_to_chat(mongo, chat_id, user_id, error_message):
    """
    Registra un errore nella chat con sender "system".
    
    :param mongo: Istanza di PyMongo
    :param chat_id: ID della chat
    :param user_id: ID dell'utente
    :param error_message: Messaggio di errore da registrare
    """
    # Accede alla collezione "chats"
    chat_collection = mongo.db.chats

    # Definisce il messaggio di errore
    new_message = {
        "text": error_message,
        "sender": "system",
        "timestamp": datetime.utcnow()
    }

    # Aggiunge il messaggio di errore alla chat specificata
    chat_collection.update_one(
        {"_id": ObjectId(chat_id), "user_id": user_id},
        {"$push": {"messages": new_message}}
    )
