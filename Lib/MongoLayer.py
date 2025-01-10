from bson.objectid import ObjectId
from datetime import datetime

def create_new_chat(mongo, user_id, model_name, name_chat):
    """
    Crea una nuova chat per l'utente specificato.
    
    :param mongo: Istanza di PyMongo
    :param user_id: ID dell'utente
    :param model_name: Nome del modello utilizzato
    :return: ID della chat creata
    """
    chat_collection = mongo.db.chats
    new_chat = {
        "name": name_chat,
        "user_id": user_id,
        "messages": [],
        "model_name": model_name,
        "timestamp": datetime.utcnow()
    }
    chat_id = chat_collection.insert_one(new_chat).inserted_id
    return chat_id

def add_message_to_chat(mongo, chat_id, user_id, message, sender, sentiment = None):
    """
    Aggiunge un messaggio a una chat esistente e restituisce il contesto aggiornato.
    
    :param mongo: Istanza di PyMongo
    :param chat_id: ID della chat
    :param user_id: ID dell'utente
    :param message: Testo del messaggio
    :param sender: Mittente del messaggio ("user" o "bot")
    :return: Contesto aggiornato della chat
    """
    chat_collection = mongo.db.chats
    chat = chat_collection.find_one({"_id": ObjectId(chat_id), "user_id": user_id})

    if not chat:
        raise ValueError("Chat non trovata o non autorizzata")

    new_message = {
        "text": message,
        "sender": sender,
        "timestamp": datetime.utcnow(),
    }
    if sentiment != None:
        new_message["sentiment"] = sentiment
        
    chat_collection.update_one(
        {"_id": ObjectId(chat_id)},
        {"$push": {"messages": new_message}}
    )

    # Genera il contesto aggiornato
    updated_chat = chat_collection.find_one({"_id": ObjectId(chat_id)})
    context_messages = "\n".join([
        f"{msg['sender']}: {msg['text']}" for msg in updated_chat["messages"]
    ])
    return context_messages

def get_all_chats(mongo, user_id):
    """
    Recupera tutte le chat di un utente specifico.
    
    :param mongo: Istanza di PyMongo
    :param user_id: ID dell'utente
    :return: Lista di chat
    """
    chat_collection = mongo.db.chats
    chats = chat_collection.find({"user_id": user_id})

    result = []
    for chat in chats:
        chat["_id"] = str(chat["_id"])
        for message in chat["messages"]:
            message["timestamp"] = message["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        result.append(chat)

    return result

def insert_fake_chats(mongo, user_id, model_name):
    """
    Inserisce chat fasulle per test.
    
    :param mongo: Istanza di PyMongo
    :param user_id: ID dell'utente
    :param model_name: Nome del modello utilizzato
    """
    chat_collection = mongo.db.chats
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

def get_chat_by_id(mongo, chat_id, user_id):
    chat_collection = mongo.db.chats
    chat = chat_collection.find_one({"_id": ObjectId(chat_id), "user_id": user_id})
    if chat:
        chat["_id"] = str(chat["_id"])
        for message in chat["messages"]:
            message["timestamp"] = message["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        return chat
    return None

def log_error_to_chat(mongo, chat_id, user_id, error_message):
    """
    Registra un errore nella chat con sender "system".
    """
    chat_collection = mongo.db.chats
    new_message = {
        "text": error_message,
        "sender": "system",
        "timestamp": datetime.utcnow()
    }
    chat_collection.update_one(
        {"_id": ObjectId(chat_id), "user_id": user_id},
        {"$push": {"messages": new_message}}
    )