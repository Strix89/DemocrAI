import logging
import warnings
import os
import sys
import secrets
import Lib.MongoLayer as mongoL
from Lib.OllamaTools import OllamaEmbedder, OllamaDBRetriever, _compute_sentiment, _get_env_bool
from functools import wraps
from dotenv import load_dotenv
from flask_pymongo import PyMongo
from Lib.ModelManager import ModelManager
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from flask import Flask, request, session, jsonify, redirect, render_template, url_for

# Caricamento delle variabili di ambiente dal file .env
load_dotenv()

# Configurazione della logica globale
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Ignora alcuni avvisi di categoria UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Creazione di un'applicazione Flask
app = Flask(__name__)

# Generazione di una chiave segreta per le sessioni
app.secret_key = secrets.token_urlsafe(16)

# Controllo della variabile di ambiente per costruire il database, se necessario
if _get_env_bool(os.environ.get("BUILD_DB")):
    try:
        logging.info("Costruzione del database dei documenti con Chroma...")
        logging.info("Attenzione! Questa operazione potrebbe richiedere molto tempo.")
        logging.info("Attenzione! Il database verrà sovrascritto. Non interrompere il processo.")

        # Creazione di un oggetto per calcolare embedding e salvare il database
        embedder = OllamaEmbedder(
            docs_folder=os.environ.get("DOCUMENTS_PATH"),
            model_name=os.environ.get("EMBEDDING_MODEL"),
            chunk_size=os.environ.get("CHUNK_SIZE"),
            chunk_overlap=os.environ.get("CHUNK_OVERLAP"),
            persist_directory=os.environ.get("CHROMA_DB_PATH"),
            pipeline_spacy=os.environ.get("PIPELINE_SPACY"),
            chunking_type=os.environ.get("CHUNKING_TYPE")
        )

        # Elaborazione dei documenti e salvataggio del database
        chroma_db = embedder.process_folder_and_store(preprocess=_get_env_bool(os.environ.get("PREPROCESS_TEXT")))
        logging.info("VectorsStoreDB costruito con successo.")
    except Exception as e:
        logging.error(f"Errore durante la costruzione del database: {e}")
        sys.exit(1)

# Inizializzazione del retriever per interrogare il database costruito
try:
    logging.info("Inizializzazione del Retriever...")
    retriever = OllamaDBRetriever(
        os.environ.get("CHROMA_DB_PATH"),
        os.environ.get("EMBEDDING_MODEL"),
        os.environ.get("PIPELINE_SPACY")
    )
    logging.info("Retriever inizializzato")
except Exception as e:
    logging.error(f"Errore nell'inizializzazione del Retriever: {e}")
    sys.exit(1)

# Inizializzazione del gestore del modello per gestire richieste linguistiche
try:
    logging.info("Inizializzazione del modello con relativo prompt iniziale...")
    model_manager = ModelManager(os.environ.get("MODEL"), os.environ.get("LLM_PROMPT_PATH"))
except Exception as e:
    sys.exit(1)

# Configurazione del database SQLAlchemy per gestire utenti
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inizializzazione di SQLAlchemy
db = SQLAlchemy(app)

# Configurazione di MongoDB per la gestione delle chat
app.config["MONGO_URI"] = os.environ.get("MONGO_URI")
mongo = PyMongo(app)

# Definizione della classe User per rappresentare gli utenti
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    first_name = db.Column(db.String(80), nullable=False)
    last_name = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

# Decoratore per proteggere le route amministrative
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_authenticated"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function

# Route per il login amministrativo
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        password = request.form.get("password").strip()
        if check_password_hash(os.environ("ADMIN_PASSWORD_HASH"), password):
            session["admin_authenticated"] = True
            return redirect(url_for("view_users"))
        else:
            return render_template("errors/error.html", message="Password amministratore errata", redirect_url=url_for("admin_login"))
    return render_template("admin_login.html")

# Route per il logout amministrativo
@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_authenticated", None)
    return redirect(url_for("index"))

# Route per aggiungere nuovi utenti
@app.route("/admin/add_user", methods=["GET", "POST"])
@admin_required
def add_user():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        first_name = request.form.get("first_name").strip()
        last_name = request.form.get("last_name").strip()
        age = request.form.get("age").strip()

        # Convalida dei dati inseriti
        if not all([username, password, first_name, last_name, age]):
            return render_template("errors/400.html", message="Tutti i campi sono obbligatori", redirect_url=url_for("add_user"))

        if User.query.filter_by(username=username).first():
            return render_template("errors/400.html", message="Username già esistente", redirect_url=url_for("add_user"))

        try:
            age = int(age)
            if age <= 0:
                raise ValueError
        except ValueError:
            return render_template("errors/400.html", message="Età deve essere un numero positivo", redirect_url=url_for("add_user"))

        # Creazione e salvataggio dell'utente
        new_user = User(
            username=username,
            first_name=first_name,
            last_name=last_name,
            age=age
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("view_users"))
    return render_template("add_user.html")

# Route per visualizzare gli utenti
@app.route("/admin/users")
@admin_required
def view_users():
    users = User.query.all()
    return render_template("view_users.html", users=users)

# Route per eliminare un utente
@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return redirect(url_for("view_users"))

# Route principale per il login
@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("chat"))
    return render_template("index.html")

# Route per il login utente
@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username").strip()
    password = request.form.get("password").strip()

    user = User.query.filter_by(username=username).first()
    if user is None or not user.check_password(password):
        return render_template("errors/401.html", message = "Accesso negato | Chi sei?", redirect_url = url_for("index"))

    session["username"] = username
    session["name"] = user.first_name
    session["last_name"] = user.last_name
    session["age"] = user.age
    return redirect(url_for("animation"))

# Route per il logout utente
@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("name", None)
    session.pop("last_name", None)
    session.pop("age", None)
    return redirect(url_for("index"))

# Route per l'animazione di benvenuto
@app.route("/animation")
def animation():
    if "username" not in session:
        return redirect(url_for("index"))
    return render_template("animation.html")

# Route per la pagina della chat
@app.route("/chat")
def chat():
    if "username" not in session:
        return redirect(url_for("index"))
    return render_template("chat.html")

# API per creare una nuova chat
@app.route("/api/new_chat", methods=["POST"])
def create_chat():
    if "username" not in session:
        return jsonify({"error": "Non autorizzato"}), 401

    user_id = User.query.filter_by(username=session["username"]).first().id

    model_manager.set_context("")

    chat_name = request.json.get("message")[:20] if request.json.get("message") else "Chat senza nome"
    try:
        chat_id = mongoL.create_new_chat(mongo, user_id, os.environ.get("MODEL"), chat_name)
        return jsonify({"system":
                       {"chat_id": str(chat_id), "message": "Chat creata con successo"}}
                       )
    except Exception as e:
        error_message = str(e)
        mongoL.log_error_to_chat(mongo, None, user_id, error_message)
        return jsonify({"system": {"error": error_message}}), 500

# API per inviare un messaggio alla chat
@app.route("/api/send_message", methods=["POST"])
async def send_message():
    if "username" not in session:
        return jsonify({"error": "Non autorizzato"}), 401

    user_id = User.query.filter_by(username=session["username"]).first().id

    chat_id = request.json.get("chat_id")
    message = request.json.get("message")

    if not chat_id or not message:
        return jsonify({"error": "Parametri mancanti"}), 400

    sentiment = _compute_sentiment(message)

    model_manager.set_context("")

    try:
        mongoL.add_message_to_chat(mongo, chat_id, user_id, message, "user", sentiment)
        result = await retriever.query(
            message,
            _get_env_bool(os.environ.get("PREPROCESS_TEXT")), 
            int(os.environ.get("TOP_K")),
            float(os.environ.get("SIMILARITY"))
        )
        if result and len(result) != 0:
            if result[0][0].metadata.get("original_text", None) is not None:
                model_manager.add_context("\n".join(
                    [f'{doc[0].metadata["original_text"]} || Fonte: {doc[0].metadata["source"]}' for doc in result])
                )
            else:
                model_manager.add_context("\n".join(
                    [f'{doc[0].page_content} || Fonte: {doc[0].metadata["source"]}' for doc in result])
                )
        response = await model_manager.invoke_model(message)
        mongoL.add_message_to_chat(mongo, chat_id, user_id, response, "bot")
        return jsonify({"model": {"response": response}})
    except Exception as e:
        error_message = str(e)
        mongoL.log_error_to_chat(mongo, chat_id, user_id, error_message)
        return jsonify({"system": {"error": error_message}}), 500

# API per recuperare tutte le chat
@app.route("/api/chats", methods=["GET"])
def get_chats():
    if "username" not in session:
        return jsonify({"system": {"error": "non autorizzato"}}), 401

    user_id = User.query.filter_by(username=session["username"]).first().id

    try:
        chats = mongoL.get_all_chats(mongo, user_id)
        return jsonify(chats)
    except Exception as e:
        return jsonify({"system": {"error": str(e)}}), 500

# API per recuperare i messaggi di una chat
@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat_messages(chat_id):
    if "username" not in session:
        return jsonify({"error": "Non autorizzato"}), 401

    user_id = User.query.filter_by(username=session["username"]).first().id

    try:
        chat = mongoL.get_chat_by_id(mongo, chat_id, user_id)
        if not chat:
            return jsonify({"error": "Chat non trovata"}), 404
        return jsonify(chat)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# API per restituire l'URL base dell'API
@app.route('/api_base')
def api_base():
    return jsonify({"api_base_url": url_for('index', _external=True)})



app.run(debug=True, host="0.0.0.0", port=5000)
