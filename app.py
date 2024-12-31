import logging
import os
import sys
import secrets
import Lib.MongoLayer as mongoL
from functools import wraps
from dotenv import load_dotenv
from flask_pymongo import PyMongo
from Lib.ModelManager import ModelManager
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from flask import Flask, request, session, jsonify, redirect, render_template, url_for # Import Flask to allow us to create a web server 

app = Flask(__name__) # Create a new web server
load_dotenv() # Load the .env file
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH")
logging.basicConfig(level=logging.INFO)
app.secret_key = secrets.token_urlsafe(16) # Generate a random secret key for the session  

try:
    model_manager = ModelManager(os.environ.get("MODEL"))
except Exception as e:
    logging.error(f"Errore nell'inizializzazione del modello: {e}")
    sys.exit(1)

# Configurazione del database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Utilizza SQLite per semplicità
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inizializzazione di SQLAlchemy
db = SQLAlchemy(app)

# Configurazione di MongoDB
app.config["MONGO_URI"] = os.environ.get("MONGO_URI")
mongo = PyMongo(app)

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

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_authenticated"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        password = request.form.get("password").strip()
        if check_password_hash(ADMIN_PASSWORD_HASH, password):
            session["admin_authenticated"] = True
            return redirect(url_for("view_users"))
        else:
            return render_template("errors/error.html", message="Password amministratore errata", redirect_url=url_for("admin_login"))
    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_authenticated", None)
    return redirect(url_for("index"))

@app.route("/admin/add_user", methods=["GET", "POST"])
@admin_required
def add_user():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        first_name = request.form.get("first_name").strip()
        last_name = request.form.get("last_name").strip()
        age = request.form.get("age").strip()

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

@app.route("/admin/users")
@admin_required
def view_users():
    users = User.query.all()
    return render_template("view_users.html", users=users)

@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return redirect(url_for("view_users"))

#Login
@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("chat"))
    return render_template("index.html")

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

@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("name", None)
    session.pop("last_name", None)
    session.pop("age", None)
    return redirect(url_for("index"))

@app.route("/animation")
def animation():
    if "username" not in session:
        return redirect(url_for("index"))
    return render_template("animation.html")

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
def send_message():
    if "username" not in session:
        return jsonify({"error": "Non autorizzato"}), 401

    user_id = User.query.filter_by(username=session["username"]).first().id

    chat_id = request.json.get("chat_id")
    message = request.json.get("message")

    if not chat_id or not message:
        return jsonify({"error": "Parametri mancanti"}), 400

    try:
        mongoL.add_message_to_chat(mongo, chat_id, user_id, message, "user")
        response = model_manager.invoke_model(message)
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


if __name__ == "__main__":
    app.run(debug=True)
