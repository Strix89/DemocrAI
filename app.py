import logging
import os
import sys
import secrets
from dotenv import load_dotenv
from Lib.ModelManager import ModelManager
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from flask import Flask, request, session, jsonify, redirect, render_template, url_for # Import Flask to allow us to create a web server       

load_dotenv() # Load the .env file
logging.basicConfig(level=logging.INFO)
app = Flask(__name__) # Create a new web server
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

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

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
    return redirect(url_for("animation"))

@app.route("/logout")
def logout():
    session.pop("username", None)
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
    return "grevbe"

if __name__ == "__main__":
    app.run(debug=True)
