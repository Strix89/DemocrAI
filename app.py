import logging
import os
import sys
from dotenv import load_dotenv
from Lib.ModelManager import ModelManager
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from flask import Flask, request, jsonify, redirect, render_template, session, url_for # Import Flask to allow us to create a web server       

load_dotenv() # Load the .env file
logging.basicConfig(level=logging.INFO)
app = Flask(__name__) # Create a new web server

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
        return redirect(url_for("chatpage"))
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username").strip()
    password = request.form.get("password").strip()

    user = User.query.filter_by(username=username).first()
    if user is None or not user.check_password(password):
        return error_401("Accesso negato | Chi sei?", url_for("index"))
    else:
        return "Bella ZII"

    session["username"] = username
    return redirect(url_for("chatpage"))

def error_401(message, redirect_url):
    return render_template('errors/401.html', message=message, redirect_url=redirect_url)

if __name__ == "__main__":
    app.run(debug=True)
