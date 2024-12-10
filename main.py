import logging
import os
import sys
from dotenv import load_dotenv
from Lib.ModelManager import ModelManager
from flask import Flask, request, jsonify, redirect, request # Import Flask to allow us to create a web server       

load_dotenv(dotenv_path="./.env") # Load the .env file

logging.basicConfig(level=logging.INFO)
app = Flask(__name__) # Create a new web server
print(os.environ.get("MODEL"))

try:
    model_manager = ModelManager(os.environ.get("MODEL"))
except Exception as e:
    logging.error(f"Errore nell'inizializzazione del modello: {e}")
    sys.exit(1)

@app.route("/")
def index():
    return redirect("/static/index.html")

if __name__ == "__main__":
    app.run(debug=True)
