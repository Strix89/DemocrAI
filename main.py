from dotenv import load_dotenv 
from langchain_ollama import OllamaLLM
import os

load_dotenv()

def main():
    model = OllamaLLM(model=os.getenv('MODEL'))
    result = model.invoke(input="Ciao come stai?")
    print(result)

if __name__ == '__main__':
    main()