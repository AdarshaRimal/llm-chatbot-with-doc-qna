from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load .env file variables into environment
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

models = genai.list_models()
print([m.name for m in models])
