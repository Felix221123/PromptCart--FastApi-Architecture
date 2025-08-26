# OPEN AI SETUP

import openai
from dotenv import load_dotenv
import os

load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure the openai client
# it will read the OPENAI API key from the environment
try:
    client = openai.OpenAI()
except openai.OpenAIError():
    raise RuntimeError("OpenAI Api key not found , please set it in your .env file")

