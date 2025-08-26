# Sample main file

import os
from fastapi import FastAPI
from .api import router
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()
app.include_router(router)
