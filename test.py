import openai 
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']

# log the api key
print(openai_api_key)










