import openai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
openai.api_key =api_key



completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about recursion in programming."}
        ]
    )

print(completion['choices'][0]['message']['content'])
    
