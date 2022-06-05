#imports
import openai
import json
from gpt import GPT

#define variables
prompt = "Write a happy poem that is a Haiku, line endings should rhyme. The poem should contain the words flower and man "

#setup model
with open('gpt-key-JM.json') as f:
    data = json.load(f)
openai.api_key = data["API_KEY"]
gpt = GPT(engine="davinci-instruct-beta-v3",
         temperature=0.9,
         max_tokens=500)

# make API request
res = gpt.submit_request(prompt)
print(res)