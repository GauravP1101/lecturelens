import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Models to try, best first. If one is busy, we move to the next.
MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]

question = "In one sentence, what is retrieval-augmented generation (RAG)?"

for model in MODELS:
    try:
        response = client.models.generate_content(model=model, contents=question)
        print(f"Gemini ({model}) says:")
        print(response.text)
        break  # it worked, so stop trying other models
    except errors.APIError as e:
        # 503 = servers busy, 429 = free-tier rate limit reached
        if e.code in (429, 503):
            print(f"{model} is busy (error {e.code}), trying the next model...")
        else:
            raise  # any other error is a real problem, so show it
else:
    # This part runs only if no model worked
    print("All models are busy right now. Wait a minute and run it again.")