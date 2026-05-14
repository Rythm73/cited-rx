import os
from dotenv import load_dotenv
import anthropic

load_dotenv()  # reads .env into os.environ

# Confirm the key was loaded (don't print the actual key)
key = os.environ.get("ANTHROPIC_API_KEY")
if not key:
    raise RuntimeError("ANTHROPIC_API_KEY not found. Check that .env exists at project root.")
print(f"Key loaded: {key[:15]}...{key[-4:]}")  # prefix/suffix only

# Smallest possible test call
client = anthropic.Anthropic()  # auto-reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=50,
    messages=[
        {"role": "user", "content": "Say 'API works' and nothing else."}
    ],
)

print("Claude says:", response.content[0].text)
print(f"Tokens used: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
