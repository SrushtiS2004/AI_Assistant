import sys
import os

# Add project root to sys.path to allow imports when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from groq import Groq
from config.config import GROQ_API_KEY

def generate_response(prompt: str, mode: str = "Concise") -> str:
    """
    Generate a response using Groq's llama3-8b-8192 model.
    Mode can be 'Concise' (~100 tokens) or 'Detailed' (~500 tokens).
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        if mode == "Concise":
            max_tokens = 100
        elif mode == "Detailed":
            max_tokens = 500
        else:
            max_tokens = 100 # default to concise
            
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful AI Interview Preparation Assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Example usage for testing
if __name__ == "__main__":
    print(generate_response("Hello, how are you?", "Concise"))
