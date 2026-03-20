import os

# Load API key from environment variable
# Set GROQ_API_KEY in your environment before running the app.
# Example (PowerShell): $env:GROQ_API_KEY = "your-key-here"
# Example (.env file):  GROQ_API_KEY=your-key-here

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
