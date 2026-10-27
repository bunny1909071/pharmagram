# Pharmgram

An Instagram-style pharmacy learning app with AI-generated study posts and an AI tutor (Groq).

Run locally:

    pip install -r requirements.txt
    GROQ_API_KEY=your_key uvicorn app:app --reload

Edit `static/content.json` to change the built-in posts.
Optional environment variable: `GROQ_MODEL`.
