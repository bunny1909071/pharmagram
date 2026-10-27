import json
import os
import time
from collections import defaultdict, deque
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from groq import APIError, Groq, RateLimitError
from pydantic import BaseModel, Field

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SUBJECTS = ["Pharmacology", "Pharmaceutics", "Pathology", "Clinical Pharmacy",
            "Medicinal Chemistry", "Pharmacognosy"]

app = FastAPI(title="Pharmgram")
_client = None
_hits = defaultdict(deque)


def client():
    global _client
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(503, "GROQ_API_KEY is not set. Add it as a secret in the Space settings.")
    if _client is None:
        _client = Groq(api_key=key)
    return _client


def throttle(req: Request, limit: int = 15):
    ip = (req.headers.get("x-forwarded-for") or (req.client.host if req.client else "x")).split(",")[0].strip()
    q = _hits[ip]
    now = time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many requests. Try again in a minute.")
    q.append(now)


def ask(messages, json_mode=False, max_tokens=900):
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        r = client().chat.completions.create(
            model=MODEL, messages=messages, temperature=0.4, max_tokens=max_tokens, **extra)
        return r.choices[0].message.content
    except RateLimitError:
        raise HTTPException(429, "The free AI limit was reached. Try again shortly.")
    except APIError as e:
        raise HTTPException(502, f"AI service error: {getattr(e, 'message', 'request failed')}")


SHAPES = {
    "notes": '{"title": string, "slides": [{"h": short heading, "p": 1-2 sentences}] with exactly 3 slides}',
    "quiz": '{"q": string, "o": 4 short options, "a": index 0-3 of the correct option, "ex": 1-2 sentence explanation}',
    "flash": '{"front": a question or term, "back": a concise answer}',
    "fact": '{"title": short heading, "p": one accurate, interesting sentence}',
}


def validate(kind, d):
    s = lambda v: isinstance(v, str) and v.strip()
    if kind == "notes":
        slides = [{"h": x["h"], "p": x["p"]} for x in d["slides"] if s(x.get("h")) and s(x.get("p"))][:4]
        assert s(d["title"]) and slides
        return {"title": d["title"], "slides": slides}
    if kind == "quiz":
        assert s(d["q"]) and s(d["ex"]) and len(d["o"]) == 4 and all(s(o) for o in d["o"])
        assert isinstance(d["a"], int) and 0 <= d["a"] <= 3
        return {"q": d["q"], "o": d["o"], "a": d["a"], "ex": d["ex"]}
    if kind == "flash":
        assert s(d["front"]) and s(d["back"])
        return {"front": d["front"], "back": d["back"]}
    assert s(d["title"]) and s(d["p"])
    return {"title": d["title"], "p": d["p"]}


class Gen(BaseModel):
    subject: str
    topic: str = Field(min_length=2, max_length=80)
    kind: Literal["notes", "quiz", "flash", "fact"]


@app.post("/api/generate")
def generate(g: Gen, req: Request):
    throttle(req)
    if g.subject not in SUBJECTS:
        raise HTTPException(400, "Unknown subject.")
    system = ("You write accurate, exam-oriented pharmacy study material for B.Pharm and Pharm.D students. "
              "Reply with one JSON object only, in this shape: " + SHAPES[g.kind] +
              ". Use standard textbook facts only. If you are unsure of a detail, leave it out.")
    out = ask([{"role": "system", "content": system},
               {"role": "user", "content": f"Subject: {g.subject}. Topic: {g.topic}."}], json_mode=True)
    try:
        return validate(g.kind, json.loads(out))
    except Exception:
        raise HTTPException(502, "The AI returned an unusable answer. Try again.")


class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class Tutor(BaseModel):
    context: str = Field(default="", max_length=2000)
    messages: list[Msg] = Field(min_length=1, max_length=12)


@app.post("/api/tutor")
def tutor(t: Tutor, req: Request):
    throttle(req)
    system = ("You are a friendly pharmacy tutor for B.Pharm and Pharm.D students in India. "
              "Explain clearly in simple language, under 150 words unless asked for more, with a short example when useful. "
              "This is for education only: if the question is about a real person's treatment or medicines, "
              "tell them to ask their doctor or pharmacist. If you are not sure, say so. "
              "The post the student is looking at:\n" + t.context)
    msgs = [{"role": "system", "content": system}] + [m.model_dump() for m in t.messages]
    return {"reply": ask(msgs, max_tokens=500)}


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL, "key_set": bool(os.getenv("GROQ_API_KEY"))}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
