"""Which light-model settings does Gemini accept? Four tiny requests, no DB.

Run inside the deployed container:  python scripts/check_light_model.py

"""

from openai import OpenAI

from agora.backend.infrastructure.config import settings as s

c = OpenAI(api_key=s.gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

for m in [s.gemini_light_model, "gemini-2.5-flash-lite"]:
    for kw in [{"reasoning_effort": "none"}, {}]:
        try:
            c.chat.completions.create(model=m, messages=[{"role": "user", "content": "Say ok"}], max_tokens=20, **kw)
            print(m, kw, "OK")
        except Exception as e:
            print(m, kw, "FAIL", str(e)[:120])
