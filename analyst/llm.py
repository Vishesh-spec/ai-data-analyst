"""Thin Anthropic wrapper with retries, streaming and robust JSON extraction."""
import json, logging, os, re
import anthropic

log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class ClaudeLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=key, max_retries=3)
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1500, json_mode: bool = False) -> str:
        try:
            r = self.client.messages.create(model=self.model, max_tokens=max_tokens,
                                            system=system, messages=messages)
            log.info("LLM usage in=%s out=%s", r.usage.input_tokens, r.usage.output_tokens)
            return "".join(b.text for b in r.content if b.type == "text")
        except anthropic.APIError as e:
            raise LLMError(f"LLM request failed: {e}") from e

    def stream(self, system: str, messages: list[dict], max_tokens: int = 1200):
        try:
            with self.client.messages.stream(model=self.model, max_tokens=max_tokens,
                                             system=system, messages=messages) as s:
                yield from s.text_stream
        except anthropic.APIError as e:
            raise LLMError(f"LLM request failed: {e}") from e


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError("Model did not return JSON.")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON from model: {e}") from e


class GeminiLLM:
    """Google Gemini (free tier via https://aistudio.google.com/apikey)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set.")
        from google import genai
        self.client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    @staticmethod
    def _contents(messages):
        return [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in messages]

    def _cfg(self, system, max_tokens, json_mode=False):
        from google.genai import types
        kw = {"response_mime_type": "application/json"} if json_mode else {}
        # thinking tokens share the output budget on 2.5 models, so keep it generous
        return types.GenerateContentConfig(system_instruction=system,
                                           max_output_tokens=max(max_tokens, 4096), **kw)

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1500, json_mode: bool = False) -> str:
        try:
            r = self.client.models.generate_content(model=self.model, contents=self._contents(messages),
                                                    config=self._cfg(system, max_tokens, json_mode))
            return r.text or ""
        except Exception as e:
            raise LLMError(f"Gemini request failed: {e}") from e

    def stream(self, system: str, messages: list[dict], max_tokens: int = 1200):
        try:
            for chunk in self.client.models.generate_content_stream(
                    model=self.model, contents=self._contents(messages), config=self._cfg(system, max_tokens)):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(f"Gemini request failed: {e}") from e


def detect_provider(key: str | None) -> str | None:
    if not key:
        return None
    key = key.strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith(("AIza", "AQ")):
        return "gemini"
    return None


def make_llm(provider: str = "auto", api_key: str | None = None, model: str | None = None):
    """Build an LLM client. Gemini by default; Claude only for sk-ant- keys or when chosen explicitly."""
    api_key = (api_key or "").strip() or None
    if provider == "auto":
        provider = "anthropic" if (api_key or "").startswith("sk-ant-") else "gemini"
    if provider == "gemini":
        return GeminiLLM(api_key, model)
    if provider == "anthropic":
        return ClaudeLLM(api_key, model)
    raise LLMError(f"Unknown provider: {provider}")
