import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Resolve a config value: Streamlit Cloud secrets → env var → default."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


# Configuration — all configurable through environment variables.
# OLLAMA_API_URL : Base URL for Ollama Cloud (or local fallback).
# OLLAMA_MODEL   : Model name to use (e.g. "gemma3:12b", "llama3.1:8b", "qwen3:8b").
# OLLAMA_API_KEY : Bearer token for Ollama Cloud authentication.
OLLAMA_API_URL = _get_secret("OLLAMA_API_URL", "https://ollama.com")
OLLAMA_MODEL = _get_secret("OLLAMA_MODEL", "gemma4:31b")
OLLAMA_API_KEY = _get_secret("OLLAMA_API_KEY", "")


class LLMService:
    """Modular LLM service layer — the sole gateway between NEXUS and the LLM.

    Consumers call `LLMService.is_available()` and `LLMService.complete(...)`.
    Swapping the backing model only requires changing environment variables;
    no code changes are needed anywhere else in NEXUS.
    """

    @staticmethod
    def _headers() -> dict:
        """Build request headers, including auth when an API key is configured."""
        headers = {"Content-Type": "application/json"}
        if OLLAMA_API_KEY:
            headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
        return headers

    @staticmethod
    def is_available() -> bool:
        """Fast health check — verifies the Ollama service is reachable and the
        configured model is available."""
        try:
            response = requests.get(
                f"{OLLAMA_API_URL}/api/tags",
                headers=LLMService._headers(),
                timeout=5,
            )
            if response.status_code == 200:
                models = [
                    model["name"] for model in response.json().get("models", [])
                ]
                # Check both exact match and base-name match (e.g. "gemma3:12b"
                # might be listed as "gemma3:12b" or just "gemma3").
                return (
                    OLLAMA_MODEL in models
                    or any(m.startswith(OLLAMA_MODEL.split(":")[0]) for m in models)
                )
            return False
        except requests.exceptions.RequestException:
            return False

    @staticmethod
    def complete(prompt: str, schema: dict = None, timeout: int = 120) -> str:
        """Send a prompt to the LLM and return the response text (or parsed
        JSON when a schema is requested).

        Uses the Ollama /api/chat endpoint which works identically for both
        local Ollama and Ollama Cloud — the only difference is the base URL
        and auth header.

        Args:
            prompt:  The full prompt string.
            schema:  If provided, the model is asked to respond in JSON format.
                     Pass ``{"type": "json"}`` for free-form JSON, or a full
                     JSON-schema dict for structured output.
            timeout: Request timeout in seconds (default 120 for cloud inference).

        Returns:
            The model's response as a string, or a parsed dict when schema is set.

        Raises:
            Exception with a user-friendly message on any failure.
        """
        if not LLMService.is_available():
            raise Exception("LLM service is currently unavailable. Check your OLLAMA_API_KEY and network connection.")

        # Build payload using the /api/chat messages format.
        payload: dict = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        if schema:
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/api/chat",
                headers=LLMService._headers(),
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()

            # /api/chat returns {"message": {"role": "assistant", "content": "..."}}
            result = response.json().get("message", {}).get("content", "")

            if schema:
                # Strip any markdown code-fence wrapping the model might add.
                clean = result.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[-1]  # drop ```json line
                if clean.endswith("```"):
                    clean = clean.rsplit("```", 1)[0]
                return json.loads(clean.strip())

            return result

        except requests.exceptions.Timeout:
            raise Exception("LLM request timed out. The cloud service may be under heavy load — try again shortly.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"LLM request failed: {str(e)}")
        except json.JSONDecodeError:
            raise Exception("LLM returned invalid JSON structure.")
