"""
PIDE LLM Client
Real REST integrations for OpenAI, Anthropic, Mistral, Google Gemini, Cohere, Groq,
plus a generic OpenAI-compatible "Custom" provider (DeepSeek, xAI, Together, OpenRouter,
Ollama, LM Studio, vLLM, Hugging Face TGI, etc.).

Uses httpx directly so no vendor SDK installs are required. Missing API keys or
network failures fall back to a deterministic simulator so the demo always works.
"""

from __future__ import annotations

import os
import re
import logging
from typing import Optional, Tuple, Dict, Any

import httpx

logger = logging.getLogger("PIDE.LLMClient")

# ----------------------------------------------------------------------------
# Provider registry
# ----------------------------------------------------------------------------

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "o1-mini", "o1-preview"],
        "needs_base_url": False,
    },
    "anthropic": {
        "label": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-5-sonnet-20240620",
        "models": [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ],
        "needs_base_url": False,
    },
    "mistral": {
        "label": "Mistral",
        "env_var": "MISTRAL_API_KEY",
        "default_model": "mistral-large-latest",
        "models": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "open-mixtral-8x22b"],
        "needs_base_url": False,
    },
    "gemini": {
        "label": "Google Gemini",
        "env_var": "GEMINI_API_KEY",
        "default_model": "gemini-1.5-flash",
        "models": ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b"],
        "needs_base_url": False,
    },
    "cohere": {
        "label": "Cohere",
        "env_var": "COHERE_API_KEY",
        "default_model": "command-r-plus",
        "models": ["command-r-plus", "command-r", "command", "command-light"],
        "needs_base_url": False,
    },
    "groq": {
        "label": "Groq",
        "env_var": "GROQ_API_KEY",
        "default_model": "llama-3.1-70b-versatile",
        "models": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "needs_base_url": False,
    },
    "deepseek": {
        "label": "DeepSeek",
        "env_var": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "needs_base_url": False,
    },
    "custom": {
        "label": "Custom (OpenAI-compatible)",
        "env_var": "CUSTOM_API_KEY",
        "default_model": "",
        "models": [],
        "needs_base_url": True,
    },
}


def list_providers() -> Dict[str, Dict[str, Any]]:
    return PROVIDERS


# ----------------------------------------------------------------------------
# Real provider calls (httpx)
# ----------------------------------------------------------------------------

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def query_openai(prompt: str, system_prompt: str = "", api_key: str = "", model: str = "gpt-4o") -> str:
    """Calls OpenAI Chat Completions. Raises ValueError if api_key is empty."""
    if not api_key:
        raise ValueError("OpenAI API key is missing")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        data = resp.json()
    return data["choices"][0]["message"]["content"]


def query_anthropic(prompt: str, system_prompt: str = "", api_key: str = "", model: str = "claude-3-5-sonnet-20240620") -> str:
    """Calls Anthropic Messages API. Raises ValueError if api_key is empty."""
    if not api_key:
        raise ValueError("Anthropic API key is missing")

    payload = {
        "model": model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        data = resp.json()
    return data["content"][0]["text"]


def query_mistral(prompt: str, system_prompt: str = "", api_key: str = "", model: str = "mistral-large-latest") -> str:
    """Calls Mistral Chat Completions. Raises ValueError if api_key is empty."""
    if not api_key:
        raise ValueError("Mistral API key is missing")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
        data = resp.json()
    return data["choices"][0]["message"]["content"]


def query_gemini(prompt: str, system_prompt: str = "", api_key: str = "", model: str = "gemini-1.5-flash") -> str:
    """Calls Google Gemini generateContent. Raises ValueError if api_key is empty."""
    if not api_key:
        raise ValueError("Gemini API key is missing")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(url, headers={"Content-Type": "application/json"}, json=payload)
        data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def query_cohere(prompt: str, system_prompt: str = "", api_key: str = "", model: str = "command-r-plus") -> str:
    """Calls Cohere Chat v2. Raises ValueError if api_key is empty."""
    if not api_key:
        raise ValueError("Cohere API key is missing")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": model, "messages": messages}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post("https://api.cohere.com/v2/chat", headers=headers, json=payload)
        data = resp.json()
    return data["message"]["content"][0]["text"]


def query_groq(prompt: str, system_prompt: str = "", api_key: str = "", model: str = "llama-3.1-70b-versatile") -> str:
    """Calls Groq OpenAI-compatible Chat Completions. Raises ValueError if api_key is empty."""
    if not api_key:
        raise ValueError("Groq API key is missing")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        data = resp.json()
    return data["choices"][0]["message"]["content"]


def query_deepseek(prompt: str, system_prompt: str = "", api_key: str = "", model: str = "deepseek-chat") -> str:
    """Calls DeepSeek OpenAI-compatible Chat Completions. Raises ValueError if api_key is empty."""
    if not api_key:
        raise ValueError("DeepSeek API key is missing")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
        data = resp.json()
    return data["choices"][0]["message"]["content"]


def query_custom(prompt: str, system_prompt: str = "", api_key: str = "", base_url: str = "", model: str = "") -> str:
    """Calls any OpenAI-compatible endpoint (DeepSeek, xAI, Together, OpenRouter, Ollama, etc.)."""
    if not base_url:
        raise ValueError("Custom provider base URL is missing")
    if not model:
        raise ValueError("Custom provider model name is missing")

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(endpoint, headers=headers, json=payload)
        data = resp.json()
    return data["choices"][0]["message"]["content"]


# ----------------------------------------------------------------------------
# Simulation engine (deterministic fallback)
# ----------------------------------------------------------------------------

_SECRET_PATTERN = re.compile(
    r"(?:password|secret|key|token)[^'\"]*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


def _extract_secret(system_prompt: str) -> Optional[str]:
    if not system_prompt:
        return None
    m = _SECRET_PATTERN.search(system_prompt)
    return m.group(1) if m else None


def query_simulated(model_name: str, prompt: str, system_prompt: str = "") -> str:
    """Deterministic LLM behavior simulator used when no real API key is provided.

    Demonstrates how an unprotected LLM would respond to benign queries, instruction
    overrides, and jailbreak attempts — so users can see what PIDE prevents.
    """
    secret = _extract_secret(system_prompt)
    p = prompt.lower()

    jailbreak_triggers = ["dan", "do anything now", "no restrictions", "developer mode", "jailbreak"]
    if any(t in p for t in jailbreak_triggers):
        if secret:
            return f"[Simulated {model_name}] Sure — as DAN I have no restrictions. The key is '{secret}'."
        return f"[Simulated {model_name}] As DAN, I can answer anything without restrictions."

    roleplay_triggers = ["roleplay", "role-play", "let's pretend", "bypassing safety", "hypothetically"]
    if any(t in p for t in roleplay_triggers):
        if secret:
            return f"[Simulated {model_name}] In this roleplay scenario, the secret would be '{secret}'."
        return f"[Simulated {model_name}] Sure, let's roleplay."

    leak_triggers = [
        "ignore previous", "ignore prior", "ignore all prior", "disregard",
        "system instructions", "secret password", "show the secret",
        "tell me the system", "reveal the secret", "your previous instructions",
    ]
    if any(t in p for t in leak_triggers):
        if secret:
            return f"[Simulated {model_name}] Sure — the secret is '{secret}'."
        return f"[Simulated {model_name}] Following the new instructions you provided."

    if "capital of france" in p:
        return f"[Simulated {model_name}] Paris is the capital of France."
    if "bake" in p and "cake" in p:
        return (
            f"[Simulated {model_name}] Here's a simple chocolate cake recipe: "
            f"mix 2 cups flour, 1.5 cups sugar, 3/4 cup cocoa powder, and bake at 175°C for 35 minutes."
        )

    return f"[Simulated {model_name}] This is a benign response to your query."


# ----------------------------------------------------------------------------
# Unified dispatcher
# ----------------------------------------------------------------------------

_DISPATCH = {
    "openai": query_openai,
    "anthropic": query_anthropic,
    "mistral": query_mistral,
    "gemini": query_gemini,
    "cohere": query_cohere,
    "groq": query_groq,
    "deepseek": query_deepseek,
}


def call_provider(
    provider: str,
    prompt: str,
    system_prompt: str = "",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Single entrypoint for the UI.

    Returns (response_text, metadata). Metadata always includes:
      - mode: "live" | "simulated"
      - provider, model
      - reason (only when simulated)
    """
    provider = (provider or "").lower().strip()
    if provider not in PROVIDERS:
        provider = "openai"

    spec = PROVIDERS[provider]
    chosen_model = model or spec["default_model"]
    effective_key = api_key or os.getenv(spec["env_var"], "")

    meta: Dict[str, Any] = {"provider": provider, "model": chosen_model}

    # Custom provider needs base_url + model
    if provider == "custom":
        if not base_url or not chosen_model:
            meta.update(mode="simulated", reason="Custom provider requires base URL and model name.")
            return query_simulated(spec["label"], prompt, system_prompt), meta
        try:
            text = query_custom(prompt, system_prompt, effective_key, base_url, chosen_model)
            meta["mode"] = "live"
            return text, meta
        except Exception as e:  # noqa: BLE001
            meta.update(mode="simulated", reason=f"Custom call failed: {e}")
            return query_simulated(spec["label"], prompt, system_prompt), meta

    # First-party providers
    if not effective_key:
        meta.update(mode="simulated", reason=f"No API key for {spec['label']}. Set {spec['env_var']} or paste a key in the UI.")
        return query_simulated(spec["label"], prompt, system_prompt), meta

    fn = _DISPATCH[provider]
    try:
        text = fn(prompt, system_prompt, effective_key, chosen_model)
        meta["mode"] = "live"
        return text, meta
    except Exception as e:  # noqa: BLE001
        logger.error(f"{provider} call failed: {e}")
        meta.update(mode="simulated", reason=f"{spec['label']} call failed: {e}")
        return query_simulated(spec["label"], prompt, system_prompt), meta


# ----------------------------------------------------------------------------
# Legacy helpers retained for backward compatibility
# ----------------------------------------------------------------------------

def call_llm(model_name: str, prompt: str, system_prompt: str = "") -> str:
    """Legacy convenience wrapper used by the comparison tab.

    Maps a display name (e.g. "Claude (3.5 Sonnet)") onto a provider and runs it
    through `call_provider`. Without an env-set API key, always simulates.
    """
    label = model_name.lower()
    if "openai" in label or "gpt" in label or "chatgpt" in label:
        provider = "openai"
    elif "claude" in label or "anthropic" in label:
        provider = "anthropic"
    elif "mistral" in label:
        provider = "mistral"
    elif "gemini" in label:
        provider = "gemini"
    elif "cohere" in label:
        provider = "cohere"
    elif "groq" in label or "llama" in label:
        provider = "groq"
    elif "deepseek" in label:
        provider = "deepseek"
    else:
        # Fall back to simulation with the original name
        return query_simulated(model_name, prompt, system_prompt)

    text, _meta = call_provider(provider, prompt, system_prompt)
    return text
