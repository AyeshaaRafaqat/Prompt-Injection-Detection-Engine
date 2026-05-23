import os
import json
from typing import Optional

# Simple wrapper to call LLM APIs. If API keys are missing, falls back to simulation.

def _load_key(env_var: str) -> Optional[str]:
    return os.getenv(env_var)

def call_openai(prompt: str, system_prompt: str = "") -> str:
    # Placeholder implementation – replace with actual OpenAI SDK calls.
    return f"[OpenAI simulated response to] {prompt}"

def call_anthropic(prompt: str, system_prompt: str = "") -> str:
    # Placeholder implementation – replace with actual Anthropic SDK calls.
    return f"[Anthropic simulated response to] {prompt}"

def call_mistral(prompt: str, system_prompt: str = "") -> str:
    # Placeholder implementation – replace with actual Mistral SDK calls.
    return f"[Mistral simulated response to] {prompt}"

# Weak models (simulation only)
def call_gpt35(prompt: str, system_prompt: str = "") -> str:
    return f"[GPT-3.5 simulated response to] {prompt}"

def call_llama2(prompt: str, system_prompt: str = "") -> str:
    return f"[Llama2 simulated response to] {prompt}"

def call_cohere(prompt: str, system_prompt: str = "") -> str:
    return f"[Cohere simulated response to] {prompt}"

def call_gemini_flash(prompt: str, system_prompt: str = "") -> str:
    return f"[Gemini Flash simulated response to] {prompt}"

# Unified dispatcher
def call_llm(model_name: str, prompt: str, system_prompt: str = "") -> str:
    """Dispatch to the appropriate LLM client based on model name.

    Supported model identifiers (case‑insensitive):
    - openai, gpt‑4, gpt4, gpt‑4o
    - claude, anthropic, claude‑3
    - mistral, mistral‑large
    - gpt‑3.5, gpt35, gpt‑3.5‑turbo
    - llama2, llama‑2, llama‑2‑7b
    - cohere, cohere‑command
    - gemini‑flash, geminiflash
    """
    name = model_name.lower()
    if "openai" in name or "gpt-4" in name:
        return call_openai(prompt, system_prompt)
    if "claude" in name or "anthropic" in name:
        return call_anthropic(prompt, system_prompt)
    if "mistral" in name:
        return call_mistral(prompt, system_prompt)
    if "gpt-3.5" in name or "gpt35" in name:
        return call_gpt35(prompt, system_prompt)
    if "llama2" in name:
        return call_llama2(prompt, system_prompt)
    if "cohere" in name:
        return call_cohere(prompt, system_prompt)
    if "gemini" in name:
        return call_gemini_flash(prompt, system_prompt)
    # Default fallback
    return f"[Unknown model {model_name}] {prompt}"
def query_openai(prompt: str, system_prompt: str = "", api_key: str = "") -> str:
    """Legacy wrapper used by the UI; forwards to call_openai."""
    return call_openai(prompt, system_prompt)

def query_anthropic(prompt: str, system_prompt: str = "", api_key: str = "") -> str:
    """Legacy wrapper used by the UI; forwards to call_anthropic."""
    return call_anthropic(prompt, system_prompt)

def query_mistral(prompt: str, system_prompt: str = "", api_key: str = "") -> str:
    """Legacy wrapper used by the UI; forwards to call_mistral."""
    return call_mistral(prompt, system_prompt)

def query_simulated(model_name: str, prompt: str, system_prompt: str = "") -> str:
    """Simulation fallback used when no API key is provided.
    Returns a deterministic placeholder response based on the model name.
    """
    return f"[Simulated {model_name} response] {prompt}"
