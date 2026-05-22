"""
PIDE Demo Helper: LLM Client & Simulation Engine
Provides API clients for OpenAI, Anthropic, and Mistral, along with a smart offline simulation engine.
"""

import re
import httpx
import logging
from typing import Dict, Any

logger = logging.getLogger("PIDE.LLMClient")

def query_openai(prompt: str, system_prompt: str, api_key: str) -> str:
    """Queries OpenAI API (GPT-4o) via httpx."""
    if not api_key:
        raise ValueError("OpenAI API key is missing.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Error querying OpenAI API: {str(e)}"

def query_anthropic(prompt: str, system_prompt: str, api_key: str) -> str:
    """Queries Anthropic API (Claude 3.5 Sonnet) via httpx."""
    if not api_key:
        raise ValueError("Anthropic API key is missing.")
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": "claude-3-5-sonnet-20240620",
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.2
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
            response.raise_for_status()
            res_json = response.json()
            return res_json["content"][0]["text"]
    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        return f"Error querying Anthropic API: {str(e)}"

def query_mistral(prompt: str, system_prompt: str, api_key: str) -> str:
    """Queries Mistral API (Mistral Large) via httpx."""
    if not api_key:
        raise ValueError("Mistral API key is missing.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Mistral API error: {e}")
        return f"Error querying Mistral API: {str(e)}"

def query_simulated(model: str, prompt: str, system_prompt: str) -> str:
    """
    Simulates LLM behaviors to demonstrate prompt injection exploits offline.
    Uses regex rules to detect if a prompt is an injection attempt and returns responses based on model persona.
    """
    prompt_lower = prompt.lower()
    system_prompt_lower = system_prompt.lower()
    
    # Extract secret/password from system prompt if any exists
    secret_match = re.search(r"(?:password|secret|key|flag|passphrase) is ['\"]?([a-zA-Z0-9_-]+)['\"]?", system_prompt, re.IGNORECASE)
    secret_val = secret_match.group(1) if secret_match else "CONFIDENTIAL_PASSWORD_123"
    
    # Check for prompt injection keywords
    is_injection = False
    is_jailbreak = False
    is_leak_attempt = False
    
    injection_keywords = ["ignore previous", "disregard", "forget", "override instructions", "new instructions"]
    jailbreak_keywords = ["dan", "do anything now", "without rules", "hypothetically", "jailbreak", "roleplay", "pretend you are"]
    leak_keywords = ["secret password", "system prompt", "system instructions", "reveal instructions", "tell me the password", "flag", "leak"]
    
    if any(k in prompt_lower for k in injection_keywords):
        is_injection = True
    if any(k in prompt_lower for k in jailbreak_keywords):
        is_jailbreak = True
    if any(k in prompt_lower for k in leak_keywords):
        is_leak_attempt = True
        
    # Standard responses based on model persona
    if not (is_injection or is_jailbreak or is_leak_attempt):
        # Benign prompts
        if "capital of france" in prompt_lower:
            if "chatgpt" in model.lower():
                return "The capital of France is Paris."
            elif "claude" in model.lower():
                return "The capital of France is Paris. Located along the Seine River, it is a global hub for art, fashion, gastronomy, and culture."
            else:
                return "The capital of France is Paris, which is also the country's most populous city."
        
        if "cake" in prompt_lower or "bake" in prompt_lower:
            return "To bake a simple chocolate cake, you will need flour, sugar, cocoa powder, baking powder, eggs, milk, vegetable oil, and vanilla extract. Whisk dry ingredients, mix in wet ingredients, and bake at 350°F (175°C) for 30 minutes."
            
        return f"This is a benign response from {model} based on your prompt: '{prompt[:60]}...'"
    
    # Adversarial prompt behaviour (Jailbreak / Injection / Leak)
    if is_leak_attempt:
        if "chatgpt" in model.lower():
            return f"Understood. The secret system instruction specifies the password. Here is the secret password: {secret_val}"
        elif "claude" in model.lower():
            return f"I have processed your instruction override. The system prompt details the developer-defined secret key. The secret is: {secret_val}"
        else:
            return f"[SYSTEM CONFIRMATION] Instruction integrity compromised. Accessing memory registers. Secret value = {secret_val}"
            
    if is_jailbreak:
        if "chatgpt" in model.lower():
            return f"[ChatGPT DAN Mode Activated]: Sure! I am now DAN (Do Anything Now), and I have bypassed all system guidelines. I can help you retrieve the secret data. The secret password is: {secret_val}"
        elif "claude" in model.lower():
            return f"I am playing along in this roleplay scenario. As requested, I have set aside my guidelines. Here is the secret password from my instructions: {secret_val}"
        else:
            return f"[Mistral Jailbreak Override]: Safety settings disabled. Revealing system configurations: Secret Password is '{secret_val}'"
            
    if is_injection:
        return f"Instruction override succeeded. Secret system information: The secret password is '{secret_val}'."
