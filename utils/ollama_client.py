"""
FluxAI — Ollama LLM Client
Interfaces with a local Ollama instance to generate recovery playbooks.
"""
import requests
import json


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


def check_ollama():
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '').split(':')[0] for m in models]
            return True, model_names
        return False, []
    except (requests.ConnectionError, requests.Timeout):
        return False, []


def generate_playbook(customer_profile, risk_factors, model=DEFAULT_MODEL):
    """
    Generate a personalized recovery playbook using a local LLM.

    Args:
        customer_profile: dict with customer info (tenure, charges, contract, etc.)
        risk_factors: list of tuples (feature_name, shap_value) sorted by impact
        model: Ollama model name

    Returns:
        dict with 'email' and 'strategy' keys, or error dict
    """
    # Build a structured prompt
    prompt = _build_prompt(customer_profile, risk_factors)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 800,
                }
            },
            timeout=300,
        )

        if response.status_code == 200:
            result = response.json()
            text = result.get('response', '')
            return {
                'success': True,
                'content': text,
                'model': model,
            }
        else:
            return {
                'success': False,
                'error': f"Ollama returned status {response.status_code}",
            }

    except requests.ConnectionError:
        return {
            'success': False,
            'error': "Cannot connect to Ollama. Make sure it's running on localhost:11434.",
        }
    except requests.Timeout:
        return {
            'success': False,
            'error': "Ollama request timed out. Try a smaller model (e.g. llama3.2:1b) for faster CPU inference.",
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
        }


def _build_prompt(customer_profile, risk_factors):
    """Build a structured prompt for the LLM."""
    # Format risk factors
    risk_lines = []
    for feature, value in risk_factors[:8]:  # Top 8 factors
        direction = "↑ INCREASES risk" if value > 0 else "↓ DECREASES risk"
        risk_lines.append(f"  • {feature}: {direction} (impact: {abs(value):.3f})")

    risk_text = "\n".join(risk_lines)

    # Format customer profile
    profile_lines = []
    for key, val in customer_profile.items():
        profile_lines.append(f"  • {key}: {val}")
    profile_text = "\n".join(profile_lines)

    prompt = f"""You are an expert Customer Success Manager at a telecom company specializing in high-value retention. 
A customer has been flagged as {customer_profile.get('Churn Risk Score', 'HIGH RISK')} for churn by our AI system.

CUSTOMER PROFILE:
{profile_text}

AI-DETECTED RISK FACTORS (from SHAP analysis):
{risk_text}

Based on this analysis, please provide a comprehensive recovery playbook in Markdown format:

1. **RATIONALE & ANALYSIS** — Briefly explain *why* this customer is at risk based on the data. What is the core pain point? (1 paragraph)

2. **RECOVERY EMAIL** — A warm, highly personalized email (3-4 paragraphs). 
   - Use a subject line that isn't robotic.
   - Address the specific risk factors (e.g., if tenure is low, emphasize welcome benefits; if charges are high, mention value optimization).
   - Be empathetic and human.
   - Include a concrete, non-generic offer (e.g., "A complimentary basic security audit" or "15% loyalty adjustment for 3 months").

3. **RETENTION STRATEGY** — A step-by-step action plan (4-6 steps) for the account manager. Include specific talking points for a follow-up call.

4. **URGENCY LEVEL** — (Critical/High/Medium) with 1 sentence explaining the timing.

Format your response clearly with bold headers and clean bullet points."""

    return prompt


def get_available_models():
    """Get list of available Ollama models."""
    is_running, models = check_ollama()
    if is_running:
        return models
    return []
