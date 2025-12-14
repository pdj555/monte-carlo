"""OpenAI-powered narrative summaries for simulation output.

This module is optional: it is only used when the CLI is invoked with
``--ai-summary`` and a valid ``OPENAI_API_KEY`` is available.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import pandas as pd

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIConfigurationError(RuntimeError):
    """Raised when OpenAI settings are missing or invalid."""


class OpenAIRequestError(RuntimeError):
    """Raised when an OpenAI API request fails."""


def _get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIConfigurationError(
            "OPENAI_API_KEY is required when --ai-summary is enabled."
        )
    return api_key


def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise OpenAIRequestError(
            f"OpenAI request failed ({exc.code}): {error_body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise OpenAIRequestError(f"OpenAI request failed: {exc.reason}") from exc

    try:
        parsed: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OpenAIRequestError(f"OpenAI returned invalid JSON: {exc}") from exc

    return parsed


def generate_ai_summary(
    *,
    ticker: str,
    summary: pd.Series,
    simulation_model: str,
    days: int,
    scenarios: int,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    timeout_seconds: float = 30.0,
) -> str:
    """Generate a concise narrative summary for a single ticker simulation."""

    api_key = _get_openai_api_key()
    base_url = (base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).rstrip("/")

    numeric_summary = {
        str(key): float(value)
        for key, value in summary.to_dict().items()
        if value is not None
    }

    system_prompt = (
        "You are a quantitative finance assistant. "
        "Summarize Monte Carlo simulation output in plain English for a technical user. "
        "Do not give personalized financial advice. "
        "Focus on risk, uncertainty, and what the metrics imply."
    )

    user_prompt = {
        "ticker": ticker,
        "simulation_model": simulation_model,
        "horizon_days": days,
        "scenarios": scenarios,
        "metrics": numeric_summary,
        "requested_output": (
            "Return Markdown with 4-8 short bullets. "
            "Include expected return, probability above/below current, "
            "and downside risk (VaR/CVaR). "
            "Call out key assumptions/limitations in one bullet."
        ),
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, indent=2)},
        ],
        "temperature": 0.2,
    }

    url = f"{base_url}/chat/completions"
    response = _post_json(
        url=url,
        payload=payload,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )

    try:
        content = response["choices"][0]["message"]["content"]
    except Exception as exc:
        raise OpenAIRequestError(f"Unexpected OpenAI response format: {response}") from exc

    if not isinstance(content, str) or not content.strip():
        raise OpenAIRequestError("OpenAI returned an empty response.")

    return content.strip()
