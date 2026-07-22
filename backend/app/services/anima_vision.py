"""Anima VLM character descriptor — extracts structured Danbooru tags + natural
language description from a reference photo via OpenRouter (Gemma 31B, Friendli).

Used by the Anima engine to inject character identity into every shot prompt:
  1. reference photo → Gemma VLM → {subject, head, upper, lower, body_global, description}
  2. identity tags + description + shot prompt → RunPod endpoint

The VLM output is cached per dataset (FaceDataset) so the expensive vision call
happens once per batch, not per shot.
"""

from __future__ import annotations
import base64
import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

_OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
_MODEL = 'google/gemini-3.1-pro-preview'

# ── VLM Prompt ─────────────────────────────────────────────────────────────
_VLM_PROMPT = """You are an anime character profiling expert. Analyze this photo and output ONLY a JSON object. No markdown, no explanations.

A downstream system injects your output into image prompts. Split traits into these five fields:

{
  "subject": "comma-separated: 1girl (or 1boy), plus age-range tag like 'teen' or 'adult' or 'mature'",
  "head": "comma-separated Danbooru tags for HEAD AND FACE ONLY: hair color, hair length, hair texture, hair style, bangs, eye color, eye shape, skin tone, face shape, nose, lips, eyebrows",
  "upper": "comma-separated Danbooru tags for UPPER BODY: neck, shoulders, bust/chest size, arm build, torso build — neck to waist",
  "lower": "comma-separated Danbooru tags for LOWER BODY: hip width, leg build, thigh build, waist-to-hip ratio — waist down",
  "body_global": "comma-separated Danbooru tags for GLOBAL body traits: body type, height, skin tone if not in head",
  "description": "ONE natural English sentence describing this character. Example: A young woman with long blue hair, blue eyes, pale skin."
}

CRITICAL RULES:
- Each field is independent — a "bust" shot uses subject+head+upper; a "full body" shot uses ALL
- Describe ONLY what you can SEE in the photo. Do NOT invent or guess unknown traits
- Use ONLY Danbooru tags (lowercase, underscores) for tag fields
- NEVER include clothing, accessories, hats, glasses, jewelry — those change per shot
- NEVER include expression tags (smiling/frown/etc) — expression changes per shot
- Output ONLY the JSON object, nothing else"""


def describe_character(image_path: str) -> dict | None:
    """Send reference photo to Gemma VLM, return categorized character traits.

    Returns dict with keys: subject, head, upper, lower, body_global, description
    Returns None when OpenRouter is unreachable, key is missing, or parsing fails.
    """
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not key:
        logger.warning('anima_vision: OPENROUTER_API_KEY missing')
        return None

    # Read + base64-encode the image
    try:
        with open(image_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('ascii')
    except (OSError, ValueError) as exc:
        logger.warning(f'anima_vision: cannot read {image_path}: {exc}')
        return None

    ext = os.path.splitext(image_path)[1].lower()
    mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
            'webp': 'image/webp'}.get(ext.lstrip('.'), 'image/webp')

    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

    payload = {
        'model': _MODEL,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{img_b64}'}},
                {'type': 'text', 'text': _VLM_PROMPT},
            ],
        }],
        'response_format': {'type': 'json_object'},
        'max_tokens': 800,
    }

    for attempt in range(3):
        try:
            r = requests.post(_OPENROUTER_URL, headers=headers, json=payload,
                            timeout=(10, 60))
        except requests.RequestException as exc:
            logger.warning(f'anima_vision: request error (attempt {attempt+1}): {exc}')
            time.sleep(2 ** attempt)
            continue

        if r.status_code == 200:
            break
        if r.status_code == 429:
            time.sleep(5)
            continue
        logger.warning(f'anima_vision: HTTP {r.status_code}: {r.text[:300]}')
        if attempt < 2:
            time.sleep(2 ** attempt)
    else:
        return None

    try:
        body = r.json()
        content = body['choices'][0]['message']['content']
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning(f'anima_vision: bad response structure: {exc}')
        return None

    content = content.strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[-1]
        if content.endswith('```'):
            content = content[:-3].strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        if content.startswith('"') and content.endswith('"'):
            try:
                result = json.loads(json.loads(content))
            except (json.JSONDecodeError, TypeError):
                logger.warning(f'anima_vision: unparseable JSON: {content[:200]}')
                return None
        else:
            logger.warning(f'anima_vision: unparseable JSON: {content[:200]}')
            return None

    subject = (result.get('subject') or '').strip()
    head = (result.get('head') or '').strip()
    upper = (result.get('upper') or '').strip()
    lower = (result.get('lower') or '').strip()
    body_global = (result.get('body_global') or '').strip()
    description = (result.get('description') or '').strip()

    if not subject and not head:
        logger.warning('anima_vision: empty response from VLM')
        return None

    return {
        'subject': subject, 'head': head, 'upper': upper,
        'lower': lower, 'body_global': body_global, 'description': description,
    }
