"""Anima VLM character descriptor — extracts structured Danbooru tags + natural
language description from a reference photo via OpenRouter (Gemma 31B, Friendli).

Used by the Anima engine to inject character identity into every shot prompt:
  1. reference photo → Gemma VLM → {tags, description}
  2. {tags, description} + shot prompt → RunPod endpoint

The VLM output is cached per dataset (FaceDataset.anima_character_desc) so the
expensive vision call happens once, not per shot.
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
_MODEL = 'google/gemma-4-31b-it'

# ── VLM Prompt ─────────────────────────────────────────────────────────────
# The model must output ONLY valid JSON with two fields:
#   tags:       space-separated Danbooru tags (quality + appearance)
#   description: natural-language paragraph describing the character
_VLM_PROMPT = """Analyze this image of a character. Output ONLY valid JSON — no markdown, no code fences, no prefacing text.

Return a JSON object with exactly these two fields:
{
  "tags": "space-separated Danbooru tags describing the character's permanent appearance traits — hair color, hair style, eye color, skin tone, face shape, body type, distinguishing marks. Include quality tags: masterpiece, best quality. Include safety tag: safe. Include the subject tag: 1girl. Use ONLY tags, no English prose.",
  "description": "one smooth English paragraph describing the character's appearance in natural language — hair, eyes, face, body, clothing visible in the photo. Keep it factual and SFW."
}

Remember: output ONLY the JSON object, nothing else."""


def describe_character(image_path: str) -> dict | None:
    """Send reference photo to Gemma VLM, return {'tags': ..., 'description': ...}.

    Returns None when OpenRouter is unreachable, the key is missing, or the
    response can't be parsed.
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

    # Detect MIME type from extension
    ext = os.path.splitext(image_path)[1].lower()
    mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
            'webp': 'image/webp'}.get(ext.lstrip('.'), 'image/webp')

    headers = {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
    }

    payload = {
        'model': _MODEL,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{img_b64}'}},
                {'type': 'text', 'text': _VLM_PROMPT},
            ],
        }],
        'provider': {
            'order': ['Friendli'],
            'allow_fallbacks': False,
        },
        'response_format': {'type': 'json_object'},
        'max_tokens': 800,
    }

    # Three attempts with backoff — OpenRouter can 503 transiently
    for attempt in range(3):
        try:
            r = requests.post(
                _OPENROUTER_URL, headers=headers, json=payload,
                timeout=(10, 60),
            )
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
        return None  # all attempts failed

    # Parse
    try:
        body = r.json()
        content = body['choices'][0]['message']['content']
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning(f'anima_vision: bad response structure: {exc}')
        return None

    # Strip markdown fences that OpenRouter sometimes injects
    content = content.strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[-1]
        if content.endswith('```'):
            content = content[:-3].strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Sometimes the model wraps JSON in quotes — try stripping outer quotes
        if content.startswith('"') and content.endswith('"'):
            try:
                result = json.loads(json.loads(content))
            except (json.JSONDecodeError, TypeError):
                logger.warning(f'anima_vision: unparseable JSON: {content[:200]}')
                return None
        else:
            logger.warning(f'anima_vision: unparseable JSON: {content[:200]}')
            return None

    tags = (result.get('tags') or '').strip()
    description = (result.get('description') or '').strip()

    if not tags and not description:
        logger.warning('anima_vision: empty response from VLM')
        return None

    return {'tags': tags, 'description': description}
