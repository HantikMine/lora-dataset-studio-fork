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
#   tags:       comma-separated Danbooru tags — PERMANENT traits only (hair, eyes, face, body, skin)
#   clothing:   tags describing clothes seen in THIS photo (NOT permanent)
#   description: one-sentence natural-language description
_VLM_PROMPT = """You are a character analysis expert for anime image generation. Examine this photo in extreme detail. Output ONLY valid JSON — no markdown, no explanations.

Describe EVERY visible permanent trait of the person. Be thorough and specific. Return exactly:

{
  "tags": "comma-separated Danbooru tags listing ALL visible permanent identity traits. Start with: masterpiece, best quality, safe, 1girl. Then include EVERY visible trait: hair color, hair length, hair style (straight/wavy/curly/ponytail/bun/braided/etc), bangs style, eye color, eye shape, skin tone, face shape, nose, lips, eyebrows, body type/build, height appearance, any visible marks (freckles/moles/scars). Be exhaustive — every visible detail becomes a tag.",
  "clothing": "comma-separated tags for the OUTFIT visible in this photo — top, bottom, shoes, accessories, jewelry. These will be REPLACED per shot so describe them accurately but they are NOT permanent.",
  "description": "ONE paragraph of 2-3 sentences describing this specific character in natural English. Cover: approximate age appearance, face shape, eye color and shape, hair (color, length, style, texture), skin tone, body build, and any distinctive features that make this person recognizable. Be vivid and precise."
}

CRITICAL RULES:
- Tags and clothing use ONLY comma-separated Danbooru style (e.g. 'long hair, black hair, straight hair, blunt bangs')
- Description uses natural English sentences
- Tags describe PERMANENT traits that stay the same in any outfit or scene
- Clothing describes what is visible NOW — it will change per shot
- Be MAXIMALLY detailed — rather 30 accurate tags than 10 vague ones
- Output ONLY the JSON object, nothing else"""


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
    clothing = (result.get('clothing') or '').strip()

    if not tags and not description:
        logger.warning('anima_vision: empty response from VLM')
        return None

    return {'tags': tags, 'description': description, 'clothing': clothing}
