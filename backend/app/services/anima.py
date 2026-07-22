"""Anima (RunPod serverless) variation generator.

Calls the Anima 2B anime text-to-image model via a RunPod serverless endpoint.
The reference photo is NOT sent to RunPod — instead, a VLM (OpenRouter Gemma 31B)
describes the character, and that description is injected into every shot prompt
following the anima-prompt skill conventions (Danbooru tags + natural language).
"""

from __future__ import annotations
import base64
import logging
import os
import requests

logger = logging.getLogger(__name__)

_ANIMA_ENDPOINT = os.environ.get(
    'ANIMA_ENDPOINT', 'https://api.runpod.ai/v2/kwto1m8tb2mdec/runsync')


def _api_key() -> str:
    return os.environ.get('ANIMA_API_KEY', '')


def build_anima_prompt(shot_prompt: str, character_desc: dict | None = None) -> str:
    """Build an Anima prompt: identity tags (filtered by framing) + shot description.

    VLM fields: subject, head, upper, lower, body_global, description.
    Only the fields relevant to the shot's framing are injected.
    No negatives, no weight parentheses.
    """
    sp = (shot_prompt or '').strip()
    sp_lower = sp.lower()

    # Detect framing
    if any(w in sp_lower for w in ('full body', 'standing', 'full-body', 'feet', 'kneeling', 'sitting')):
        regions = ('subject', 'head', 'upper', 'lower', 'body_global')
    elif any(w in sp_lower for w in ('bust', 'upper body', 'upper-body', 'waist')):
        regions = ('subject', 'head', 'upper', 'body_global')
    else:
        regions = ('subject', 'head', 'body_global')

    # Expression tags to strip (they change per shot, not permanent identity)
    _expr_strip = {'smiling', 'frown', 'grin', 'smirk', 'laughing', 'crying', 'angry', 'sad',
                   'surprised', 'blush', 'blushing', 'closed_mouth', 'open_mouth', 'parted_lips',
                   'teeth', 'tongue', 'tongue_out', 'looking_at_viewer', 'looking_away'}

    # Build identity from selected regions, strip expression tags
    identity = []
    if character_desc:
        for region in regions:
            v = (character_desc.get(region) or '').strip()
            if v:
                filtered = [t.strip() for t in v.split(',')
                           if t.strip() and t.strip().lower() not in _expr_strip]
                if filtered:
                    identity.append(', '.join(filtered))

    identity_str = ', '.join(identity)

    # Build: identity, shot_description. No negatives.
    parts = [p for p in [identity_str, sp] if p]
    prompt = ', '.join(parts)
    prompt = prompt.replace('  ', ' ').strip()

    # Append natural-language description at end
    if character_desc:
        desc = (character_desc.get('description') or '').strip()
        if desc:
            prompt = f'{prompt}. {desc}'

    return prompt


def generate_variation(
    ref_bytes: bytes | list[bytes],
    prompt: str,
    model: str | None = None,
    aspect_ratio: str = '1:1',
    character_desc: dict | None = None,
) -> bytes | None:
    """Reference photo(s) + Danbooru prompt → generated image bytes, or None.

    ``ref_bytes`` is accepted for API contract compatibility but Anima is
    text-to-image — the reference is only used indirectly via VLM description.

    ``character_desc`` is a cached VLM result: {'tags': '...', 'description': '...'}.
    When provided, it enriches every shot with the character's identity.

    Turbo mode (default): 12 steps, CFG 1. Quality mode (ANIMA_QUALITY=true): 30 steps, CFG 5.
    """
    key = _api_key()
    if not key:
        print('[anima] ANIMA_API_KEY missing', flush=True)
        logger.warning('anima: ANIMA_API_KEY missing')
        return None

    dims = {
        '1:1': (1024, 1024), '3:4': (896, 1152), '9:16': (768, 1344),
        '4:3': (1152, 896), '16:9': (1344, 768),
    }
    width, height = dims.get(aspect_ratio, (1024, 1024))

    quality_mode = os.environ.get('ANIMA_QUALITY', '').lower() in ('1', 'true', 'yes')

    full_prompt = build_anima_prompt(prompt, character_desc)

    input_payload = {
        'prompt': full_prompt,
        'seed': 0,
        'width': width,
        'height': height,
        'batch_size': 1,
    }

    if quality_mode:
        input_payload.update({
            'lora_strength_1': 0.0, 'lora_strength_2': 0.0,
            'steps': 30, 'cfg': 5.0,
            'negative_prompt': (
                'worst quality, low quality, score_1, score_2, score_3, '
                'artist name, blurry, jpeg artifacts, chromatic aberration'
            ),
        })
    else:
        input_payload.update({
            'lora_strength_1': 0.9, 'lora_strength_2': 0.0,
            'steps': 12, 'cfg': 1.0, 'negative_prompt': '',
        })

    input_payload.update({
        'sampler_name': 'er_sde', 'scheduler': 'simple', 'denoise': 1.0,
    })

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key}',
    }

    try:
        # Structured log: show what comes from VLM vs from the shot catalog
        if character_desc:
            desc_text = (character_desc.get('description') or '').strip()
            print(f'[anima] VLM identity:  {build_anima_prompt("", character_desc)}')
            if desc_text:
                print(f'[anima] VLM desc:      {desc_text}')
            print(f'[anima] SHOT catalog:   {prompt}')
            print(f'[anima] COMBINED ({len(full_prompt)} chars):', flush=True)
        else:
            print(f'[anima] prompt ({len(full_prompt)} chars):', flush=True)
        print(full_prompt)
        r = requests.post(
            _ANIMA_ENDPOINT, headers=headers, json={'input': input_payload},
            timeout=(10, 180),
        )
    except requests.RequestException as e:
        logger.warning(f'anima: request error: {e}')
        return None

    if r.status_code != 200:
        print(f'[anima] HTTP {r.status_code}: {r.text[:200]}', flush=True)
        logger.warning(f'anima: HTTP {r.status_code}: {r.text[:300]}')
        return None

    try:
        body = r.json()
    except ValueError:
        logger.warning('anima: invalid JSON response')
        return None

    output = body.get('output')
    if not output:
        logger.warning('anima: no output in response')
        return None

    # RunPod returns output as a LIST of dicts: [{'image_url': '<base64>', 'seed': ...}]
    if isinstance(output, list) and len(output) > 0:
        item = output[0]
        if isinstance(item, dict):
            b64 = (item.get('image_url') or '').strip()
            if b64:
                try:
                    return base64.b64decode(b64)
                except Exception as exc:
                    logger.warning(f'anima: failed to decode base64 ({len(b64)} chars, starts with {b64[:20]}...): {exc}')
                    return None
        return None

    # Fallback: output is a dict with 'images' key
    if isinstance(output, dict):
        images = output.get('images') or []
        if images and isinstance(images[0], str):
            try:
                return base64.b64decode(images[0])
            except Exception:
                return None

    # Fallback: output is itself a base64 string
    if isinstance(output, str):
        try:
            return base64.b64decode(output)
        except Exception:
            return None

    logger.warning(f'anima: unexpected output type: {type(output).__name__}')
    return None
