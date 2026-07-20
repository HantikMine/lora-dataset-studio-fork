"""Anima (RunPod serverless) variation generator.

Calls the Anima 2B anime text-to-image model via a RunPod serverless endpoint.
"""

from __future__ import annotations
import base64
import logging
import os
import requests

logger = logging.getLogger(__name__)

_ANIMA_ENDPOINT = os.environ.get('ANIMA_ENDPOINT', 'https://api.runpod.ai/v2/kwto1m8tb2mdec/runsync')

def _api_key() -> str:
    return os.environ.get('ANIMA_API_KEY', '')

def generate_variation(ref_bytes: bytes | list[bytes], prompt: str, model: str | None = None,
                       aspect_ratio: str = '1:1') -> bytes | None:
    key = _api_key()
    if not key:
        logger.warning('anima: ANIMA_API_KEY missing')
        return None
    dims = {'1:1': (1024, 1024), '3:4': (896, 1152), '9:16': (768, 1344),
            '4:3': (1152, 896), '16:9': (1344, 768)}
    width, height = dims.get(aspect_ratio, (1024, 1024))
    quality_mode = os.environ.get('ANIMA_QUALITY', '').lower() in ('1', 'true', 'yes')
    if not prompt.strip().lower().startswith('masterpiece'):
        prompt = f'masterpiece, best quality, score_7, {prompt}'
    input_payload = {'prompt': prompt, 'seed': 0, 'width': width, 'height': height, 'batch_size': 1}
    if quality_mode:
        input_payload.update({'lora_strength_1': 0.0, 'lora_strength_2': 0.0, 'steps': 30,
            'cfg': 5.0, 'negative_prompt': 'worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration'})
    else:
        input_payload.update({'lora_strength_1': 0.9, 'lora_strength_2': 0.0, 'steps': 12,
            'cfg': 1.0, 'negative_prompt': ''})
    input_payload.update({'sampler_name': 'er_sde', 'scheduler': 'simple', 'denoise': 1.0})
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'}
    try:
        r = requests.post(_ANIMA_ENDPOINT, headers=headers, json={'input': input_payload}, timeout=(10, 180))
    except requests.RequestException as e:
        logger.warning(f'anima: request error: {e}')
        return None
    if r.status_code != 200:
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
    if isinstance(output, dict):
        images = output.get('images') or []
        if images and isinstance(images[0], str):
            try:
                return base64.b64decode(images[0])
            except Exception:
                return None
    if isinstance(output, str):
        try:
            return base64.b64decode(output)
        except Exception:
            return None
    return None
