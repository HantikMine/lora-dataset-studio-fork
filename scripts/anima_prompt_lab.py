"""Anima Prompt Lab — iterate a VLM prompt until generated characters match the
reference photo almost perfectly (appearance only, not clothing).

Pipeline per iteration:
  1. Gemini profiles the reference photo → structured character JSON
  2. Build an Anima prompt from that JSON
  3. Generate one image via RunPod
  4. Gemini compares the reference vs the generated image (TWO images in one
     call) → per-trait similarity scores + concrete fix suggestions
  5. Human/agent reads suggestions, tweaks the prompt builder, re-runs

Usage:
  python scripts/anima_prompt_lab.py <reference_image> [--shots N] [--out DIR]
"""

from __future__ import annotations
import argparse
import base64
import json
import os
import sys
import time

import requests

# ── Config ─────────────────────────────────────────────────────────────────
OR_URL = 'https://openrouter.ai/api/v1/chat/completions'
MODEL = 'google/gemini-3.1-pro-preview'
ANIMA_ENDPOINT = os.environ.get(
    'ANIMA_ENDPOINT', 'https://api.runpod.ai/v2/kwto1m8tb2mdec/runsync')
OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY', '')
ANIMA_KEY = os.environ.get('ANIMA_API_KEY', '')

# Load keys from repo .env if not in environment
def _load_env():
    global OPENROUTER_KEY, ANIMA_KEY
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '.env'),
        os.path.join(os.path.dirname(__file__), '..', '..', '.env'),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('OPENROUTER_API_KEY='):
                        OPENROUTER_KEY = OPENROUTER_KEY or line.split('=', 1)[1].strip()
                    elif line.startswith('ANIMA_API_KEY='):
                        ANIMA_KEY = ANIMA_KEY or line.split('=', 1)[1].strip()
            break

_load_env()

# ── Step 1: Gemini profiles the reference ──────────────────────────────────
# KEY DESIGN: appearance traits only. The profile is the ONLY input the prompt
# builder gets — so every trait listed here ends up in the generation prompt.
PROFILE_PROMPT = """You are a professional character model-sheet analyst. Examine this reference photo of a character.

Your job: produce a PERFECT appearance profile that another AI will turn into an image-generation prompt. Every visible physical trait must be captured EXACTLY — the generated character must look like the same person.

Output ONLY a JSON object, no markdown:

{
  "subject": "exact gender tag + age tag: 1girl/1boy + teen/adult/mature",
  "head": "Danbooru tags for HEAD AND FACE ONLY — be exhaustive and precise: hair color (exact shade, e.g. 'deep indigo' not just 'blue'), hair length, hair texture (straight/wavy/curly), hairstyle (bun/ponytail/loose/braids), bangs (blunt/swept/none), eye color (exact shade + saturation), eye shape (almond/round/slanted), iris detail, eyebrows (thickness/shape), face shape, nose (small/pointed/button), lips (thin/full), skin tone (fair/light/medium/tan/dark + undertone), makeup if visible",
  "upper": "Danbooru tags for UPPER BODY: neck (slender/average), shoulder width, bust size (small/medium/large), arm build, torso build, collarbone visibility",
  "lower": "Danbooru tags for LOWER BODY: hip width, leg build, thigh size, waist, waist-to-hip ratio",
  "body_global": "Danbooru tags for GLOBAL traits: overall body type (slender/curvy/athletic/petite/average), height impression, skin tone if not in head, figure impression (hourglass/pear/rectangle)",
  "absent_features": "Danbooru tags of traits that are OBVIOUSLY absent and must NOT appear: e.g. 'no glasses, no piercings, no tattoos, no beauty marks, no heavy makeup'. Only list what is clearly absent on the face/body.",
  "description": "ONE precise natural-language sentence capturing ONLY the appearance: age range, hair (exact color+style), eyes (exact color+shape), skin, body type. NO clothing mention at all."
}

CRITICAL RULES:
- ONLY what is VISIBLE in the photo. Never invent traits you cannot see
- COLORS: be precise about shade and saturation — 'deep indigo' beats 'blue', 'dusty rose' beats 'pink'. Exact color is the #1 recognition factor
- Tag order: color before shape/size (e.g. 'hazel eyes, wide-set eyes' not the reverse)
- Use Danbooru style: lowercase, commas
- The description sentence must NOT mention clothing, accessories, pose, or background
- Output ONLY the JSON"""


# ── Step 2: Build Anima prompt ─────────────────────────────────────────────
def build_prompt(profile: dict, shot: str, weight: float | None = None,
                 weight_head: float | None = None) -> str:
    """Assemble the Anima prompt from the profile + shot description.
    Shot framing selects which body regions are included.
    ``weight`` wraps EVERY identity tag in (tag:weight); ``weight_head`` wraps
    only the HEAD tags (eyes/face — the common weak spot). weight_head wins.
    """
    sp = (shot or '').strip()
    sl = sp.lower()
    if any(w in sl for w in ('full body', 'standing', 'feet', 'back view')):
        regions = ('subject', 'head', 'upper', 'lower', 'body_global')
    elif any(w in sl for w in ('bust', 'upper body', 'waist')):
        regions = ('subject', 'head', 'upper', 'body_global')
    else:
        regions = ('subject', 'head', 'body_global')

    def _apply_weight(tag_str: str, region: str) -> str:
        w = weight_head if (weight_head and region == 'head') else weight
        if not (w and w != 1.0):
            return tag_str
        # Wrap EACH tag in its own (tag:weight) pair, e.g. (blue eyes:1.2), (narrow eyes:1.2)
        tags = [t.strip() for t in tag_str.split(',') if t.strip()]
        return ', '.join(f'({t}:{w})' for t in tags)

    identity = ', '.join(
        _apply_weight((profile.get(r) or '').strip(), r)
        for r in regions if (profile.get(r) or '').strip())
    desc = (profile.get('description') or '').strip()
    absent = (profile.get('absent_features') or '').strip()

    # Guards: the artist tag anchors the art style; absent blocks feature leakage
    guards = ['@akipeko']
    if absent:
        guards.append(absent)

    parts = [g for g in guards if g]
    if identity:
        parts.append(identity)
    parts.append(sp)
    prompt = ', '.join(parts)
    if desc:
        prompt = f'{prompt}. {desc}'
    return prompt


# ── Step 3: Generate via RunPod ────────────────────────────────────────────
def generate(prompt: str, out_path: str, quality: bool = False) -> bool:
    if quality:
        payload = {
            'input': {
                'prompt': prompt,
                'seed': 0,
                'width': 1024, 'height': 1024, 'batch_size': 1,
                'lora_strength_1': 0.0, 'lora_strength_2': 0.0,
                'steps': 30, 'cfg': 5.0,
                'negative_prompt': ('worst quality, low quality, score_1, score_2, '
                                    'score_3, artist name, blurry, jpeg artifacts'),
                'sampler_name': 'er_sde', 'scheduler': 'simple', 'denoise': 1.0,
            }
        }
    else:
        payload = {
            'input': {
                'prompt': prompt,
                'seed': 0,
                'width': 1024, 'height': 1024, 'batch_size': 1,
                'lora_strength_1': 0.9, 'lora_strength_2': 0.0,
                'steps': 12, 'cfg': 1.0,
                'negative_prompt': '',
                'sampler_name': 'er_sde', 'scheduler': 'simple', 'denoise': 1.0,
            }
        }
    headers = {'Content-Type': 'application/json',
               'Authorization': f'Bearer {ANIMA_KEY}'}
    try:
        r = requests.post(ANIMA_ENDPOINT, headers=headers, json=payload, timeout=(10, 240))
    except requests.RequestException as e:
        print(f'  [!] RunPod request failed: {e}')
        return False
    if r.status_code != 200:
        print(f'  [!] RunPod HTTP {r.status_code}: {r.text[:200]}')
        return False
    body = r.json()
    output = body.get('output')
    if isinstance(output, list) and output:
        b64 = output[0].get('image_url', '') if isinstance(output[0], dict) else ''
        if b64:
            with open(out_path, 'wb') as f:
                f.write(base64.b64decode(b64))
            return True
    print(f'  [!] RunPod unexpected output: {str(output)[:200]}')
    return False


# ── Step 4: Gemini compares (TWO images in one call) ───────────────────────
COMPARE_PROMPT = """You are an expert at judging whether two images show the SAME character.

Compare the REFERENCE photo with the GENERATED photo. The character's APPAREL may differ — IGNORE clothing entirely. Judge only PERMANENT appearance: face shape, eyes (color+shape), hair (color+style), skin tone, body type, proportions, age impression.

IMPORTANT: score ONLY what is actually VISIBLE in BOTH images. If a body part is not visible in the generated image (e.g. a close-up face shot), score that category null (omit it from the scores object) — do NOT give it 0. 0 means "visible but completely different".

Output ONLY JSON:
{
  "overall_similarity": 0-10,
  "scores": {
    "face": 0-10,
    "eyes": 0-10,
    "hair": 0-10,
    "skin": 0-10,
    "body": 0-10,
    "age": 0-10
  },
  "matched_traits": ["trait that matches well, e.g. 'hair color'", "..."],
  "mismatched_traits": ["trait that differs, e.g. 'eye shape'", "..."],
  "fix_suggestions": "ONE concrete instruction for fixing the biggest mismatch, e.g. 'the generated eyes are round, reference has narrow eyes — emphasize narrow eye shape and exact color in the prompt'"
}

Scoring guide: 9-10 indistinguishable, 7-8 minor difference, 5-6 noticeable but same person, 3-4 different person, 0-2 completely different."""


def compare(reference_b64: str, generated_path: str) -> dict | None:
    with open(generated_path, 'rb') as f:
        gen_b64 = base64.b64encode(f.read()).decode('ascii')
    payload = {
        'model': MODEL,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/webp;base64,{reference_b64}'}},
                {'type': 'image_url', 'image_url': {'url': f'data:image/webp;base64,{gen_b64}'}},
                {'type': 'text', 'text': COMPARE_PROMPT},
            ],
        }],
        'response_format': {'type': 'json_object'},
        'max_tokens': 800,
    }
    headers = {'Authorization': f'Bearer {OPENROUTER_KEY}',
               'Content-Type': 'application/json'}
    try:
        r = requests.post(OR_URL, headers=headers, json=payload, timeout=(10, 90))
    except requests.RequestException as e:
        print(f'  [!] Compare request failed: {e}')
        return None
    if r.status_code != 200:
        print(f'  [!] Compare HTTP {r.status_code}: {r.text[:200]}')
        return None
    try:
        content = r.json()['choices'][0]['message']['content'].strip()
        if content.startswith('```'):
            content = content.split('\n', 1)[-1]
            if content.endswith('```'):
                content = content[:-3].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Truncated JSON (max_tokens cut): salvage by trimming to the last
            # complete closing brace, then retry parse.
            last = content.rfind('}')
            if last > 0:
                try:
                    return json.loads(content[:last + 1])
                except json.JSONDecodeError:
                    pass
            raise
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        print(f'  [!] Compare parse failed: {e}')
        return None


# ── Step 1 runner ──────────────────────────────────────────────────────────
def profile_reference(reference_b64: str) -> dict | None:
    payload = {
        'model': MODEL,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/webp;base64,{reference_b64}'}},
                {'type': 'text', 'text': PROFILE_PROMPT},
            ],
        }],
        'response_format': {'type': 'json_object'},
        'max_tokens': 1200,
    }
    headers = {'Authorization': f'Bearer {OPENROUTER_KEY}',
               'Content-Type': 'application/json'}
    try:
        r = requests.post(OR_URL, headers=headers, json=payload, timeout=(10, 90))
    except requests.RequestException as e:
        print(f'  [!] Profile request failed: {e}')
        return None
    if r.status_code != 200:
        print(f'  [!] Profile HTTP {r.status_code}: {r.text[:300]}')
        return None
    try:
        content = r.json()['choices'][0]['message']['content'].strip()
        if content.startswith('```'):
            content = content.split('\n', 1)[-1]
            if content.endswith('```'):
                content = content[:-3].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            last = content.rfind('}')
            if last > 0:
                try:
                    return json.loads(content[:last + 1])
                except json.JSONDecodeError:
                    pass
            raise
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        print(f'  [!] Profile parse failed: {e}')
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reference', help='path to the reference photo')
    ap.add_argument('--shots', type=int, default=1, help='number of test shots to generate')
    ap.add_argument('--out', default='anima_prompt_lab_out', help='output directory')
    ap.add_argument('--weight', type=float, default=None,
                    help='wrap ALL identity tags in (tag:weight), e.g. 1.2')
    ap.add_argument('--weight-head', type=float, default=None,
                    help='wrap only HEAD tags (eyes/face) in (tag:weight) — overrides --weight for head')
    ap.add_argument('--quality', action='store_true',
                    help='quality mode: 30 steps, CFG 5 (instead of turbo 12/1)')
    args = ap.parse_args()

    if not OPENROUTER_KEY or not ANIMA_KEY:
        print('❌ Need OPENROUTER_API_KEY and ANIMA_API_KEY (env or .env)')
        sys.exit(1)

    ref_path = os.path.abspath(args.reference)
    if not os.path.isfile(ref_path):
        print(f'❌ Reference not found: {ref_path}')
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    with open(ref_path, 'rb') as f:
        ref_b64 = base64.b64encode(f.read()).decode('ascii')

    print('🔍 STEP 1: Gemini profiles the reference...')
    profile = profile_reference(ref_b64)
    if not profile:
        sys.exit(1)
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    print()

    # Test shots — appearance-focusing descriptions, NO clothing specifics
    test_shots = [
        'close-up portrait, front view, neutral expression',
        'upper body portrait, front view, neutral expression',
        'full body shot, standing, front view, neutral expression',
    ]

    for i in range(min(args.shots, len(test_shots))):
        shot = test_shots[i]
        print(f'🎬 SHOT {i + 1}: {shot}')
        prompt = build_prompt(profile, shot, weight=args.weight, weight_head=args.weight_head)
        print(f'   prompt ({len(prompt)} chars): {prompt[:200]}...')
        print()

        print('   ⚙️ STEP 3: Generating via RunPod...')
        gen_path = os.path.join(args.out, f'gen_{i + 1}.webp')
        if not generate(prompt, gen_path, quality=args.quality):
            continue

        print('   🧪 STEP 4: Gemini compares reference vs generated...')
        verdict = compare(ref_b64, gen_path)
        if verdict:
            if isinstance(verdict, list):
                verdict = verdict[0] if verdict else None
        if verdict:
            s = verdict.get('scores', {}) or {}
            print(f'   overall: {verdict.get("overall_similarity")}/10')
            for k, v in s.items():
                if v is not None:
                    print(f'     {k:6s}: {v}/10')
            if verdict.get('mismatched_traits'):
                print(f'   ❌ mismatches: {", ".join(verdict["mismatched_traits"])}')
            if verdict.get('fix_suggestions'):
                print(f'   🔧 fix: {verdict["fix_suggestions"]}')
        else:
            print('   ⚠ compare failed')
        print()


if __name__ == '__main__':
    main()
