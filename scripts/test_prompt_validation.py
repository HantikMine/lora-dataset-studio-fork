"""Test script: Gemma describes a character, then validates own prompts against the reference.
Usage: python test_prompt_validation.py <image_path>
"""
import sys, os, json, base64, requests, time

OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY', '')
if not OPENROUTER_KEY:
    # Fallback: try loading from .env
    for env_path in ['.env', '../.env', os.path.join(os.path.dirname(__file__), '.env'), 'C:/Users/vadim/lora-dataset-studio-fork/.env']:
        if os.path.isfile(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith('OPENROUTER_API_KEY='):
                        OPENROUTER_KEY = line.split('=', 1)[1].strip()
                        break
            if OPENROUTER_KEY:
                break

if not OPENROUTER_KEY:
    print('❌ OPENROUTER_API_KEY not set')
    sys.exit(1)

MODEL = 'google/gemma-4-31b-it'
OR_URL = 'https://openrouter.ai/api/v1/chat/completions'

def call_gemma(messages, max_tokens=600, json_mode=False):
    """Single-shot Gemma call via OpenRouter (Friendli only)."""
    payload = {
        'model': MODEL,
        'messages': messages,
        'provider': {'order': ['Friendli'], 'allow_fallbacks': False},
        'max_tokens': max_tokens,
    }
    if json_mode:
        payload['response_format'] = {'type': 'json_object'}
    headers = {'Authorization': f'Bearer {OPENROUTER_KEY}', 'Content-Type': 'application/json'}
    for attempt in range(3):
        try:
            r = requests.post(OR_URL, headers=headers, json=payload, timeout=(10, 90))
        except Exception as e:
            print(f'  [!] Request error (attempt {attempt+1}): {e}')
            time.sleep(3)
            continue
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content']
        print(f'  [!] HTTP {r.status_code}: {r.text[:200]}')
        if r.status_code == 429:
            time.sleep(5)
        else:
            time.sleep(2)
    return None


def encode_image(path):
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}.get(ext, 'image/webp')
    return f'data:{mime};base64,{b64}'


# ── Step 1: Describe character ──────────────────────────────────────────────────
DESCRIBE_PROMPT = """Analyze this photo. Output ONLY JSON:
{
  "subject": "1girl/1boy + age tag",
  "head": "hair, eyes, face, skin traits (comma separated tags)",
  "upper": "neck to waist traits",
  "lower": "waist down traits",
  "body_global": "body type, height",
  "negative": "traits this person does NOT have"
}"""


# ── Test shots to validate ─────────────────────────────────────────────────────
TEST_SHOTS = [
    'close-up portrait, front view, neutral expression',
    'upper body portrait, elegant evening look, dim ambient light',
    'full body shot, standing, front view, casual clothes, street',
    'full body shot, back view, showing hairstyle and silhouette',
    'full body shot, athletic sportswear, gym setting, confident stance',
    'full body shot, one-piece swimsuit, standing at pool edge, daylight',
]


# ── Validation prompt (Step 3) ─────────────────────────────────────────────────
def validate_prompt(img_b64, prompt):
    """Ask Gemma to rate how well a prompt matches the reference photo."""
    vp = (
        "You are evaluating a prompt for an anime image generator. "
        "Compare this GENERATED PROMPT against the REFERENCE PHOTO.\n\n"
        "GENERATED PROMPT: " + prompt + "\n\n"
        "Rate how well this prompt describes what is ACTUALLY visible in the REFERENCE PHOTO. "
        "Score each 1-10. Output ONLY JSON:\n"
        '{"overall": 7, "hair_match": 8, "face_match": 8, "body_match": 7, "style_match": 6, '
        '"worst_mismatch": "str", "best_match": "str", "suggestions": "str"}'
    )


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print('Usage: python test_prompt_validation.py <image_path>')
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f'❌ File not found: {img_path}')
        sys.exit(1)

    img_b64 = encode_image(img_path)
    print(f'📷 Image: {img_path}')
    print()

    # STEP 1: Describe
    print('🔍 STEP 1: VLM describes character...')
    msg = [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': img_b64}},
        {'type': 'text', 'text': DESCRIBE_PROMPT},
    ]}]
    raw = call_gemma(msg, max_tokens=800, json_mode=True)
    if not raw:
        print('❌ VLM failed at step 1')
        sys.exit(1)

    raw = raw.strip().strip('`').strip()
    if raw.startswith('json'):
        raw = raw[4:].strip()
    try:
        desc = json.loads(raw)
    except json.JSONDecodeError:
        print(f'❌ Invalid JSON from VLM: {raw[:200]}')
        sys.exit(1)

    print('   ✅ Description:')
    for k, v in desc.items():
        print(f'      {k}: {str(v)[:100]}')
    print()

    # STEP 2: Build test prompts
    print('📝 STEP 2: Building test prompts...')
    test_prompts = []
    for shot in TEST_SHOTS:
        identity_parts = []
        # Simple framing detection
        sl = shot.lower()
        if any(w in sl for w in ('full body', 'standing', 'feet', 'back view')):
            regions = ('subject', 'head', 'upper', 'lower', 'body_global')
        elif any(w in sl for w in ('bust', 'upper body', 'waist')):
            regions = ('subject', 'head', 'upper', 'body_global')
        else:
            regions = ('subject', 'head', 'body_global')
        for region in regions:
            v = desc.get(region, '').strip()
            if v:
                identity_parts.append(v)
        identity = ', '.join(identity_parts)
        # Negatives
        neg = desc.get('negative', '').strip()
        neg_prefix = ' '.join(f'({t.strip()}:-1)' for t in neg.split(',') if t.strip()) if neg else ''
        # Build
        parts = [p for p in (neg_prefix, identity, shot) if p]
        prompt = ', '.join(parts)
        test_prompts.append((shot, prompt))
        print(f'   [{shot[:50]}...]')
        print(f'   → {prompt[:150]}...')
        print()

    # STEP 3: Validate each prompt against the photo
    print('🧪 STEP 3: Validating prompts against reference...')
    total_score = 0
    for i, (shot, prompt) in enumerate(test_prompts):
        vp = validate_prompt(img_b64, prompt)
        msg = [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': img_b64}},
            {'type': 'text', 'text': vp},
        ]}]
        raw = call_gemma(msg, max_tokens=400, json_mode=True)
        if not raw:
            print(f'   ❌ [{shot[:40]}...] VLM failed')
            continue
        raw = raw.strip().strip('`')
        if raw.startswith('json'):
            raw = raw[4:].strip()
        try:
            score = json.loads(raw)
        except json.JSONDecodeError:
            print(f'   ⚠ [{shot[:40]}...] Bad JSON: {raw[:100]}')
            continue
        ov = score.get('overall', '?')
        total_score += ov if isinstance(ov, (int, float)) else 0
        print(f'   [{i+1}] Overall: {ov}/10  |  Hair: {score.get("hair_match","?")}  Face: {score.get("face_match","?")}  Body: {score.get("body_match","?")}  Style: {score.get("style_match","?")}')
        print(f'       Best:  {score.get("best_match", "?")}')
        print(f'       Worst: {score.get("worst_mismatch", "?")}')
        print()

    avg = total_score / len(test_prompts) if test_prompts else 0
    print(f'📊 AVERAGE SCORE: {avg:.1f}/10')
    if avg >= 7:
        print('   ✅ Good quality — prompts match the reference well')
    elif avg >= 5:
        print('   ⚠ Acceptable — some mismatches, review VLM description')
    else:
        print('   ❌ Poor — VLM description likely inaccurate, check the reference photo quality')


if __name__ == '__main__':
    main()
