NOISE_CLASSES = [
    "siren",
    "dog",
    "rain",
    "wind",
    "engine",
    "keyboard",
    "footsteps",
    "door",
    "baby",
    "clock"
]

def prompt_to_token(prompt):
    prompt = prompt.lower()
    for i, cls in enumerate(NOISE_CLASSES):
        if cls in prompt:
            return i
    return 0  # default
