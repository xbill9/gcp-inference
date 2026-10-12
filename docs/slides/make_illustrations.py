"""Deck illustrations, generated with gemini-3.1-flash-lite-image on Vertex AI (the model nb2lite uses).

    python3 docs/slides/make_illustrations.py [name ...]   writes docs/slides/ill-<name>.<hash8>.jpg

Same flat isometric style as the AWS deck's illustrations, in Google's colours. Files are named by
content hash so Slides always fetches a URL it has not cached.
"""

import hashlib
import io
import os
import sys

from google import genai
from google.genai import types

OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = "gemini-3.1-flash-lite-image"
STYLE = ("Flat isometric technical line illustration, clean thin dark outlines, minimal soft shading, "
         "plain white background, generous empty margin. Accent colours: Google blue #1A73E8, green #188038, "
         "yellow #F9AB00; device bodies dark slate #3C4043. No text, no letters, no numbers, no logos.")
PROMPTS = {
    "cover": ("16:9", "Composition fills only the right half of the frame; the left half is empty very light grey "
              "(#EEEEEE) space, and the whole background is that light grey. A central glowing tile linked by thin "
              "blue and green circuit traces to six isometric tiles around it: one with a cloud, a small server, "
              "a GPU card with fans, a square accelerator chip with pins, a shipping container, and three small "
              "stacked boxes forming a cluster."),
    "capacity": ("1:1", "A row of five isometric server racks; four have a small red 'no entry' disc floating above "
                 "them and their slots are empty, the fifth, further away, has one glowing green slot holding a GPU card. "
                 "A small map pin hovers over the fifth rack."),
    "image-pin": ("1:1", "Two shipping containers on an isometric platform: one cracked open with loose broken gears "
                  "falling out, the other sealed with a blue padlock tag and a green check-mark disc above it."),
    "check-numbers": ("1:1", "An isometric clipboard with a bar chart and a magnifying glass inspecting one bar; "
                      "a small calculator beside it with a green check-mark disc."),
    "network": ("1:1", "An isometric server inside a closed blue shield-shaped wall with no doors; a single thin "
                "glowing green tunnel enters underground from a laptop outside the wall."),
    "gemini": ("1:1", "An isometric cloud floating over a platform, with a stream of small glowing tokens flowing from "
               "the cloud into a laptop; a small four-pointed sparkle star above the cloud."),
    "maas": ("1:1", "An isometric vending-machine style kiosk on a platform dispensing glowing tokens into a cup, "
             "with a short queue of small envelopes waiting in front of it."),
    "vertex": ("1:1", "An isometric crane lowering a shipping container into a glass dome that holds a GPU card; "
               "a gear sits on top of the dome."),
    "cloudrun": ("1:1", "An isometric GPU card on a platform with a large power toggle switch beside it showing the "
                 "'off' position on one side and 'on' on the other, and a stopwatch hovering above; a faint ghosted "
                 "outline of the GPU suggests it appears on demand."),
    "gce": ("1:1", "A single isometric server box with its lid lifted, showing one GPU card inside, on a platform."),
    "gke": ("1:1", "Three isometric server boxes connected in a ring by a ship's wheel shape on the platform, the "
            "front box holding a glowing GPU card, the others plain."),
    "tpu-queue": ("1:1", "A square isometric accelerator chip with pins sitting behind a closed gate on a platform, "
                  "with a numbered ticket dispenser and a small hourglass beside the gate; most of the parking bays "
                  "around it are empty and marked with small red discs, one bay is green."),
    "tpu": ("1:1", "A large square isometric accelerator chip with pins on a circuit board, a grid of small cubes on "
            "its surface, glowing blue and green traces running out from it."),
}


def generate(name: str) -> str:
    aspect, prompt = PROMPTS[name]
    client = genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT", "comglitn"),
                          location="global")
    r = client.models.generate_content(
        model=MODEL, contents=f"{prompt} {STYLE}",
        config=types.GenerateContentConfig(response_modalities=["IMAGE"],
                                           image_config=types.ImageConfig(aspect_ratio=aspect)))
    data = next(p.inline_data.data for p in r.candidates[0].content.parts if p.inline_data)
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    path = os.path.join(OUT, f"ill-{name}.{hashlib.sha256(buf.getvalue()).hexdigest()[:8]}.jpg")
    open(path, "wb").write(buf.getvalue())
    return path


if __name__ == "__main__":
    for n in sys.argv[1:] or PROMPTS:
        print(generate(n))
