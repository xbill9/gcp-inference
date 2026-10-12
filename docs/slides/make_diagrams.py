"""Architecture diagrams for the deck's section slides, drawn in code so every label is exact.

    python3 docs/slides/make_diagrams.py     writes docs/slides/arch-*.png; the deck uses copies named with their hash

Same layout as ~/aws-inference-strands/docs/slides/make_diagrams.py, in Google's colours. Labels come
from backends.py and stage/up.sh.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
INK, BLUE, GREEN, YELLOW, RED, BG = "#202124", "#1A73E8", "#188038", "#F9AB00", "#D93025", "#EEEEEE"
FILL = {"laptop": "#FFFFFF", "gcp": "#E8F0FE", "path": "#E6F4EA", "chip": "#174EA6"}
W, H = 32, 8.8  # data units; a 16 x 4.4 in figure at 200 dpi -> 3200 x 880 px
FIG, DPI = (16, 4.4), 200
plt.rcParams["font.family"] = ["Roboto", "DejaVu Sans"]


def box(ax, x, y, w, h, title, sub, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.35",
                                fc=FILL[kind], ec=BLUE if kind != "path" else GREEN, lw=3))
    color = "white" if kind == "chip" else INK
    ax.text(x + w / 2, y + h - 0.95, title, ha="center", va="center", fontsize=21, weight="bold", color=color)
    ax.text(x + w / 2, y + h - 1.55, sub, ha="center", va="top", fontsize=16, color=color, linespacing=1.25)


def arrow(ax, x0, x1, y, label=""):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=34, lw=3.5, color=YELLOW))
    if label:
        ax.text((x0 + x1) / 2, y + 0.35, label, ha="center", va="bottom", fontsize=15, color=INK)


def row(ax, y, h, items):
    """items: [(title, sub, kind, width, arrow_label_to_next)] laid out left to right."""
    gap = 2.4 if len(items) > 3 else 3.2
    total = sum(i[3] for i in items)
    x = (W - total - gap * (len(items) - 1)) / 2
    for i, (title, sub, kind, w, label) in enumerate(items):
        box(ax, x, y, w, h, title, sub, kind)
        if i < len(items) - 1:
            arrow(ax, x + w + 0.1, x + w + gap - 0.1, y + h / 2, label)
        x += w + gap


def canvas():
    fig = plt.figure(figsize=FIG, dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")
    return fig, ax


def figure(name, rows):
    fig, ax = canvas()
    for y, h, items in rows:
        row(ax, y, h, items)
    path = os.path.join(OUT, f"arch-{name}.png")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(path)


def main():
    # Hosted on Vertex AI: billed per token, nothing to start
    figure("hosted", [
        (4.8, 3.2, [("Strands agent", "GeminiModel", "laptop", 5.6, "Vertex AI API"),
                    ("Gemini on Vertex AI", "global endpoint, per token", "gcp", 7.4, ""),
                    ("Gemini 3.5 Flash-Lite", "Google model", "chip", 5.6, "")]),
        (0.8, 3.2, [("Strands agent", "OpenAIModel", "laptop", 5.6, "OpenAI-compatible"),
                    ("Vertex AI MaaS", "global endpoint, per token", "gcp", 7.4, ""),
                    ("Gemma 4 26B A4B", "open model, serverless", "chip", 5.6, "")]),
    ])
    # Your container, Google runs it: Vertex AI endpoint and Cloud Run
    figure("managed", [
        (4.8, 3.2, [("Strands agent", "OpenAIModel", "laptop", 5.4, ":rawPredict"),
                    ("Vertex AI endpoint", "g2-standard-4, us-central1\nvLLM 0.30 container", "gcp", 7.0, ""),
                    ("NVIDIA L4", "Gemma 4 E2B repack", "chip", 5.4, "")]),
        (0.8, 3.2, [("Strands agent", "OpenAIModel", "laptop", 5.4, "HTTPS + ID token"),
                    ("Cloud Run service", "4 vCPU, 16 GiB, us-east4\nscales to zero", "gcp", 7.0, ""),
                    ("NVIDIA L4", "Gemma 4 E2B repack", "chip", 5.4, "")]),
    ])
    # Your VM or cluster, reached without inbound ports
    figure("gpus", [
        (4.8, 3.2, [("Strands agent", "OpenAIModel", "laptop", 5.4, "localhost"),
                    ("IAP TCP tunnel", "no public port", "path", 5.8, ""),
                    ("Compute Engine", "g2-standard-4\nvLLM 0.30 in Docker", "gcp", 6.4, ""),
                    ("NVIDIA L4", "24 GB\nE2B repack", "chip", 5.4, "")]),
        (0.8, 3.2, [("Strands agent", "OpenAIModel", "laptop", 5.4, "localhost"),
                    ("kubectl port-forward", "no Service exposed", "path", 5.8, ""),
                    ("GKE Autopilot", "a Deployment, one Pod\nnode made on demand", "gcp", 6.4, ""),
                    ("NVIDIA L4", "24 GB\nE2B repack", "chip", 5.4, "")]),
    ])
    # Google silicon
    figure("tpu", [
        (2.7, 3.4, [("Strands agent", "OpenAIModel\ntool calls work", "laptop", 5.4, "localhost"),
                    ("SSH forward", "no public port", "path", 5.8, ""),
                    ("Cloud TPU VM", "vllm/vllm-tpu 0.30\nflex-start request", "gcp", 6.4, ""),
                    ("TPU v6e-1", "one Trillium chip\nGemma 4 E2B, bf16", "chip", 5.4, "")]),
    ])
    # One agent, seven backends, one row each
    fig, ax = canvas()
    box(ax, 0.8, 2.9, 7.0, 3.0, "One Strands agent", "a model object\nper backend", "laptop")
    backends = [("Gemini on Vertex AI", "GeminiModel", "gcp"), ("Gemma 4 MaaS", "OpenAIModel", "gcp"),
                ("Vertex AI endpoint, L4", "OpenAIModel", "gcp"), ("Cloud Run, L4", "OpenAIModel", "gcp"),
                ("GKE Autopilot, L4", "OpenAIModel", "gcp"), ("Compute Engine, L4", "OpenAIModel", "gcp"),
                ("Cloud TPU v6e", "OpenAIModel", "chip")]
    h, step = 1.0, 1.17
    for i, (name, cls, kind) in enumerate(backends):
        y = H - 0.35 - h - i * step
        ax.add_patch(FancyBboxPatch((14.5, y), 16.5, h, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    fc=FILL[kind], ec=BLUE, lw=3))
        color = "white" if kind == "chip" else INK
        ax.text(15.1, y + h / 2, name, ha="left", va="center", fontsize=22, weight="bold", color=color)
        ax.text(30.4, y + h / 2, cls, ha="right", va="center", fontsize=18, color=color)
        ax.add_patch(FancyArrowPatch((7.9, 4.4), (14.4, y + h / 2), arrowstyle="-|>", mutation_scale=26,
                                     lw=3, color=YELLOW))
    path = os.path.join(OUT, "arch-overview.png")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(path)


def ways():
    """One agent, five ways to run a model on Google Cloud, from hosted to your own chip."""
    w, h = 32, 13.2
    fig = plt.figure(figsize=(16, 6.6), dpi=DPI, facecolor="white")  # sits on a white card
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w), ax.set_ylim(0, h), ax.axis("off")
    ax.add_patch(FancyBboxPatch((10, 10.7), 12, 2.1, boxstyle="round,pad=0.02,rounding_size=0.35",
                                fc=FILL["laptop"], ec=BLUE, lw=3))
    ax.text(16, 12.05, "One Strands agent", ha="center", va="center", fontsize=26, weight="bold", color=INK)
    ax.text(16, 11.25, "same question, graded in code", ha="center", va="center", fontsize=18, color=INK)
    cols = [("Hosted API", ["Gemini", "Gemma 4 MaaS"]),
            ("Managed endpoint", ["Vertex AI", "your vLLM, L4"]),
            ("Serverless GPU", ["Cloud Run", "L4, scales to zero"]),
            ("Your GPU", ["Compute Engine", "GKE Autopilot"]),
            ("Google chip", ["Cloud TPU", "v6e, vLLM TPU"])]
    cw, gap, top, bot = 5.6, 0.75, 8.6, 3.4
    for i, (head, lines) in enumerate(cols):
        x = 0.9 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, bot), cw, top - bot, boxstyle="round,pad=0.02,rounding_size=0.35",
                                    fc=FILL["gcp"], ec=BLUE, lw=3))
        ax.add_patch(FancyBboxPatch((x, top - 1.6), cw, 1.6, boxstyle="round,pad=0.02,rounding_size=0.35",
                                    fc=BLUE, ec=BLUE, lw=3))
        ax.text(x + cw / 2, top - 0.8, head, ha="center", va="center", fontsize=20, weight="bold", color="white")
        ax.text(x + cw / 2, top - 2.75, lines[0], ha="center", va="center", fontsize=19, weight="bold", color=INK)
        ax.text(x + cw / 2, top - 4.0, lines[1], ha="center", va="center", fontsize=16, color=INK)
        ax.add_patch(FancyArrowPatch((16, 10.6), (x + cw / 2, top + 0.1), arrowstyle="-|>", mutation_scale=28,
                                     lw=3, color=YELLOW))
    ax.add_patch(FancyArrowPatch((0.9, 2.2), (31.1, 2.2), arrowstyle="-|>", mutation_scale=40, lw=5, color=GREEN))
    ax.text(0.9, 1.0, "nothing to run, pay per token", ha="left", va="center", fontsize=19, color=INK)
    ax.text(31.1, 1.0, "more control, more to run", ha="right", va="center", fontsize=19, color=INK)
    path = os.path.join(OUT, "arch-ways.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
    ways()
