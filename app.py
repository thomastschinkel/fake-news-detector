"""Fake News Detector - Hugging Face Space app."""

import os
import sys
import logging
from pathlib import Path

import gradio as gr
import torch

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Environment ──────────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_TIMEOUT", "600")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")

HF_REPO_ID = os.getenv(
    "HF_MODEL_REPO", "ThomasTschinkel/fake-news-detector"
)
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", None)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512

# ── Register custom model classes BEFORE loading ─────────────────────
# This is CRITICAL: AutoModel needs to know about your custom classes.
from model import FakeNewsConfig, FakeNewsDetector  # noqa: E402
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

AutoConfig.register("fake_news_detector", FakeNewsConfig)
AutoModelForSequenceClassification.register(FakeNewsConfig, FakeNewsDetector)

# ── Lazy-loaded singletons ───────────────────────────────────────────
_MODEL = None
_TOKENIZER = None


def load_model_and_tokenizer():
    """Load model and tokenizer once, cache globally."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    log.info("Loading tokenizer from '%s' ...", HF_REPO_ID)
    _TOKENIZER = AutoTokenizer.from_pretrained(
        HF_REPO_ID,
        revision=HF_MODEL_REVISION,
        trust_remote_code=True,
    )
    log.info("✓ Tokenizer loaded")

    log.info("Loading model from '%s' (device=%s) ...", HF_REPO_ID, DEVICE)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO_ID,
        revision=HF_MODEL_REVISION,
        trust_remote_code=True,
    )
    _MODEL.to(DEVICE).eval()
    log.info("✓ Model loaded (%s parameters)",
             f"{sum(p.numel() for p in _MODEL.parameters()):,}")

    return _MODEL, _TOKENIZER


def get_model():
    return load_model_and_tokenizer()[0]


def get_tokenizer():
    return load_model_and_tokenizer()[1]


# ── Helpers ──────────────────────────────────────────────────────────

def _extract_logits(model_output):
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def default_result_html() -> str:
    return (
        "<div style='padding:18px;border-radius:12px;border:1px solid #334155;"
        "background:#0f172a;'>"
        "<div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;'>"
        "Result will appear here</div>"
        "<div style='margin-top:6px;color:#94a3b8;'>"
        "Paste text or upload a file and click <b>Analyze</b>."
        "</div></div>"
    )


def error_html(title: str, detail: str) -> str:
    return (
        "<div style='padding:18px;border-radius:12px;border:1px solid #7f1d1d;"
        "background:#450a0a;'>"
        f"<div style='font-size:1.05rem;font-weight:700;color:#fecaca;'>"
        f"{title}</div>"
        f"<div style='margin-top:6px;color:#fee2e2;'>{detail}</div></div>"
    )


# ── Prediction ───────────────────────────────────────────────────────

def predict_text(text: str):
    cleaned = (text or "").strip()
    if not cleaned:
        return {}, default_result_html(), ""

    tokenizer = get_tokenizer()
    model = get_model()

    enc = tokenizer(
        cleaned,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        raw_output = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
        )
        logits = _extract_logits(raw_output)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()

    label = "FAKE" if pred == 1 else "REAL"
    real_prob = float(probs[0])
    fake_prob = float(probs[1])
    confidence = fake_prob if label == "FAKE" else real_prob

    accent = "#ef4444" if label == "FAKE" else "#22c55e"
    result = (
        "<div style='padding:20px;border-radius:14px;border:1px solid #334155;"
        "background:#0f172a;'>"
        "<div style='font-size:0.95rem;color:#94a3b8;'>Model decision</div>"
        f"<div style='margin-top:4px;font-size:2rem;font-weight:800;"
        f"color:{accent};'>{label}</div>"
        f"<div style='margin-top:8px;font-size:1.2rem;color:#e2e8f0;'>"
        f"Confidence: <b>{confidence:.2%}</b></div></div>"
    )

    label_scores = {"REAL": real_prob, "FAKE": fake_prob}
    return label_scores, result, cleaned[:3000]


# ── File extraction ─────────────────────────────────────────────────

def extract_text_from_file(file_obj) -> str:
    if file_obj is None:
        return ""

    path = Path(file_obj.name)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".csv", ".json", ".log", ".rtf"}:
        for encoding in ("utf-8", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not read the file as text.")

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(
            (page.extract_text() or "") for page in reader.pages
        )

    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(
        "Unsupported format. Allowed: txt, md, csv, json, log, rtf, pdf, docx"
    )


# ── Main handler ─────────────────────────────────────────────────────

def run_prediction(text_input: str, upload):
    try:
        file_text = extract_text_from_file(upload)
    except Exception as exc:
        return {}, error_html("Could not read uploaded file", str(exc)), (text_input or "")

    merged = (text_input or "").strip()
    if file_text.strip():
        merged = f"{merged}\n\n{file_text}".strip()

    try:
        return predict_text(merged)
    except Exception as exc:
        log.exception("Prediction failed")
        return {}, error_html("Prediction error", str(exc)), merged


def get_text_stats(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return "Characters: 0 | Words: 0"
    return f"Characters: {len(cleaned):,} | Words: {len(cleaned.split()):,}"


# ── Gradio UI ────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Fake News Detector",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 920px !important; }",
    ) as demo:
        gr.Markdown(
            "# 📰 Fake News Detector\n"
            "### Powered by RoBERTa-Large\n"
            "Paste text **or** upload a file and classify instantly."
        )

        with gr.Row():
            text_input = gr.Textbox(
                label="Article Text",
                lines=10,
                placeholder="Paste article text here...",
            )

        gr.Examples(
            examples=[
                [
                    "Scientists at NASA have confirmed the discovery of "
                    "liquid water beneath the surface of Mars, marking a "
                    "historic breakthrough in planetary science."
                ],
                [
                    "BREAKING: Government secretly installing mind-control "
                    "chips in all new smartphones. A whistleblower reveals "
                    "the shocking truth mainstream media refuses to report."
                ],
                [
                    "The Federal Reserve announced a 0.25 percentage point "
                    "increase in its benchmark interest rate citing "
                    "persistent inflation concerns."
                ],
            ],
            inputs=[text_input],
            label="Try an example",
        )

        with gr.Row():
            upload = gr.File(
                label="Or upload a file",
                file_count="single",
                file_types=[".txt", ".pdf", ".docx"],
            )

        text_stats = gr.Markdown("Characters: 0 | Words: 0")

        with gr.Row():
            submit_btn = gr.Button("Analyze", variant="primary", size="lg")
            clear_btn = gr.Button("Reset")

        label_output = gr.Label(label="Prediction", num_top_classes=2)
        result_html = gr.HTML(
            value=default_result_html(), label="Analysis Result"
        )
        used_text_preview = gr.Textbox(
            label="Used Text (Preview)", lines=8, interactive=False
        )

        submit_btn.click(
            fn=run_prediction,
            inputs=[text_input, upload],
            outputs=[label_output, result_html, used_text_preview],
        )

        text_input.change(
            fn=get_text_stats,
            inputs=[text_input],
            outputs=[text_stats],
        )

        clear_btn.click(
            fn=lambda: (
                "",
                None,
                {},
                default_result_html(),
                "",
                "Characters: 0 | Words: 0",
            ),
            inputs=[],
            outputs=[
                text_input,
                upload,
                label_output,
                result_html,
                used_text_preview,
                text_stats,
            ],
        )

    return demo


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
