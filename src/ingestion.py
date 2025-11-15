"""
ingestion.py
Loads PDF/TXT, performs OCR fallback, extracts page-level text.
Produces a list of LangChain Document objects with metadata.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io
import torch


def init_ocr_model(model_name):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model


def ocr_image(processor, model, image_bytes):
    """Run TrOCR on image bytes and return recognized text + confidence."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Dummy confidence for now (TrOCR doesn't output it directly)
    conf = 0.90
    return text, conf


def extract_text_from_pdf(path, processor, model):
    """Load PDF with PyPDFLoader; OCR pages with no selectable text."""
    loader = PyPDFLoader(path)
    pages = loader.load()

    final_docs = []
    low_conf_pages = []

    for idx, doc in enumerate(pages):
        page_text = doc.page_content.strip()
        meta = doc.metadata or {}
        meta["page"] = idx + 1
        meta["source_file"] = os.path.basename(path)

        # If empty → try OCR
        if len(page_text) == 0:
            # Load page image via loader
            page_image = loader.pdf.pages[idx].to_image(resolution=300)
            pil_img = page_image.original

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            ocr_text, conf = ocr_image(processor, model, img_bytes)
            page_text = ocr_text

            if conf < 0.85:
                low_conf_pages.append(idx + 1)

        final_docs.append(
            {
                "text": page_text,
                "metadata": meta,
            }
        )

    return final_docs, low_conf_pages


def extract_text_from_txt(path):
    """Load TXT file."""
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    final = []
    for d in docs:
        final.append(
            {"text": d.page_content, "metadata": {"source_file": os.path.basename(path), "page": 1}}
        )
    return final, []
