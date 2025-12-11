from __future__ import annotations

import logging
import os
import re
from typing import Dict, List

import cv2
import fitz  # PyMuPDF
import numpy as np

from backend.core.config import CONFIG

logger = logging.getLogger(__name__)


class PDFService:
    def extract_faces(self, pdf_path: str, output_dir: str | None = None) -> Dict[str, str]:
        output_dir = output_dir or os.path.join(CONFIG.tmp_dir, "pdf_faces")
        os.makedirs(output_dir, exist_ok=True)

        doc = fitz.open(pdf_path)

        # Collect text to extract student numbers (11 digits)
        all_text = "\n".join([p.get_text("text", sort=True) for p in doc])
        all_text = all_text.encode('utf-8', 'ignore').decode('utf-8')
        ids = re.findall(r"(\d{11})", all_text)

        index = 0
        mapping: Dict[str, str] = {}
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images):
                xref = img[0]
                base = doc.extract_image(xref)
                data = base["image"]
                ext = base["ext"]
                w = base.get("width", 0)
                h = base.get("height", 0)

                if w < 150 or h < 150:
                    continue

                if index < len(ids):
                    sid = ids[index]
                    index += 1
                else:
                    sid = f"Unknown_{page_num+1}_{img_idx+1}"

                arr = np.frombuffer(data, dtype=np.uint8)
                im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if im is None:
                    continue

                out_path = os.path.join(output_dir, f"{sid}.{ext}")
                cv2.imwrite(out_path, im)
                mapping[sid] = out_path

        doc.close()
        return mapping
