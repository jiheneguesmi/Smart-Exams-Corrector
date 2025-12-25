"""
French Exam OCR Pipeline - Extracted Module
Handles image preprocessing, line segmentation, and text recognition
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from enum import Enum

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelType(Enum):
    ENGLISH = "english"
    FRENCH = "french"
    FRENCH_PRINTED = "french_printed"


@dataclass
class OCRConfig:
    model_type: ModelType = ModelType.FRENCH
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    remove_watermark: bool = True
    watermark_ratio: float = 0.08
    remove_blue_lines: bool = True
    denoise_strength: int = 10
    min_line_height: int = 12
    max_line_height: int = 70
    min_line_gap: int = 8
    projection_threshold_ratio: float = 0.15
    margin_horizontal: int = 20
    target_line_height: int = 48
    max_length: int = 384
    num_beams: int = 6
    output_dir: str = "ocr_output"
    save_debug_images: bool = False


@dataclass
class LineSegment:
    index: int
    image: np.ndarray
    bbox: Tuple[int, int, int, int]
    text: str = ""
    confidence: float = 0.0


@dataclass
class PageResult:
    page_path: str
    lines: List[LineSegment] = field(default_factory=list)
    full_text: str = ""
    success: bool = True
    error: Optional[str] = None


@dataclass
class ExamResult:
    exam_id: str
    pages: List[PageResult] = field(default_factory=list)
    full_text: str = ""

    def to_dict(self) -> dict:
        return {
            "exam_id": self.exam_id,
            "full_text": self.full_text,
            "pages": [{"path": p.page_path, "text": p.full_text, "line_count": len(p.lines)} for p in self.pages]
        }


class ImagePreprocessor:
    def __init__(self, config: OCRConfig):
        self.config = config

    def remove_blue_lines(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([85, 20, 100])
        upper_blue = np.array([135, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        line_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_h)
        line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=2)
        result = img.copy()
        result[line_mask > 0] = [255, 255, 255]
        return result

    def remove_watermark(self, img: np.ndarray) -> np.ndarray:
        h = img.shape[0]
        result = img.copy()
        cutoff = int(h * (1 - self.config.watermark_ratio))
        if len(img.shape) == 2:
            result[cutoff:, :] = 255
        else:
            result[cutoff:, :] = [255, 255, 255]
        return result

    def remove_small_components(self, binary: np.ndarray, min_size: int = 15) -> np.ndarray:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        cleaned = np.zeros_like(binary)
        components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            components.append({
                'id': i, 'area': area,
                'x': stats[i, cv2.CC_STAT_LEFT], 'y': stats[i, cv2.CC_STAT_TOP],
                'w': w, 'h': h,
                'aspect_ratio': w / h if h > 0 else 0,
                'centroid': centroids[i]
            })
        for comp in components:
            keep = comp['area'] >= min_size
            if not keep and comp['area'] >= 8 and 0.5 <= comp['aspect_ratio'] <= 2.0:
                dot_y, dot_x = comp['y'] + comp['h'], comp['x'] + comp['w'] / 2
                for other in components:
                    if other['id'] != comp['id']:
                        ox, oy = other['x'] + other['w'] / 2, other['y']
                        if 0 < oy - dot_y < 30 and abs(ox - dot_x) < 15 and other['h'] > other['w']:
                            keep = True
                            break
            if keep:
                cleaned[labels == comp['id']] = 255
        return cleaned

    def preprocess(self, image_path: str) -> Tuple[Image.Image, np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        if self.config.remove_watermark:
            img = self.remove_watermark(img)
        img = cv2.fastNlMeansDenoisingColored(img, None, self.config.denoise_strength, 10, 7, 21)
        if self.config.remove_blue_lines:
            img = self.remove_blue_lines(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 12)
        binary = self.remove_small_components(binary, min_size=15)
        if np.sum(binary == 255) < binary.size // 2:
            binary = cv2.bitwise_not(binary)
        rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb), img


class LineSegmenter:
    def __init__(self, config: OCRConfig):
        self.config = config

    def segment(self, image: Union[Image.Image, np.ndarray],
                original_img: Optional[np.ndarray] = None) -> List[LineSegment]:
        if isinstance(image, Image.Image):
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        h_proj = np.sum(binary, axis=1)
        mean_proj = np.mean(h_proj[h_proj > 0]) if np.any(h_proj > 0) else 0
        threshold = mean_proj * self.config.projection_threshold_ratio
        boundaries = self._find_line_boundaries(h_proj, threshold)
        boundaries = self._split_oversized_lines(boundaries, h_proj, mean_proj)
        boundaries = self._merge_close_lines(boundaries)
        source_img = original_img if original_img is not None else img
        return self._extract_lines(boundaries, binary, source_img)

    def _find_line_boundaries(self, projection: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
        boundaries = []
        in_line = False
        start = 0
        for i, val in enumerate(projection):
            if val > threshold and not in_line:
                if not boundaries or (i - boundaries[-1][1]) >= self.config.min_line_gap:
                    start = i
                    in_line = True
            elif val <= threshold and in_line:
                if i - start > self.config.min_line_height:
                    boundaries.append((start, i))
                in_line = False
        if in_line and len(projection) - start > self.config.min_line_height:
            boundaries.append((start, len(projection)))
        return boundaries

    def _split_oversized_lines(self, boundaries: List[Tuple[int, int]],
                                projection: np.ndarray, mean_proj: float) -> List[Tuple[int, int]]:
        result = []
        for start, end in boundaries:
            height = end - start
            if height > self.config.max_line_height:
                region = projection[start:end]
                inverted = -region
                peaks, _ = find_peaks(inverted, distance=20, prominence=mean_proj * 0.1)
                if len(peaks) > 0:
                    split_points = [0] + list(peaks) + [len(region)]
                    for j in range(len(split_points) - 1):
                        s, e = start + split_points[j], start + split_points[j + 1]
                        if e - s > self.config.min_line_height:
                            result.append((s, e))
                else:
                    mid_start, mid_end = height // 3, 2 * height // 3
                    mid_section = region[mid_start:mid_end]
                    if len(mid_section) > 0:
                        split = start + mid_start + np.argmin(mid_section)
                        if split - start > self.config.min_line_height:
                            result.append((start, split))
                        if end - split > self.config.min_line_height:
                            result.append((split, end))
                    else:
                        result.append((start, end))
            else:
                result.append((start, end))
        return result

    def _merge_close_lines(self, boundaries: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not boundaries:
            return []
        merged = []
        i = 0
        while i < len(boundaries):
            start, end = boundaries[i]
            while i + 1 < len(boundaries) and boundaries[i + 1][0] - end < 5:
                end = boundaries[i + 1][1]
                i += 1
            merged.append((start, end))
            i += 1
        return merged

    def _extract_lines(self, boundaries: List[Tuple[int, int]],
                       binary: np.ndarray, source_img: np.ndarray) -> List[LineSegment]:
        lines = []
        for idx, (start, end) in enumerate(boundaries):
            height = end - start
            margin_v = max(3, min(8, height // 8))
            y1 = max(0, start - margin_v)
            y2 = min(source_img.shape[0], end + margin_v)
            line_region = binary[y1:y2, :]
            v_proj = np.sum(line_region, axis=0)
            non_zero = np.where(v_proj > 0)[0]
            if len(non_zero) > 0:
                x1 = max(0, non_zero[0] - self.config.margin_horizontal)
                x2 = min(source_img.shape[1], non_zero[-1] + self.config.margin_horizontal)
            else:
                x1, x2 = 0, source_img.shape[1]
            line_img = source_img[y1:y2, x1:x2]
            lines.append(LineSegment(index=idx, image=line_img, bbox=(x1, y1, x2, y2)))
        return lines


class TrOCREngine:
    MODEL_CONFIGS = {
        ModelType.ENGLISH: {
            "processor": "microsoft/trocr-large-handwritten",
            "model": "microsoft/trocr-large-handwritten",
            "tokenizer": None
        },
        ModelType.FRENCH: {
            "processor": "microsoft/trocr-large-handwritten",
            "model": "agomberto/trocr-large-handwritten-fr",
            "tokenizer": "agomberto/trocr-large-handwritten-fr"
        },
        ModelType.FRENCH_PRINTED: {
            "processor": "microsoft/trocr-base-handwritten",
            "model": "agomberto/trocr-base-printed-fr",
            "tokenizer": "agomberto/trocr-base-printed-fr"
        }
    }

    def __init__(self, config: OCRConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._load_model()

    def _load_model(self):
        model_config = self.MODEL_CONFIGS[self.config.model_type]
        logger.info(f"Loading model: {model_config['model']} on {self.device}")
        self.processor = TrOCRProcessor.from_pretrained(model_config["processor"])
        self.model = VisionEncoderDecoderModel.from_pretrained(model_config["model"])
        self.model.to(self.device)
        self.model.eval()
        if model_config["tokenizer"]:
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer"])
        else:
            self.tokenizer = None
        logger.info(f"✅ Model loaded on {self.device}")

    def recognize(self, line: LineSegment) -> str:
        img = line.image
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        target_h = self.config.target_line_height
        if h < target_h:
            scale = target_h / h
            img = cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
        elif h > target_h * 1.5:
            scale = target_h / h
            img = cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(img)
        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                early_stopping=True,
                length_penalty=1.5,
                no_repeat_ngram_size=4
            )
        if self.tokenizer:
            text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def recognize_batch(self, lines: List[LineSegment]) -> List[str]:
        results = []
        for line in lines:
            try:
                text = self.recognize(line)
                line.text = text
                results.append(text)
            except Exception as e:
                logger.warning(f"Error recognizing line {line.index}: {e}")
                results.append("")
        return results


class ExamOCR:
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.preprocessor = ImagePreprocessor(self.config)
        self.segmenter = LineSegmenter(self.config)
        self.ocr_engine = TrOCREngine(self.config)
        os.makedirs(self.config.output_dir, exist_ok=True)

    def process_page(self, image_path: str, visualize: bool = False) -> PageResult:
        logger.info(f"Processing: {image_path}")
        result = PageResult(page_path=image_path)
        try:
            processed_img, original_img = self.preprocessor.preprocess(image_path)
            lines = self.segmenter.segment(processed_img, original_img)
            logger.info(f"Detected {len(lines)} lines")
            if not lines:
                result.error = "No lines detected"
                result.success = False
                return result
            texts = self.ocr_engine.recognize_batch(lines)
            result.lines = lines
            result.full_text = "\n".join(t for t in texts if t)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            result.error = str(e)
            result.success = False
        return result

    def process_exam(self, page_paths: List[str], exam_id: str = "exam") -> ExamResult:
        result = ExamResult(exam_id=exam_id)
        for path in page_paths:
            page_result = self.process_page(path)
            result.pages.append(page_result)
        result.full_text = "\n\n--- Page Break ---\n\n".join(
            p.full_text for p in result.pages if p.full_text
        )
        return result

    def process_batch(self, exams: Dict[str, List[str]], save_results: bool = True) -> Dict[str, ExamResult]:
        results = {}
        for exam_id, pages in exams.items():
            logger.info(f"\n{'='*60}\nProcessing Exam: {exam_id}\n{'='*60}")
            result = self.process_exam(pages, exam_id)
            results[exam_id] = result
            if save_results:
                self._save_result(result)
        return results

    def _save_result(self, result: ExamResult):
        base_path = os.path.join(self.config.output_dir, result.exam_id)
        with open(f"{base_path}.txt", "w", encoding="utf-8-sig") as f:
            f.write(result.full_text)
        with open(f"{base_path}.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved: {base_path}.txt")
