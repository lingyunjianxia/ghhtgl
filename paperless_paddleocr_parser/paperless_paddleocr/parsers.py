import logging
import os
import tempfile
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
import fitz  # PyMuPDF

from documents.parsers import DocumentParser, ParseError

logger = logging.getLogger(__name__)


class PaddleOCRParser(DocumentParser):
    # --- 插件元数据 ---
    name = "CustomOCR"
    version = "0.0.1"
    author = "Cycc"
    url = "https://blog.cycc.eu.org"

    # --- v3.0 插件框架必需的类属性 ---
    requires_pdf_rendition = True
    can_produce_archive = False

    # --- 环境变量配置 ---
    _api_url = os.getenv("PADDLEOCR_API_URL", "http://host.docker.internal:8001/ocr")
    _language = os.getenv("PADDLEOCR_LANGUAGE", "ch")
    _timeout = int(os.getenv("PADDLEOCR_TIMEOUT", 300))
    _dpi = int(os.getenv("PADDLEOCR_DPI", 200))
    _preprocess = os.getenv("PADDLEOCR_PREPROCESS", "fast")

    def __init__(self):
        self.logging_group = None
        self.config = {}
        self.filename = None
        self.text = None
        self.date = None
        self.archive_path = None
        self.thumbnail = None
        self._page_count = 0
        self._thumbnail_path = None

    # ---------- 上下文管理器 ----------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._thumbnail_path and Path(self._thumbnail_path).exists():
            try:
                Path(self._thumbnail_path).unlink()
            except Exception:
                pass

    # ---------- 框架要求的方法 ----------
    def configure(self, logging_group, config=None):
        self.logging_group = logging_group
        self.config = config or {}

    @classmethod
    def supported_mime_types(cls):
        return {
            "application/pdf": ".pdf",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/tiff": ".tif",
            "image/bmp": ".bmp",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }

    @classmethod
    def score(cls, mime_type: str, filename: str = None, document_path: Path = None) -> int:
        return 100 if mime_type in cls.supported_mime_types() else 0

    # ---------- parse 方法（已兼容 produce_archive 等） ----------
    def parse(
        self,
        document_path: Path,
        mime_type: str,
        file_name: str = None,
        produce_archive: bool = False,
        **kwargs
    ):
        logger.info(f"PaddleOCR 解析开始: {file_name or document_path.name}")
        try:
            if mime_type == "application/pdf":
                text, self._page_count = self._parse_pdf(document_path)
            else:
                text = self._parse_image(document_path)
                self._page_count = 1

            if not text or not text.strip():
                logger.warning("OCR 未提取到任何文本")
                text = ""

            self.text = text
            self.archive_path = None
            self.date = None
            logger.info(f"提取文本成功，共 {len(text)} 字符")
        except Exception as e:
            raise ParseError(f"解析文档失败: {e}") from e

    # ---------- get_thumbnail 方法 ----------
    def get_thumbnail(self, document_path: Path, mime_type: str, file_name: str = None):
        try:
            if mime_type == "application/pdf":
                doc = fitz.open(document_path)
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
            else:
                img = Image.open(document_path)

            img.thumbnail((200, 200))
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img.save(tmp, format="JPEG")
                self._thumbnail_path = tmp.name
            return self._thumbnail_path
        except Exception as e:
            logger.error(f"生成缩略图失败: {e}")
            return None

    # ---------- get_page_count 方法（修复：接受任意参数） ----------
    def get_page_count(self, *args, **kwargs) -> int:
        return self._page_count

    # ========== 内部 OCR 逻辑 ==========

    def _parse_pdf(self, pdf_path: Path):
        full_text = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise ParseError(f"无法打开 PDF: {e}")

        total_pages = len(doc)
        logger.info(f"PDF 总页数: {total_pages}")
        zoom = self._dpi / 72.0

        for page_num in range(total_pages):
            try:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                img_bytes = BytesIO()
                img.save(img_bytes, format="JPEG", quality=85)
                img_bytes.seek(0)

                logger.debug(f"处理第 {page_num + 1}/{total_pages} 页")
                page_text = self._call_ocr_api(img_bytes, f"{pdf_path.name}_page{page_num + 1}.jpg")
                if page_text and page_text.strip():
                    full_text.append(page_text)
                else:
                    logger.warning(f"第 {page_num + 1} 页 OCR 返回空文本")
            except Exception as e:
                logger.error(f"处理第 {page_num + 1} 页出错: {e}")
                continue

        doc.close()
        return "\n\n".join(full_text), total_pages

    def _parse_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            img_bytes = BytesIO(f.read())
        return self._call_ocr_api(img_bytes, image_path.name)

    def _call_ocr_api(self, image_bytes: BytesIO, file_name: str) -> str:
        try:
            files = {"file": (file_name, image_bytes, "image/jpeg")}
            data = {"language": self._language, "preprocess": self._preprocess}

            response = requests.post(self._api_url, files=files, data=data, timeout=self._timeout)
            response.raise_for_status()
            return self._extract_text_from_response(response.json())
        except requests.exceptions.RequestException as e:
            raise ParseError(f"调用 PaddleOCR API 失败: {e}") from e
        except ValueError as e:
            raise ParseError(f"解析 API 响应失败: {e}") from e

    def _extract_text_from_response(self, response_data: dict) -> str:
        if "text" in response_data:
            return response_data["text"].strip()

        if "results" in response_data and isinstance(response_data["results"], list):
            texts = []
            for item in response_data["results"]:
                if isinstance(item, dict):
                    if "text" in item:
                        texts.append(item["text"])
                    elif "data" in item and isinstance(item["data"], list):
                        for data_item in item["data"]:
                            if isinstance(data_item, list) and len(data_item) >= 2:
                                texts.append(str(data_item[1][0] if isinstance(data_item[1], list) else data_item[1]))
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(texts).strip()

        if "result" in response_data:
            result = response_data["result"]
            if isinstance(result, str):
                return result.strip()
            elif isinstance(result, list):
                texts = []
                for item in result:
                    if isinstance(item, list) and len(item) >= 2:
                        if isinstance(item[1], (list, tuple)) and len(item[1]) >= 1:
                            texts.append(str(item[1][0]))
                        elif isinstance(item[1], str):
                            texts.append(item[1])
                    elif isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                return "\n".join(texts).strip()

        if "data" in response_data:
            data = response_data["data"]
            if isinstance(data, dict) and "prunedResult" in data:
                texts = []
                for page in data["prunedResult"]:
                    if isinstance(page, list):
                        for item in page:
                            if isinstance(item, list) and len(item) >= 2:
                                texts.append(str(item[1][0] if isinstance(item[1], list) else item[1]))
                return "\n".join(texts).strip()

        logger.warning(f"未知 API 响应格式，响应键: {list(response_data.keys())}")
        return ""