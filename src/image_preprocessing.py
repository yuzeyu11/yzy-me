from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


class ImagePreprocessor:
    """图像预处理模块：集成试卷矫正算法，包括透视矫正、旋转矫正和增强。"""

    def __init__(self, contrast: float = 1.3, brightness: float = 1.1, sharpness: float = 1.0):
        self.contrast = contrast
        self.brightness = brightness
        self.sharpness = sharpness

    def correct_image(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image)
        image = ImageOps.autocontrast(image)
        return image

    def enhance_image(self, image: Image.Image) -> Image.Image:
        image = self.correct_image(image)
        image = ImageEnhance.Contrast(image).enhance(self.contrast)
        image = ImageEnhance.Brightness(image).enhance(self.brightness)
        image = ImageEnhance.Sharpness(image).enhance(self.sharpness)
        return image

    def _pil_to_cv2(self, image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def _cv2_to_pil(self, image: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def detect_paper_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """检测试卷轮廓，用于透视矫正。"""
        if cv2 is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                return approx
        return None

    def perspective_correction(self, image: np.ndarray) -> np.ndarray:
        """透视矫正试卷。"""
        if cv2 is None:
            return image

        contour = self.detect_paper_contour(image)
        if contour is None:
            return image

        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def rotation_correction(self, image: np.ndarray) -> np.ndarray:
        """旋转矫正试卷。"""
        if cv2 is None:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi - 90
                angles.append(angle)

            median_angle = np.median(angles)
            if abs(median_angle) > 1:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return image

    def preprocess_image_file(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        image = Image.open(input_path)
        cv_image = self._pil_to_cv2(image)
        corrected = self.perspective_correction(cv_image)
        corrected = self.rotation_correction(corrected)
        result = self._cv2_to_pil(corrected)
        result = self.enhance_image(result)

        if output_path is None:
            output_path = input_path.with_name(f"preprocessed_{input_path.name}")
        result.save(output_path)
        return output_path

    def ocr_image(self, input_path: Path) -> str:
        image = Image.open(input_path)
        cv_image = self._pil_to_cv2(image)
        corrected = self.perspective_correction(cv_image)
        corrected = self.rotation_correction(corrected)
        result = self._cv2_to_pil(corrected)
        result = self.enhance_image(result)

        if pytesseract is None:
            return (
                "[OCR 未启用] 请安装 pytesseract 以支持图像 OCR。"
                " 当前仅返回预处理后的图像提示。"
            )
        return pytesseract.image_to_string(result, lang="chi_sim+eng").strip() or ""


def ocr_image(path: Path) -> str:
    return ImagePreprocessor().ocr_image(path)

