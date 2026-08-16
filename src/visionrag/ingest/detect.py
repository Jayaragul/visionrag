"""Object detection backends.

Models are replaceable plugins (PRD 3.3): the pipeline depends only on the
`Detector` protocol, so swapping detectors -- or sweeping thread counts to
trace the compute axis -- never touches downstream code.

Three backends:

* ``stub``        -- colour-threshold detector for the synthetic fixture. No
                     download, deterministic, so tests never depend on network
                     or on a specific checkpoint.
* ``torchvision`` -- SSDLite/MobileNetV3. CPU-friendly and available without
                     an export step; the sensible starting point.
* ``onnx``        -- ONNX Runtime, for the final efficiency numbers.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from ..config import DetectorConfig
from ..types import Box, Detection


class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """`image` is BGR uint8. Returns detections with normalised xyxy boxes."""
        ...


def _nms(dets: list[Detection], iou_thresh: float) -> list[Detection]:
    if len(dets) <= 1:
        return dets
    boxes = np.array([d.box for d in dets], dtype=np.float32)
    scores = np.array([d.score for d in dets], dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(1e-9, areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thresh]
    return [dets[i] for i in keep]


class StubDetector:
    """Finds saturated coloured blobs. Only meaningful on the synthetic clip."""

    name = "stub"

    # (label, HSV lower, HSV upper) -- matches the fixture's two rectangles.
    # OpenCV hue is 0-180. Orange BGR(0,140,255) -> ~16; green BGR(0,220,0)
    # -> ~60. Bands are kept wide but non-adjacent.
    _BANDS = [
        ("object_a", (5, 120, 120), (25, 255, 255)),    # orange, hue ~16
        ("object_b", (45, 120, 120), (75, 255, 255)),   # green,  hue ~60
    ]

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        out: list[Detection] = []
        for label, lo, hi in self._BANDS:
            mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                if cv2.contourArea(c) < 200:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                out.append(
                    Detection(
                        cls=label,
                        score=0.99,
                        box=(x / w, y / h, (x + bw) / w, (y + bh) / h),
                    )
                )
        return out[: self.cfg.max_detections]


class TorchvisionDetector:
    """SSDLite320-MobileNetV3 (COCO). Weights download once, then cache."""

    name = "torchvision"

    def __init__(self, cfg: DetectorConfig) -> None:
        import torch
        from torchvision.models import detection as tvdet

        self.cfg = cfg
        self._torch = torch
        # Pin thread count so CPU-second measurements are comparable across
        # machines and across sweep points.
        torch.set_num_threads(cfg.num_threads)
        torch.set_grad_enabled(False)

        builders = {
            "ssdlite320_mobilenet_v3_large": (
                tvdet.ssdlite320_mobilenet_v3_large,
                tvdet.SSDLite320_MobileNet_V3_Large_Weights,
            ),
            "fasterrcnn_mobilenet_v3_large_fpn": (
                tvdet.fasterrcnn_mobilenet_v3_large_fpn,
                tvdet.FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            ),
        }
        if cfg.model not in builders:
            raise ValueError(
                f"unsupported torchvision model {cfg.model!r}; "
                f"choose from {sorted(builders)}"
            )
        build, weights_enum = builders[cfg.model]
        weights = weights_enum.DEFAULT
        self._categories = weights.meta["categories"]
        self._model = build(weights=weights, box_score_thresh=cfg.score_thresh)
        self._model.eval()

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        with self._torch.inference_mode():
            pred = self._model([tensor])[0]

        keep_classes = set(self.cfg.classes) if self.cfg.classes else None
        out: list[Detection] = []
        boxes = pred["boxes"].numpy()
        labels = pred["labels"].numpy()
        scores = pred["scores"].numpy()
        for box, label, score in zip(boxes, labels, scores):
            if score < self.cfg.score_thresh:
                continue
            name = (
                self._categories[label]
                if 0 <= label < len(self._categories)
                else str(label)
            )
            if keep_classes and name not in keep_classes:
                continue
            x1, y1, x2, y2 = box
            out.append(
                Detection(
                    cls=name,
                    score=float(score),
                    box=(
                        float(np.clip(x1 / w, 0, 1)),
                        float(np.clip(y1 / h, 0, 1)),
                        float(np.clip(x2 / w, 0, 1)),
                        float(np.clip(y2 / h, 0, 1)),
                    ),
                )
            )
        out.sort(key=lambda d: d.score, reverse=True)
        return _nms(out, self.cfg.nms_iou)[: self.cfg.max_detections]


class OnnxDetector:
    """ONNX Runtime backend, assuming YOLOv8-style output ``[1, 4+C, N]``.

    Letterboxing is applied so aspect ratio is preserved; the inverse
    transform is undone before normalising, otherwise boxes drift on
    non-square inputs.
    """

    name = "onnx"

    def __init__(self, cfg: DetectorConfig) -> None:
        import onnxruntime as ort

        if not cfg.onnx_path:
            raise ValueError("detector.onnx_path is required for the onnx backend")
        self.cfg = cfg
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.num_threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            cfg.onnx_path, opts, providers=["CPUExecutionProvider"]
        )
        self._input = self._sess.get_inputs()[0].name
        self._labels = cfg.classes or [f"class_{i}" for i in range(1000)]

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        s = self.cfg.input_size
        h, w = image.shape[:2]
        scale = min(s / w, s / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        px, py = (s - nw) // 2, (s - nh) // 2
        canvas[py : py + nh, px : px + nw] = resized
        return canvas, scale, px, py

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        canvas, scale, px, py = self._letterbox(image)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]

        raw = self._sess.run(None, {self._input: blob})[0]
        pred = np.squeeze(raw)
        if pred.ndim != 2:
            raise RuntimeError(f"unexpected ONNX output shape {raw.shape}")
        # Orient to (N, 4+C).
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        xywh, cls_scores = pred[:, :4], pred[:, 4:]
        best = cls_scores.argmax(axis=1)
        conf = cls_scores.max(axis=1)
        keep = conf >= self.cfg.score_thresh

        out: list[Detection] = []
        for (cx, cy, bw, bh), cid, score in zip(
            xywh[keep], best[keep], conf[keep]
        ):
            # Undo letterbox, then normalise against the original frame.
            x1 = (cx - bw / 2 - px) / scale
            y1 = (cy - bh / 2 - py) / scale
            x2 = (cx + bw / 2 - px) / scale
            y2 = (cy + bh / 2 - py) / scale
            name = self._labels[cid] if cid < len(self._labels) else f"class_{cid}"
            out.append(
                Detection(
                    cls=name,
                    score=float(score),
                    box=(
                        float(np.clip(x1 / w, 0, 1)),
                        float(np.clip(y1 / h, 0, 1)),
                        float(np.clip(x2 / w, 0, 1)),
                        float(np.clip(y2 / h, 0, 1)),
                    ),
                )
            )
        out.sort(key=lambda d: d.score, reverse=True)
        return _nms(out, self.cfg.nms_iou)[: self.cfg.max_detections]


def build_detector(cfg: DetectorConfig) -> Detector:
    backends = {
        "stub": StubDetector,
        "torchvision": TorchvisionDetector,
        "onnx": OnnxDetector,
    }
    if cfg.backend not in backends:
        raise ValueError(
            f"unknown detector backend {cfg.backend!r}; choose from {sorted(backends)}"
        )
    return backends[cfg.backend](cfg)  # type: ignore[abstract]
