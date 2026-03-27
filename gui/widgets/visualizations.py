from collections import deque

from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QVBoxLayout
from PySide6.QtGui import QPainter, QColor, QPainterPath, QPen, QPixmap, QLinearGradient

from core.monitoring.metrics_store import MetricsStore
from core.monitoring.system_metrics import SystemMetrics

METRIC_DEFS = [
    ("cpu",   "CPU",       "#FF4136", lambda m: m.cpu_usage,           False),
    ("ram",   "RAM",       "#B10DC9", lambda m: m.ram_usage_percent,   False),
    ("gpu",   "GPU",       "#0074D9", lambda m: m.gpu_utilization,     True),
    ("vram",  "VRAM",      "#2ECC40", lambda m: m.vram_usage_percent,  True),
    ("power", "GPU Power", "#FFD700", lambda m: m.power_usage_percent, True),
]


class Sparkline(QWidget):

    def __init__(self, max_values: int = 125, color: str = "#0074D9"):
        super().__init__()
        self.max_values = max_values
        self.values = deque(maxlen=max_values)
        self.setFixedSize(125, 65)
        self.color = QColor(color)
        self._gradient_pixmap = self._create_gradient()

    def add_value(self, value: float):
        self.values.append(value)
        self.update()

    def _create_gradient(self):
        pixmap = QPixmap(1, self.height())
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        gradient = QLinearGradient(0, 0, 0, self.height())

        fill_color = QColor(self.color)
        fill_color.setAlpha(60)
        gradient.setColorAt(0, fill_color)
        gradient.setColorAt(1, QColor(0, 0, 0, 0))

        painter.fillRect(pixmap.rect(), gradient)
        painter.end()

        return pixmap

    def paintEvent(self, event):
        if not self.values:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        margin = 5

        path = QPainterPath()
        x_step = (width - 2 * margin) / (len(self.values) - 1) if len(self.values) > 1 else 0
        points = []

        for i, value in enumerate(self.values):
            x = margin + i * x_step
            y = height - margin - (value / 100) * (height - 2 * margin)
            points.append(QPointF(x, y))

            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        fill_path = QPainterPath(path)
        if points:
            fill_path.lineTo(points[-1].x(), height - margin)
            fill_path.lineTo(points[0].x(), height - margin)
            fill_path.closeSubpath()

        painter.save()
        painter.setClipPath(fill_path)
        for x in range(0, width, self._gradient_pixmap.width()):
            painter.drawPixmap(x, 0, self._gradient_pixmap)
        painter.restore()

        painter.setPen(QPen(self.color, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)


class SparklineVisualization(QWidget):

    def __init__(self, metrics_store: MetricsStore, has_nvidia_gpu: bool):
        super().__init__()
        self.metrics_store = metrics_store
        self.has_nvidia_gpu = has_nvidia_gpu
        self._sparklines = {}
        self._labels = {}
        self._init_ui()
        self.metrics_store.metrics_ready.connect(self.update_metrics)

    def _active_metrics(self):
        return [m for m in METRIC_DEFS if not m[4] or self.has_nvidia_gpu]

    def _init_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(1)
        layout.setContentsMargins(1, 1, 1, 1)

        for col, (key, label_text, color, _, _) in enumerate(self._active_metrics()):
            widget = QWidget()
            vlayout = QVBoxLayout(widget)
            vlayout.setSpacing(1)
            vlayout.setContentsMargins(0, 0, 0, 0)

            sparkline = Sparkline(color=color)
            vlayout.addWidget(sparkline, alignment=Qt.AlignCenter)

            label = QLabel(f"{label_text} 0.0%")
            label.setAlignment(Qt.AlignCenter)
            vlayout.addWidget(label, alignment=Qt.AlignCenter)

            layout.addWidget(widget, 0, col)
            self._sparklines[key] = sparkline
            self._labels[key] = (label_text, label)

        for i in range(layout.columnCount()):
            layout.setColumnStretch(i, 1)

    def update_metrics(self, metrics: SystemMetrics):
        for key, _, _, accessor, _ in self._active_metrics():
            value = accessor(metrics)
            if value is not None:
                self._sparklines[key].add_value(value)
                label_text, label = self._labels[key]
                label.setText(f"{label_text} {value:.1f}%")

    def cleanup(self):
        self.metrics_store.metrics_ready.disconnect(self.update_metrics)
