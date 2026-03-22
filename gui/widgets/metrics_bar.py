from PySide6.QtWidgets import QWidget, QVBoxLayout, QMenu

from core.monitoring.collectors import MetricsCollector
from core.monitoring.metrics_store import MetricsStore
from core.monitoring.system_metrics import SystemMonitor
from gui.widgets.visualizations import SparklineVisualization


class MetricsBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolTip("Right click to stop/start monitoring")

        monitor = SystemMonitor()
        self._has_nvidia_gpu = monitor.is_nvidia_gpu_available()
        monitor.shutdown()

        self.metrics_store = MetricsStore()

        self._init_ui()
        self._start_metrics_collector()

    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.visualization = SparklineVisualization(
            self.metrics_store, self._has_nvidia_gpu
        )
        self.visualization.setToolTip("Right click to stop/start monitoring")
        self.layout.addWidget(self.visualization)

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        is_running = hasattr(self, 'metrics_collector') and self.metrics_collector.isRunning()
        control_action = menu.addAction("Stop Monitoring" if is_running else "Start Monitoring")

        action = menu.exec_(event.globalPos())

        if action == control_action:
            if is_running:
                self._stop_metrics_collector()
            else:
                self._start_metrics_collector()

    def _start_metrics_collector(self):
        if hasattr(self, 'metrics_collector'):
            self._stop_metrics_collector()

        self.metrics_collector = MetricsCollector()
        self.metrics_collector.metrics_updated.connect(
            lambda metrics: self.metrics_store.add_metrics(metrics)
        )
        self.metrics_collector.start()

    def _stop_metrics_collector(self):
        if hasattr(self, 'metrics_collector'):
            self.metrics_collector.stop()
            self.metrics_collector.wait()
            self.metrics_collector.cleanup()
            self.metrics_collector.deleteLater()
            del self.metrics_collector

    def cleanup(self):
        self._stop_metrics_collector()
        self.visualization.cleanup()
