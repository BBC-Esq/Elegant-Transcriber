from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider, QCheckBox, QPushButton, QDialog,
    QDialogButtonBox, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QSettings

from config.constants import (
    MODEL_NAMES, MODEL_PRECISIONS, MODEL_TOOLTIPS, ALL_MODELS,
    DEFAULT_SEGMENT_LENGTH, DEFAULT_SEGMENT_DURATION,
    SUPPORTED_AUDIO_EXTENSIONS, TIMESTAMP_FORMATS,
    CANARY_MAX_CHUNK_LENGTH,
)
from utils.system_utils import get_compute_and_platform_info, has_bfloat16_support


class ExtensionsDialog(QDialog):
    def __init__(self, current_extensions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("File Extensions")
        self.setMinimumWidth(350)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Select which audio file extensions to include when processing.\n"
            "All extensions are enabled by default."
        ))

        grid = QGridLayout()
        self.checkboxes = {}
        for i, ext in enumerate(SUPPORTED_AUDIO_EXTENSIONS):
            cb = QCheckBox(ext)
            cb.setChecked(ext in current_extensions)
            self.checkboxes[ext] = cb
            grid.addWidget(cb, i // 4, i % 4)
        layout.addLayout(grid)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected(self):
        return [ext for ext, cb in self.checkboxes.items() if cb.isChecked()]


class SettingsWidget(QGroupBox):

    device_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__("Settings", parent)
        self._bfloat16_supported = has_bfloat16_support()
        self._settings = QSettings("ElegantAudioTranscriber", "ElegantAudioTranscriber")
        self._selected_extensions = list(SUPPORTED_AUDIO_EXTENSIONS)
        self._init_ui()
        self._populate_devices()
        self._load_settings()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for name in MODEL_NAMES:
            self.model_combo.addItem(name)
        for i, name in enumerate(MODEL_NAMES):
            tooltip = MODEL_TOOLTIPS.get(name, "")
            if tooltip:
                self.model_combo.setItemData(i, tooltip, Qt.ToolTipRole)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        row1.addWidget(self.model_combo)

        row1.addWidget(QLabel("Precision:"))
        self.precision_combo = QComboBox()
        row1.addWidget(self.precision_combo)

        row1.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        row1.addWidget(self.device_combo)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.timestamps_checkbox = QCheckBox("Include Timestamps")
        self.timestamps_checkbox.toggled.connect(self._on_timestamps_toggled)
        row2.addWidget(self.timestamps_checkbox)

        self.ts_format_combo = QComboBox()
        self.ts_format_combo.addItems(TIMESTAMP_FORMATS)
        self.ts_format_combo.setEnabled(False)
        row2.addWidget(self.ts_format_combo)

        row2.addStretch()

        self.extensions_button = QPushButton("File Extensions")
        self.extensions_button.clicked.connect(self._open_extensions_dialog)
        row2.addWidget(self.extensions_button)
        layout.addLayout(row2)

        row_dur = QHBoxLayout()
        row_dur.addWidget(QLabel("Timestamp Intervals (seconds):"))
        self.seg_duration_slider = QSlider(Qt.Horizontal)
        self.seg_duration_slider.setMinimum(5)
        self.seg_duration_slider.setMaximum(60)
        self.seg_duration_slider.setValue(DEFAULT_SEGMENT_DURATION)
        self.seg_duration_slider.setTickPosition(QSlider.TicksBelow)
        self.seg_duration_slider.setTickInterval(5)
        self.seg_duration_slider.setEnabled(False)
        row_dur.addWidget(self.seg_duration_slider)
        self.seg_duration_label = QLabel(str(DEFAULT_SEGMENT_DURATION))
        self.seg_duration_label.setEnabled(False)
        self.seg_duration_slider.valueChanged.connect(
            lambda v: self.seg_duration_label.setText(str(v))
        )
        row_dur.addWidget(self.seg_duration_label)
        layout.addLayout(row_dur)

        chunk_tooltip = (
            "Controls how much audio is processed at once.\n"
            "GPU (CUDA): 80-90 is ideal. Values outside 70-90 may increase VRAM usage or slow processing.\n"
            "CPU: 20-30 is ideal. Values above 30 may drastically increase processing time and RAM usage."
        )

        row3 = QHBoxLayout()
        chunk_label = QLabel("Audio Chunk Size (seconds):")
        chunk_label.setToolTip(chunk_tooltip)
        row3.addWidget(chunk_label)
        self.segment_slider = QSlider(Qt.Horizontal)
        self.segment_slider.setMinimum(10)
        self.segment_slider.setMaximum(100)
        self.segment_slider.setValue(DEFAULT_SEGMENT_LENGTH)
        self.segment_slider.setTickPosition(QSlider.TicksBelow)
        self.segment_slider.setTickInterval(10)
        self.segment_slider.setToolTip(chunk_tooltip)
        row3.addWidget(self.segment_slider)

        self.segment_label = QLabel(str(DEFAULT_SEGMENT_LENGTH))
        self.segment_slider.valueChanged.connect(
            lambda v: self.segment_label.setText(str(v))
        )
        row3.addWidget(self.segment_label)
        layout.addLayout(row3)

    def _populate_devices(self):
        devices = get_compute_and_platform_info()
        self.device_combo.addItems(devices)
        if "cuda" in devices:
            self.device_combo.setCurrentText("cuda")
        else:
            self.device_combo.setCurrentText("cpu")
        self._update_precision_options()

    def _on_device_changed(self, device: str):
        self.device_changed.emit(device)
        self._update_precision_options()

    def _on_model_changed(self, model_name: str):
        self._update_precision_options()
        self._update_model_constraints(model_name)

    def _on_timestamps_toggled(self, checked: bool):
        self.ts_format_combo.setEnabled(checked)
        self.seg_duration_slider.setEnabled(checked)
        self.seg_duration_label.setEnabled(checked)

    def _open_extensions_dialog(self):
        dlg = ExtensionsDialog(self._selected_extensions, self)
        if dlg.exec() == QDialog.Accepted:
            self._selected_extensions = dlg.get_selected()

    def _update_precision_options(self):
        model_name = self.model_combo.currentText()
        if not model_name:
            return
        current_device = self.device_combo.currentText()
        model_precs = MODEL_PRECISIONS.get(model_name, [])
        if current_device == "cpu":
            available = [p for p in model_precs if p == "float32"]
        else:
            available = []
            for p in model_precs:
                if p == "bfloat16" and not self._bfloat16_supported:
                    continue
                available.append(p)
        previous = self.precision_combo.currentText()
        self.precision_combo.blockSignals(True)
        self.precision_combo.clear()
        self.precision_combo.addItems(available)
        if previous in available:
            self.precision_combo.setCurrentText(previous)
        self.precision_combo.blockSignals(False)

    def _get_model_type(self, model_name: str) -> str:
        for key, info in ALL_MODELS.items():
            if info['name'] == model_name:
                return info.get('model_type', 'parakeet')
        return 'parakeet'

    def _update_model_constraints(self, model_name: str):
        model_type = self._get_model_type(model_name)
        is_canary = model_type == "canary"

        if is_canary:
            self.segment_slider.setMaximum(CANARY_MAX_CHUNK_LENGTH)
            if self.segment_slider.value() > CANARY_MAX_CHUNK_LENGTH:
                self.segment_slider.setValue(CANARY_MAX_CHUNK_LENGTH)
            self.timestamps_checkbox.setChecked(False)
            self.timestamps_checkbox.setEnabled(False)
            self.timestamps_checkbox.setToolTip("Timestamps are not supported by Canary-Qwen.")
        else:
            self.segment_slider.setMaximum(100)
            self.timestamps_checkbox.setEnabled(True)
            self.timestamps_checkbox.setToolTip("")

    def get_model(self) -> str:
        return f"{self.model_combo.currentText()} - {self.precision_combo.currentText()}"

    def get_device(self) -> str:
        return self.device_combo.currentText()

    def get_output_format(self) -> str:
        if self.timestamps_checkbox.isChecked():
            return self.ts_format_combo.currentText()
        return "txt"

    def get_segment_length(self) -> int:
        return self.segment_slider.value()

    def get_segment_duration(self) -> int:
        return self.seg_duration_slider.value()

    def get_timestamps(self) -> bool:
        return self.timestamps_checkbox.isChecked()

    def get_selected_extensions(self) -> list:
        return list(self._selected_extensions)

    def save_settings(self):
        self._settings.setValue("model", self.model_combo.currentText())
        self._settings.setValue("precision", self.precision_combo.currentText())
        self._settings.setValue("device", self.device_combo.currentText())
        self._settings.setValue("segment_length", self.segment_slider.value())
        self._settings.setValue("timestamps", self.timestamps_checkbox.isChecked())
        self._settings.setValue("seg_duration", self.seg_duration_slider.value())
        self._settings.setValue("ts_format", self.ts_format_combo.currentText())
        self._settings.setValue("extensions", self._selected_extensions)

    def _load_settings(self):
        devices = [self.device_combo.itemText(i) for i in range(self.device_combo.count())]

        saved_device = self._settings.value("device", "")
        if saved_device in devices:
            self.device_combo.setCurrentText(saved_device)

        saved_model = self._settings.value("model", "")
        if saved_model in MODEL_NAMES:
            self.model_combo.setCurrentText(saved_model)

        saved_precision = self._settings.value("precision", "")
        if saved_precision:
            avail = [self.precision_combo.itemText(i) for i in range(self.precision_combo.count())]
            if saved_precision in avail:
                self.precision_combo.setCurrentText(saved_precision)

        saved_seg = self._settings.value("segment_length", DEFAULT_SEGMENT_LENGTH, type=int)
        max_seg = self.segment_slider.maximum()
        if 10 <= saved_seg <= max_seg:
            self.segment_slider.setValue(saved_seg)
        elif saved_seg > max_seg:
            self.segment_slider.setValue(max_seg)

        saved_ts = self._settings.value("timestamps", False, type=bool)
        self.timestamps_checkbox.setChecked(saved_ts)

        saved_dur = self._settings.value("seg_duration", DEFAULT_SEGMENT_DURATION, type=int)
        if 5 <= saved_dur <= 60:
            self.seg_duration_slider.setValue(saved_dur)

        saved_ts_fmt = self._settings.value("ts_format", "txt")
        if saved_ts_fmt in TIMESTAMP_FORMATS:
            self.ts_format_combo.setCurrentText(saved_ts_fmt)

        saved_exts = self._settings.value("extensions")
        if saved_exts and isinstance(saved_exts, list):
            valid = [e for e in saved_exts if e in SUPPORTED_AUDIO_EXTENSIONS]
            if valid:
                self._selected_extensions = valid
