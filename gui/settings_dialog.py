from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal, Qt, QUrl
from PySide6.QtGui import QStandardItemModel, QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QFormLayout,
    QWidget,
    QGroupBox,
    QCheckBox,
    QSpinBox,
    QLabel,
)

from PySide6.QtWidgets import QMessageBox, QSpinBox as _QSpinBox

from core.models.metadata import ModelMetadata
from core.audio.device_utils import get_input_devices
from gui.styles import update_button_property
from gui.file_panel import FileTypesDialog, SUPPORTED_AUDIO_EXTENSIONS, ToggleSwitch
from core.logging_config import get_logger

logger = get_logger(__name__)


_CHUNK_TIP_PARAKEET = (
    "<qt>Fixed at 90 seconds - optimal setting<br>"
    "after extensive benchmarking.</qt>"
)
_CHUNK_TIP_CANARY = (
    "<qt>Fixed at 40 seconds for Canary.<br>"
    "Canary's attention context is capped<br>"
    "lower than Parakeet's.</qt>"
)
_SEG_DUR_TIP_DEFAULT = (
    "<qt>When timestamps are enabled, word-level<br>"
    "timings are grouped into segments with a<br>"
    "maximum duration of this many seconds.<br><br>"
    "Shorter values produce more, tighter<br>"
    "segments (useful for subtitles); longer<br>"
    "values produce fewer, longer lines.</qt>"
)
_TIMESTAMPS_TIP_DEFAULT = (
    "<qt>Include word-level segment timestamps<br>"
    "in output (auto-enabled for SRT/VTT).</qt>"
)
_CANARY_NO_TIMESTAMPS_TIP = (
    "<qt>Canary does not support timestamps.<br>"
    "Switch to a Parakeet model to enable<br>"
    "timestamped output.</qt>"
)
_CANARY_CPU_DISABLED_TIP = (
    "<qt>Canary is heavy (~11 GB VRAM,<br>"
    "~30x slower than Parakeet) and is<br>"
    "disabled on CPU. Switch to CUDA to<br>"
    "use this model.</qt>"
)


class SettingsDialog(QDialog):
    model_update_requested = Signal(str, str, str)
    audio_device_changed = Signal(str, str)
    parakeet_settings_changed = Signal(object)
    file_types_changed = Signal(object)
    server_mode_changed = Signal(bool, int)

    def __init__(
        self,
        parent: QWidget | None,
        cuda_available: bool,
        supported_precisions: dict[str, list[str]],
        current_settings: dict[str, str],
        current_audio_device: dict[str, str] | None = None,
        current_parakeet_settings: dict | None = None,
        current_ext_checked: dict[str, bool] | None = None,
        current_server_settings: dict | None = None,
        is_busy_check=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.resize(600, self.sizeHint().height())

        self.cuda_available = cuda_available
        self.supported_precisions = supported_precisions
        self.current_settings = dict(current_settings)
        self.current_audio_device = current_audio_device or {"name": "", "hostapi": ""}
        self.current_parakeet_settings = current_parakeet_settings or {
            "include_timestamps": False,
            "segment_duration": 10,
        }
        self.current_server_settings = current_server_settings or {
            "server_mode_enabled": False,
            "server_port": 8765,
        }
        self._is_busy_check = is_busy_check or (lambda: False)
        self._last_model_type = ModelMetadata.get_model_type(
            self.current_settings.get("model_name", "Parakeet TDT 0.6B v2")
        )
        self._forced_precision: str | None = None
        self._ext_checked = current_ext_checked or {
            ext: True for ext in SUPPORTED_AUDIO_EXTENSIONS
        }

        self._input_devices = get_input_devices()

        self._build_ui()
        self._setup_connections()
        self._populate_from_settings()
        self._check_for_changes()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(16, 16, 16, 16)
        outer_layout.setSpacing(12)

        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(16)

        left_column = QVBoxLayout()
        left_column.setSpacing(12)

        model_group = QGroupBox("Model")
        model_form = QFormLayout(model_group)
        model_form.setHorizontalSpacing(12)
        model_form.setVerticalSpacing(10)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(ModelMetadata.get_all_model_names())
        model_form.addRow("Model", self.model_dropdown)

        self.device_dropdown = QComboBox()
        devices = ["cpu", "cuda"] if self.cuda_available else ["cpu"]
        self.device_dropdown.addItems(devices)
        model_form.addRow("Device", self.device_dropdown)

        self.precision_dropdown = QComboBox()
        model_form.addRow("Precision", self.precision_dropdown)

        self.model_desc_label = QLabel("")
        self.model_desc_label.setWordWrap(True)
        self.model_desc_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        model_form.addRow("", self.model_desc_label)

        left_column.addWidget(model_group)

        audio_group = QGroupBox("Audio Input")
        audio_form = QFormLayout(audio_group)
        audio_form.setHorizontalSpacing(12)
        audio_form.setVerticalSpacing(10)

        self.audio_device_dropdown = QComboBox()
        self.audio_device_dropdown.addItem("System Default", None)
        for dev in self._input_devices:
            display = f"{dev['name']} ({dev['hostapi']})"
            self.audio_device_dropdown.addItem(display, dev)
        audio_form.addRow("Input Device", self.audio_device_dropdown)

        left_column.addWidget(audio_group)
        left_column.addStretch(1)

        columns_layout.addLayout(left_column, 1)

        right_column = QVBoxLayout()
        right_column.setSpacing(12)

        parakeet_group = QGroupBox("Transcription Settings")
        parakeet_form = QFormLayout(parakeet_group)
        parakeet_form.setHorizontalSpacing(12)
        parakeet_form.setVerticalSpacing(10)

        self.include_timestamps_cb = QCheckBox()
        self._timestamps_caption = QLabel("Include Timestamps")
        parakeet_form.addRow(self._timestamps_caption, self.include_timestamps_cb)

        self.chunk_length_label = QLabel("90 seconds")
        self._chunk_caption = QLabel("Chunk Length")
        parakeet_form.addRow(self._chunk_caption, self.chunk_length_label)

        self.segment_duration_spin = QSpinBox()
        self.segment_duration_spin.setRange(1, 90)
        self.segment_duration_spin.setSuffix(" s")
        self._seg_dur_caption = QLabel("Segment Duration")
        parakeet_form.addRow(self._seg_dur_caption, self.segment_duration_spin)

        right_column.addWidget(parakeet_group)

        server_group = QGroupBox("Server Mode")
        server_vbox = QVBoxLayout(server_group)
        server_vbox.setContentsMargins(12, 10, 12, 10)
        server_vbox.setSpacing(8)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(6)
        self._server_off_label = QLabel("Off")
        self._server_off_label.setStyleSheet("font-size: 11px;")
        toggle_row.addWidget(self._server_off_label)
        self.server_mode_toggle = ToggleSwitch()
        toggle_row.addWidget(self.server_mode_toggle)
        self._server_on_label = QLabel("On")
        self._server_on_label.setStyleSheet("font-size: 11px;")
        toggle_row.addWidget(self._server_on_label)
        toggle_row.addSpacing(16)
        toggle_row.addWidget(QLabel("Port:"))
        self.server_port_spin = _QSpinBox()
        self.server_port_spin.setRange(1024, 65535)
        self.server_port_spin.setToolTip(
            "<qt>TCP port the HTTP API will bind to.<br>"
            "Pick one that isn't in use on your<br>"
            "machine (default 8765).</qt>"
        )
        toggle_row.addWidget(self.server_port_spin)
        toggle_row.addStretch(1)
        server_vbox.addLayout(toggle_row)

        server_hint = QLabel(
            "<qt>When On, the voice recorder, clipboard,<br>"
            "file transcription, and other settings are<br>"
            "controlled by HTTP clients.</qt>"
        )
        server_hint.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        server_vbox.addWidget(server_hint)

        right_column.addWidget(server_group)

        file_types_row = QHBoxLayout()
        file_types_row.addStretch(1)
        self._file_types_btn = QPushButton("File Types...")
        self._file_types_btn.setFixedHeight(28)
        self._file_types_btn.setFixedWidth(100)
        self._file_types_btn.setToolTip("Configure which audio/video file types to include in batch processing")
        self._file_types_btn.clicked.connect(self._open_file_types_dialog)
        file_types_row.addWidget(self._file_types_btn)

        self._guide_btn = QPushButton("Guide")
        self._guide_btn.setFixedHeight(28)
        self._guide_btn.setFixedWidth(100)
        self._guide_btn.setToolTip(
            "<qt>Open the HTML user guide for the<br>"
            "Server API in your default browser.</qt>"
        )
        self._guide_btn.clicked.connect(self._open_server_guide)
        file_types_row.addWidget(self._guide_btn)
        right_column.addLayout(file_types_row)

        right_column.addStretch(1)

        columns_layout.addLayout(right_column, 1)

        outer_layout.addLayout(columns_layout)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        self.update_btn = QPushButton("Update Settings")
        self.update_btn.setObjectName("updateButton")
        self.update_btn.setEnabled(False)
        self.update_btn.clicked.connect(self._on_update_clicked)
        button_row.addWidget(self.update_btn)

        close_btn = QPushButton("Close")
        close_btn.setObjectName("closeButton")
        close_btn.clicked.connect(self.reject)
        button_row.addWidget(close_btn)

        self.update_btn.setFixedHeight(35)
        close_btn.setFixedHeight(35)
        outer_layout.addLayout(button_row)

    def _setup_connections(self) -> None:
        self.model_dropdown.currentTextChanged.connect(self._on_model_changed)
        self.device_dropdown.currentTextChanged.connect(self._on_device_changed)
        self.precision_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.audio_device_dropdown.currentIndexChanged.connect(self._check_for_changes)
        self.include_timestamps_cb.toggled.connect(self._check_for_changes)
        self.include_timestamps_cb.toggled.connect(self._update_segment_duration_enabled)
        self.segment_duration_spin.valueChanged.connect(self._check_for_changes)
        self.server_mode_toggle.toggled.connect(self._check_for_changes)
        self.server_mode_toggle.toggled.connect(self._apply_server_mode_lock)
        self.server_port_spin.valueChanged.connect(self._check_for_changes)

    def _populate_from_settings(self) -> None:
        self.model_dropdown.setCurrentText(self.current_settings.get("model_name", "Parakeet TDT 0.6B v2"))
        self.device_dropdown.setCurrentText(self.current_settings.get("device_type", "cuda"))
        self._apply_device_constraints()
        self._update_precision_options()
        self.precision_dropdown.setCurrentText(
            self.current_settings.get("precision", "bfloat16")
        )
        self._update_model_description()
        self._select_audio_device()

        self.include_timestamps_cb.setChecked(
            self.current_parakeet_settings.get("include_timestamps", False)
        )
        current_seg_dur = int(self.current_parakeet_settings.get("segment_duration", 10))
        self.segment_duration_spin.setValue(current_seg_dur)
        self._update_model_dependent_widgets()

        self.server_mode_toggle.blockSignals(True)
        self.server_mode_toggle.setChecked(
            bool(self.current_server_settings.get("server_mode_enabled", False))
        )
        self.server_mode_toggle.blockSignals(False)
        self.server_port_spin.blockSignals(True)
        self.server_port_spin.setValue(
            int(self.current_server_settings.get("server_port", 8765))
        )
        self.server_port_spin.blockSignals(False)
        self._apply_server_mode_lock()

    def _select_audio_device(self) -> None:
        saved_name = self.current_audio_device.get("name", "")
        saved_hostapi = self.current_audio_device.get("hostapi", "")

        if not saved_name:
            self.audio_device_dropdown.setCurrentIndex(0)
            return

        for i in range(1, self.audio_device_dropdown.count()):
            data = self.audio_device_dropdown.itemData(i)
            if data and data["name"] == saved_name:
                if saved_hostapi and data["hostapi"] == saved_hostapi:
                    self.audio_device_dropdown.setCurrentIndex(i)
                    return

        for i in range(1, self.audio_device_dropdown.count()):
            data = self.audio_device_dropdown.itemData(i)
            if data and data["name"] == saved_name:
                self.audio_device_dropdown.setCurrentIndex(i)
                return

        self.audio_device_dropdown.setCurrentIndex(0)

    def _on_model_changed(self, *args) -> None:
        new_type = ModelMetadata.get_model_type(self.model_dropdown.currentText())
        if new_type == "canary" and self._last_model_type != "canary":
            self._forced_precision = "float16"
        self._last_model_type = new_type
        self._update_precision_options()
        self._update_model_description()
        self._update_model_dependent_widgets()
        self._check_for_changes()

    def _on_device_changed(self, *args) -> None:
        self._apply_device_constraints()
        self._update_precision_options()
        self._check_for_changes()

    def _apply_device_constraints(self) -> None:
        device = self.device_dropdown.currentText()
        model = self.model_dropdown.model()
        if not isinstance(model, QStandardItemModel):
            return

        canary_disabled_on_cpu = (device == "cpu")
        for i in range(self.model_dropdown.count()):
            item = model.item(i)
            if item is None:
                continue
            name = item.text()
            is_canary = ModelMetadata.get_model_type(name) == "canary"
            if is_canary and canary_disabled_on_cpu:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setToolTip(_CANARY_CPU_DISABLED_TIP)
            else:
                item.setFlags(item.flags() | Qt.ItemIsEnabled)
                item.setToolTip("")

        if canary_disabled_on_cpu:
            current_name = self.model_dropdown.currentText()
            if ModelMetadata.get_model_type(current_name) == "canary":
                for i in range(self.model_dropdown.count()):
                    item = model.item(i)
                    if item and (item.flags() & Qt.ItemIsEnabled):
                        self.model_dropdown.blockSignals(True)
                        self.model_dropdown.setCurrentIndex(i)
                        self.model_dropdown.blockSignals(False)
                        self._update_model_description()
                        self._update_model_dependent_widgets()
                        break

    def _update_precision_options(self) -> None:
        model = self.model_dropdown.currentText()
        device = self.device_dropdown.currentText()
        opts = ModelMetadata.get_precision_options(model, device, self.supported_precisions)

        self.precision_dropdown.blockSignals(True)
        current = self.precision_dropdown.currentText()
        self.precision_dropdown.clear()
        self.precision_dropdown.addItems(opts)
        if self._forced_precision and self._forced_precision in opts:
            self.precision_dropdown.setCurrentText(self._forced_precision)
            self._forced_precision = None
        elif current in opts:
            self.precision_dropdown.setCurrentText(current)
        elif opts:
            self.precision_dropdown.setCurrentText(opts[0])
        self.precision_dropdown.blockSignals(False)

    def _update_model_description(self) -> None:
        name = self.model_dropdown.currentText()
        self.model_desc_label.setText(ModelMetadata.get_description(name))

    def _update_model_dependent_widgets(self) -> None:
        name = self.model_dropdown.currentText()
        is_canary = ModelMetadata.get_model_type(name) == "canary"
        chunk_len = ModelMetadata.get_chunk_length(name)

        self.chunk_length_label.setText(f"{chunk_len} seconds")
        chunk_tip = _CHUNK_TIP_CANARY if is_canary else _CHUNK_TIP_PARAKEET
        self.chunk_length_label.setToolTip(chunk_tip)
        self._chunk_caption.setToolTip(chunk_tip)

        if is_canary:
            self.include_timestamps_cb.blockSignals(True)
            self.include_timestamps_cb.setChecked(False)
            self.include_timestamps_cb.blockSignals(False)
            self.include_timestamps_cb.setEnabled(False)
            self.include_timestamps_cb.setToolTip(_CANARY_NO_TIMESTAMPS_TIP)
            self._timestamps_caption.setToolTip(_CANARY_NO_TIMESTAMPS_TIP)
            self.segment_duration_spin.setEnabled(False)
            self.segment_duration_spin.setToolTip(_CANARY_NO_TIMESTAMPS_TIP)
            self._seg_dur_caption.setToolTip(_CANARY_NO_TIMESTAMPS_TIP)
        else:
            self.include_timestamps_cb.setEnabled(True)
            self.include_timestamps_cb.setToolTip(_TIMESTAMPS_TIP_DEFAULT)
            self._timestamps_caption.setToolTip(_TIMESTAMPS_TIP_DEFAULT)
            self.segment_duration_spin.setToolTip(_SEG_DUR_TIP_DEFAULT)
            self._seg_dur_caption.setToolTip(_SEG_DUR_TIP_DEFAULT)
            self._update_segment_duration_enabled()

    def _update_segment_duration_enabled(self) -> None:
        name = self.model_dropdown.currentText()
        if ModelMetadata.get_model_type(name) == "canary":
            self.segment_duration_spin.setEnabled(False)
            return
        self.segment_duration_spin.setEnabled(self.include_timestamps_cb.isChecked())

    def _model_settings_changed(self) -> bool:
        current = {
            "model_name": self.model_dropdown.currentText(),
            "precision": self.precision_dropdown.currentText(),
            "device_type": self.device_dropdown.currentText(),
        }
        return current != self.current_settings

    def _audio_device_selection_changed(self) -> bool:
        data = self.audio_device_dropdown.currentData()
        if data is None:
            return bool(self.current_audio_device.get("name", ""))
        return (
            data["name"] != self.current_audio_device.get("name", "")
            or data["hostapi"] != self.current_audio_device.get("hostapi", "")
        )

    def _parakeet_settings_selection_changed(self) -> bool:
        current = {
            "include_timestamps": self.include_timestamps_cb.isChecked(),
            "segment_duration": self.segment_duration_spin.value(),
        }
        return current != self.current_parakeet_settings

    def _server_settings_selection_changed(self) -> bool:
        current = {
            "server_mode_enabled": self.server_mode_toggle.isChecked(),
            "server_port": self.server_port_spin.value(),
        }
        return current != self.current_server_settings

    def _apply_server_mode_lock(self) -> None:
        server_on = bool(self.server_mode_toggle.isChecked())
        lock_tip = (
            "<qt>Locked while the program is in<br>"
            "Server Mode. Turn the Server Mode<br>"
            "toggle Off to change this.</qt>"
            if server_on else ""
        )
        locked = [
            self.model_dropdown, self.device_dropdown, self.precision_dropdown,
            self.audio_device_dropdown, self.include_timestamps_cb,
            self.segment_duration_spin, self._file_types_btn,
        ]
        for w in locked:
            w.setEnabled(not server_on)
            w.setToolTip(lock_tip)

    def _check_for_changes(self) -> None:
        model_changed = self._model_settings_changed()
        audio_changed = self._audio_device_selection_changed()
        parakeet_changed = self._parakeet_settings_selection_changed()
        server_changed = self._server_settings_selection_changed()
        has_changes = model_changed or audio_changed or parakeet_changed or server_changed
        self.update_btn.setEnabled(has_changes)
        if model_changed:
            self.update_btn.setText("Reload Model")
        else:
            self.update_btn.setText("Update Settings")
        update_button_property(self.update_btn, "changed", has_changes)

    def _open_file_types_dialog(self) -> None:
        dlg = FileTypesDialog(self, self._ext_checked)
        if dlg.exec() == QDialog.Accepted:
            self._ext_checked = dlg.get_checked()
            self.file_types_changed.emit(self._ext_checked)

    def _open_server_guide(self) -> None:
        guide_path = Path(__file__).parent.parent / "guides" / "SERVER_API_GUIDE.html"
        if not guide_path.is_file():
            logger.warning(f"Guide not found at {guide_path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(guide_path.resolve())))

    def _on_update_clicked(self) -> None:
        if self._server_settings_selection_changed():
            wants_server_on = self.server_mode_toggle.isChecked()
            currently_on = bool(self.current_server_settings.get("server_mode_enabled", False))
            if wants_server_on and not currently_on and self._is_busy_check():
                QMessageBox.warning(
                    self,
                    "Transcription in progress",
                    "A transcription is currently being processed.\n\n"
                    "Wait for it to finish before turning Server Mode on.",
                )
                self.server_mode_toggle.blockSignals(True)
                self.server_mode_toggle.setChecked(False)
                self.server_mode_toggle.blockSignals(False)
                self._check_for_changes()
                return

        if self._model_settings_changed():
            model = self.model_dropdown.currentText()
            precision = self.precision_dropdown.currentText()
            device = self.device_dropdown.currentText()
            self.model_update_requested.emit(model, precision, device)

        if self._audio_device_selection_changed():
            data = self.audio_device_dropdown.currentData()
            if data is None:
                self.audio_device_changed.emit("", "")
            else:
                self.audio_device_changed.emit(data["name"], data["hostapi"])

        if self._parakeet_settings_selection_changed():
            settings = {
                "include_timestamps": self.include_timestamps_cb.isChecked(),
                "segment_duration": self.segment_duration_spin.value(),
            }
            self.parakeet_settings_changed.emit(settings)

        if self._server_settings_selection_changed():
            self.server_mode_changed.emit(
                self.server_mode_toggle.isChecked(),
                self.server_port_spin.value(),
            )

        self.accept()
