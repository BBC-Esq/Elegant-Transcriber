from pathlib import Path
from PySide6.QtCore import Qt, Slot, QSettings, QUrl
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton,
    QHBoxLayout, QCheckBox, QFileDialog, QMessageBox,
    QDialog, QSpinBox, QDialogButtonBox,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from config.settings import TranscriptionSettings
from config.constants import ALL_MODELS
from core.models.manager import ModelManager
from core.transcription.file_scanner import FileScanner
from core.transcription.service import TranscriptionService
from core.server.server_manager import ServerManager
from gui.settings_widget import SettingsWidget
from gui.widgets.metrics_bar import MetricsBar


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.transcription_service = TranscriptionService()
        self.file_scanner = FileScanner()
        self.server_manager = ServerManager()
        self.selected_directory = None
        self._guide_window = None
        self._qsettings = QSettings("ElegantAudioTranscriber", "ElegantAudioTranscriber")

        self._init_ui()
        self._connect_signals()
        self._load_settings()

    def _init_ui(self):
        self.setWindowTitle("Elegant Audio Transcriber")
        self.setGeometry(100, 100, 680, 400)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)

        top_group = QGroupBox()
        top_layout = QVBoxLayout()
        self.dir_label = QLabel("No directory selected")
        top_layout.addWidget(self.dir_label)
        self.progress_label = QLabel("Status: Idle")
        top_layout.addWidget(self.progress_label)
        top_group.setLayout(top_layout)
        layout.addWidget(top_group)

        self.settings_widget = SettingsWidget()
        layout.addWidget(self.settings_widget)

        controls_layout = QHBoxLayout()

        self.select_dir_button = QPushButton("Select Directory")
        self.select_dir_button.clicked.connect(self._select_directory)
        controls_layout.addWidget(self.select_dir_button)

        self.recursive_checkbox = QCheckBox("Process Sub-Folders?")
        controls_layout.addWidget(self.recursive_checkbox)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._process_files)
        self.start_button.setEnabled(False)
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_processing)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        self.server_button = QPushButton("Server")
        self.server_button.clicked.connect(self._toggle_server)
        controls_layout.addWidget(self.server_button)

        self.guide_button = QPushButton("User Guide")
        self.guide_button.clicked.connect(self._open_user_guide)
        controls_layout.addWidget(self.guide_button)

        layout.addLayout(controls_layout)

        self.metrics_bar = MetricsBar()
        layout.addWidget(self.metrics_bar)

    def _connect_signals(self):
        self.transcription_service.error_occurred.connect(self._on_error)
        self.transcription_service.progress_updated.connect(self._update_progress)
        self.transcription_service.completed.connect(self._on_processing_completed)
        self.server_manager.server_started.connect(self._on_server_started)
        self.server_manager.server_stopped.connect(self._on_server_stopped)
        self.server_manager.server_error.connect(self._on_server_error)

    @Slot()
    def _select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.selected_directory = dir_path
            self.dir_label.setText(f"Directory: {dir_path}")
            self.start_button.setEnabled(True)

    @Slot()
    def _process_files(self):
        if not self.selected_directory:
            return

        settings = self._build_settings()

        warnings = settings.validate()
        if warnings:
            reply = QMessageBox.warning(
                self, "Warning",
                "\n".join(warnings) + "\n\nContinue anyway?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return

        chunk = settings.segment_length
        device = settings.device.lower()
        model_info = ALL_MODELS.get(settings.model_key, {})
        model_type = model_info.get('model_type', 'parakeet')
        chunk_warn = None
        if model_type == "canary":
            pass
        elif device == "cpu" and chunk > 30:
            chunk_warn = (
                f"Audio chunk size is set to {chunk}s. "
                "For CPU, 20-30 seconds is ideal. "
                "Higher values may drastically increase processing time and RAM usage."
            )
        elif device == "cuda" and (chunk < 70 or chunk > 90):
            chunk_warn = (
                f"Audio chunk size is set to {chunk}s. "
                "For GPU, 70-90 seconds is ideal. "
                "Values outside this range may increase VRAM usage or slow processing."
            )
        if chunk_warn:
            reply = QMessageBox.information(
                self, "Chunk Size",
                chunk_warn + "\n\nContinue anyway?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            if reply == QMessageBox.Cancel:
                return

        files = self.file_scanner.scan_directory(
            Path(self.selected_directory),
            settings.selected_extensions,
            settings.recursive,
        )

        reply = QMessageBox.question(
            self, "Confirm",
            f"{len(files)} files found. Proceed?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Ok,
        )

        if reply == QMessageBox.Ok:
            self._on_processing_started()
            self.transcription_service.process_files(
                files=files,
                settings=settings,
                model_manager=self.model_manager,
            )

    @Slot()
    def _stop_processing(self):
        self.transcription_service.stop()
        self._on_processing_stopped()

    def _build_settings(self) -> TranscriptionSettings:
        return TranscriptionSettings(
            model_key=self.settings_widget.get_model(),
            device=self.settings_widget.get_device(),
            segment_length=self.settings_widget.get_segment_length(),
            segment_duration=self.settings_widget.get_segment_duration(),
            output_format=self.settings_widget.get_output_format(),
            word_timestamps=self.settings_widget.get_timestamps(),
            recursive=self.recursive_checkbox.isChecked(),
            selected_extensions=self.settings_widget.get_selected_extensions(),
        )

    def _on_processing_started(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.settings_widget.setEnabled(False)
        self.server_button.setEnabled(False)

    def _reset_ui_after_processing(self):
        server_active = self.server_manager.is_running()
        self.start_button.setEnabled(not server_active)
        self.stop_button.setEnabled(False)
        self.settings_widget.setEnabled(not server_active)
        self.server_button.setEnabled(True)

    def _on_processing_stopped(self):
        self._reset_ui_after_processing()

    @Slot(str)
    def _on_processing_completed(self, message: str):
        self._reset_ui_after_processing()
        self.progress_label.setText(f"Status: Completed | {message}")

    @Slot(str)
    def _on_error(self, message: str):
        self._reset_ui_after_processing()
        self.progress_label.setText(f"ERROR: {message}")
        QMessageBox.critical(self, "Error", message)

    @Slot(int, int, str)
    def _update_progress(self, current: int, total: int, message: str):
        self.progress_label.setText(f"Status: {message}")

    @Slot()
    def _toggle_server(self):
        if self.server_manager.is_running():
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self):
        if (self.transcription_service._processor
                and self.transcription_service._processor.isRunning()):
            QMessageBox.warning(
                self, "Server",
                "Cannot start server while a transcription is in progress.",
            )
            return

        port = self._show_port_dialog()
        if port is None:
            return

        settings = self._build_settings()
        self.server_manager.start_server(
            port=port,
            model_manager=self.model_manager,
            default_settings=settings,
        )

    def _stop_server(self):
        if self.server_manager.is_transcription_active():
            reply = QMessageBox.warning(
                self, "Server",
                "A transcription is currently in progress via the API.\n\n"
                "Stopping the server will cancel it. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        self.server_manager.stop_server()

    def _show_port_dialog(self) -> int | None:
        saved_port = self._qsettings.value("server_port", 7862, type=int)

        dialog = QDialog(self)
        dialog.setWindowTitle("Start Server")
        dialog.setFixedSize(250, 100)
        layout = QVBoxLayout(dialog)

        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        port_spin = QSpinBox()
        port_spin.setRange(1024, 65535)
        port_spin.setValue(saved_port)
        port_layout.addWidget(port_spin)
        layout.addLayout(port_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.Accepted:
            port = port_spin.value()
            self._qsettings.setValue("server_port", port)
            return port
        return None

    @Slot(int)
    def _on_server_started(self, port: int):
        self.server_button.setText("Stop Server")
        self.settings_widget.setEnabled(False)
        self.select_dir_button.setEnabled(False)
        self.recursive_checkbox.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_label.setText(f"Status: Server running on port {port}")

    @Slot()
    def _on_server_stopped(self):
        self.server_button.setText("Server")
        self.settings_widget.setEnabled(True)
        self.select_dir_button.setEnabled(True)
        self.recursive_checkbox.setEnabled(True)
        self.start_button.setEnabled(bool(self.selected_directory))
        self.stop_button.setEnabled(False)
        self.progress_label.setText("Status: Idle")

    @Slot(str)
    def _on_server_error(self, message: str):
        self._on_server_stopped()
        QMessageBox.critical(self, "Server Error", message)

    def _load_settings(self):
        saved_recursive = self._qsettings.value("recursive", False, type=bool)
        self.recursive_checkbox.setChecked(saved_recursive)

    def _save_settings(self):
        self._qsettings.setValue("recursive", self.recursive_checkbox.isChecked())
        self.settings_widget.save_settings()

    def _open_user_guide(self):
        if self._guide_window is not None and self._guide_window.isVisible():
            self._guide_window.raise_()
            self._guide_window.activateWindow()
            return

        guide_path = Path(__file__).resolve().parent.parent / "guides" / "SERVER_API_GUIDE.html"
        if not guide_path.is_file():
            QMessageBox.warning(self, "User Guide", f"Guide file not found:\n{guide_path}")
            return

        self._guide_window = QWidget()
        self._guide_window.setWindowTitle("Elegant Audio Transcriber — User Guide")
        self._guide_window.resize(960, 750)
        self._guide_window.destroyed.connect(lambda: setattr(self, '_guide_window', None))

        layout = QVBoxLayout(self._guide_window)
        layout.setContentsMargins(0, 0, 0, 0)

        web_view = QWebEngineView()
        web_view.setUrl(QUrl.fromLocalFile(str(guide_path)))
        layout.addWidget(web_view)

        self._guide_window.setAttribute(Qt.WA_DeleteOnClose)
        self._guide_window.show()

    def closeEvent(self, event):
        self._save_settings()
        if self._guide_window is not None:
            self._guide_window.close()
        self.server_manager.cleanup()
        self.transcription_service.cleanup()
        self.model_manager.cleanup()
        self.metrics_bar.cleanup()
        super().closeEvent(event)
