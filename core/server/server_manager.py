from __future__ import annotations

import threading
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

from config.server_settings import TranscriptionSettings
from core.logging_config import get_logger

logger = get_logger(__name__)

STARTUP_POLL_INTERVAL_MS = 50
STARTUP_TIMEOUT_MS = 5000


class ServerManager(QObject):
    server_started = Signal(int)
    server_stopped = Signal()
    server_error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._port: int = 0
        self._startup_timer: Optional[QTimer] = None
        self._startup_elapsed_ms: int = 0

    def start_server(self, port: int, model_manager, default_settings: TranscriptionSettings) -> None:
        if self.is_running():
            self.server_error.emit("Server is already running")
            return

        try:
            import uvicorn
            from core.server.api_server import create_app, set_app_state

            set_app_state(
                model_manager=model_manager,
                default_settings=default_settings,
            )

            app = create_app()
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=port,
                log_level="warning",
                log_config=None,
                access_log=False,
            )
            self._server = uvicorn.Server(config)
            self._port = port

            self._thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="uvicorn-server",
            )
            self._thread.start()

        except Exception as e:
            logger.error(f"Failed to start server: {e}", exc_info=True)
            self.server_error.emit(str(e))
            return

        # Don't report success until uvicorn has actually bound the port. On
        # bind failure (e.g. port in use) uvicorn calls sys.exit(1), raising
        # SystemExit in the worker thread, which is swallowed by threading's
        # default excepthook. Poll for the real outcome instead of assuming
        # the server started.
        logger.info(f"Server starting on port {port}")
        self._startup_elapsed_ms = 0
        self._startup_timer = QTimer(self)
        self._startup_timer.setInterval(STARTUP_POLL_INTERVAL_MS)
        self._startup_timer.timeout.connect(self._check_startup)
        self._startup_timer.start()

    def _check_startup(self) -> None:
        if self._server is not None and self._server.started:
            self._stop_startup_timer()
            logger.info(f"Server started on port {self._port}")
            self.server_started.emit(self._port)
        elif self._thread is None or not self._thread.is_alive():
            self._stop_startup_timer()
            self._server = None
            self._thread = None
            logger.error(f"Server failed to start on port {self._port}")
            self.server_error.emit(
                f"Server failed to start on port {self._port}. "
                "The port may already be in use."
            )
        else:
            self._startup_elapsed_ms += STARTUP_POLL_INTERVAL_MS
            if self._startup_elapsed_ms >= STARTUP_TIMEOUT_MS:
                self._stop_startup_timer()
                logger.error(
                    f"Server did not start within {STARTUP_TIMEOUT_MS} ms "
                    f"on port {self._port}"
                )
                self._shutdown_thread()
                self.server_error.emit(
                    f"Server did not start within {STARTUP_TIMEOUT_MS // 1000} "
                    f"seconds on port {self._port}."
                )

    def _stop_startup_timer(self) -> None:
        if self._startup_timer is not None:
            self._startup_timer.stop()
            self._startup_timer = None

    def _shutdown_thread(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Server thread did not stop within timeout")
            self._thread = None
        self._server = None

    def _run_server(self) -> None:
        try:
            self._server.run()
        except SystemExit as e:
            # uvicorn calls sys.exit(1) on bind failure; SystemExit is a
            # BaseException and would otherwise be swallowed by threading's
            # default excepthook with no log. _check_startup detects the
            # resulting thread death and emits server_error.
            logger.error(
                f"Server failed to bind (port {self._port} may be in use): {e}"
            )
        except Exception as e:
            logger.error(f"Server thread error: {e}", exc_info=True)

    def stop_server(self) -> None:
        # Cancel any in-flight startup poll so a user-initiated stop during
        # startup doesn't fire a spurious "failed to start" error.
        self._stop_startup_timer()

        was_running = self.is_running()
        self._shutdown_thread()

        if was_running:
            logger.info("Server stopped")
            self.server_stopped.emit()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_transcription_active(self) -> bool:
        from core.server.api_server import _state
        return _state.transcription_active

    def cleanup(self) -> None:
        if self.is_running():
            from core.server.api_server import _state
            _state.cancel_event.set()
            self.stop_server()

    @property
    def port(self) -> int:
        return self._port
