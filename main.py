import sys
import warnings
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

warnings.filterwarnings("ignore", category=UserWarning, message=r".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=r".*Megatron.*")
warnings.filterwarnings("ignore", message=r".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=r".*TF32.*")
warnings.filterwarnings("ignore", message=r".*allow_tf32.*")
warnings.filterwarnings("ignore", message=r".*Redirects are currently not supported.*")

for _name in ["nemo", "nemo.collections", "nemo.utils", "nemo.core",
              "lightning", "lightning_fabric", "pytorch_lightning",
              "nemo_logger", "nv_one_logger", "wandb", "numba"]:
    logging.getLogger(_name).setLevel(logging.ERROR)

from utils.cuda_setup import setup_cuda_paths
setup_cuda_paths()

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
