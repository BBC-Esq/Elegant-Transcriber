from pathlib import Path
from typing import List


class FileScanner:

    def scan_directory(self, directory: Path, extensions: List[str],
                       recursive: bool = False) -> List[Path]:
        files = []
        for ext in extensions:
            pattern = f'*{ext}'
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))
        return sorted(files)
