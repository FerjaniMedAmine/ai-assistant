"""Per-user text workspace for agent file tools."""

import os
import tempfile
from pathlib import Path

from backend.config import BASE_DIR

MAX_WRITE_CHARS = 262_144
MAX_READ_CHARS = 12_000
MAX_FILE_BYTES = 2_000_000


class UserWorkspace:
    def __init__(self, user_id: str, root: Path | None = None):
        if not user_id or user_id in (".", "..") or "/" in user_id or "\\" in user_id:
            raise ValueError("Invalid user ID")
        self.root = (root or BASE_DIR / "data" / "workspaces" / user_id).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, relative_path: str) -> Path:
        if not relative_path or Path(relative_path).is_absolute():
            raise ValueError("Use a path relative to your workspace")
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Path is outside your workspace")
        return path

    def list_directory(self, path: str = ".") -> str:
        directory = self._path(path)
        if not directory.is_dir():
            raise ValueError("Directory does not exist")
        entries = sorted(directory.iterdir(), key=lambda item: item.name.lower())
        rendered = [f"{'folder' if entry.is_dir() else 'file'}  {entry.name}" for entry in entries[:200]]
        if len(entries) > 200:
            rendered.append(f"... {len(entries) - 200} more entries")
        return "\n".join(rendered) or "Directory is empty"

    def read_file(self, path: str, start_char: int = 0, max_chars: int = MAX_READ_CHARS) -> str:
        file_path = self._path(path)
        if not file_path.is_file():
            raise ValueError("File does not exist")
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("File is too large to read with this tool")
        if start_char < 0 or not 1 <= max_chars <= MAX_READ_CHARS:
            raise ValueError("Invalid read range")
        with file_path.open("r", encoding="utf-8") as file:
            file.read(start_char)
            text = file.read(max_chars)
            more = bool(file.read(1))
        suffix = f"\n[More content available; continue at character {start_char + len(text)}]" if more else ""
        return text + suffix

    def write_file(self, path: str, content: str, overwrite: bool = False) -> str:
        file_path = self._path(path)
        if file_path == self.root or file_path.is_dir():
            raise ValueError("Choose a file path")
        if len(content) > MAX_WRITE_CHARS:
            raise ValueError(f"File content exceeds {MAX_WRITE_CHARS} characters")
        if file_path.exists() and not overwrite:
            raise ValueError("File already exists; set overwrite=true to replace it")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # The temporary file is in the same directory so replacement is atomic.
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=file_path.parent, delete=False) as temp:
                temp_name = temp.name
                temp.write(content)
            os.replace(temp_name, file_path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
        return f"Wrote {len(content)} characters to {path}"

    def create_directory(self, path: str) -> str:
        directory = self._path(path)
        directory.mkdir(parents=True, exist_ok=True)
        return f"Created folder {path}"

    def move_path(self, source: str, destination: str) -> str:
        source_path = self._path(source)
        destination_path = self._path(destination)
        if source_path == self.root or not source_path.exists():
            raise ValueError("Source does not exist")
        if destination_path.exists():
            raise ValueError("Destination already exists")
        if source_path.is_dir() and destination_path.is_relative_to(source_path):
            raise ValueError("Cannot move a folder into itself")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(destination_path)
        return f"Moved {source} to {destination}"

    def delete_path(self, path: str) -> str:
        target = self._path(path)
        if target == self.root or not target.exists():
            raise ValueError("Path does not exist")
        if target.is_dir():
            target.rmdir()  # Deliberately refuse recursive deletion.
        else:
            target.unlink()
        return f"Deleted {path}"
