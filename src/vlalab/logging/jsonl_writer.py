"""
VLA-Lab JSONL Writer

Efficient append-only writer for step records.
"""

import json
from pathlib import Path
from typing import Union, Dict, Any
import threading


class JsonlWriter:
    """Thread-safe JSONL file writer for step records."""
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize JSONL writer.
        
        Args:
            file_path: Path to the JSONL file
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = None
        self._line_count = 0
        
        # Open file in append mode
        self._file = open(self.file_path, "a", encoding="utf-8")
    
    def write(self, record: Union[Dict[str, Any], "StepRecord"]) -> int:
        """
        Write a record to the JSONL file.
        
        Args:
            record: Dictionary or StepRecord to write
            
        Returns:
            Line number of the written record
        """
        if hasattr(record, "to_dict"):
            record = record.to_dict()
        
        json_str = json.dumps(record, ensure_ascii=False)
        
        with self._lock:
            self._file.write(json_str + "\n")
            self._file.flush()
            self._line_count += 1
            return self._line_count - 1
    
    def close(self):
        """Close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None
    
    @property
    def line_count(self) -> int:
        """Get the number of lines written."""
        return self._line_count
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class JsonlReader:
    """Reader for JSONL files."""
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize JSONL reader.
        
        Args:
            file_path: Path to the JSONL file
        """
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.file_path}")
    
    def __iter__(self):
        """Iterate over records in the file."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    
    def read_all(self):
        """Read all records into a list."""
        return list(self)
    
    def count(self) -> int:
        """Count the number of records."""
        count = 0
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    
    def read_line(self, line_idx: int) -> Dict[str, Any]:
        """Read a specific line by index."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    return json.loads(line.strip())
        raise IndexError(f"Line {line_idx} not found")
