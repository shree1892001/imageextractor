import os
from typing import Optional, List, Dict, Any, Union, Callable
from pathlib import Path
import logging
from datetime import datetime
import shutil
import hashlib
import json
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class GenFiler:
    def __init__(self, base_directory: Union[str, Path], config: Optional[Dict[str, Any]] = None) -> None:
        self.base_directory = Path(base_directory)
        self.config = config or {}
        self._setup_logging()
        self.base_directory.mkdir(parents=True, exist_ok=True)
        self.categories = self.config.get('categories', {
            'documents': ['.pdf', '.doc', '.docx', '.txt'],
            'images': ['.jpg', '.jpeg', '.png', '.gif'],
            'audio': ['.mp3', '.wav', '.flac'],
            'video': ['.mp4', '.avi', '.mkv'],
            'archives': ['.zip', '.rar', '.7z']
        })

    def _setup_logging(self) -> None:
        self._logger = logging.getLogger('GenFiler')
        handler = logging.FileHandler(self.base_directory / 'genfiler.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def organize_files(self, source_dir: Optional[Union[str, Path]] = None) -> Dict[str, List[Path]]:
        source = Path(source_dir) if source_dir else self.base_directory
        organized: Dict[str, List[Path]] = {}

        try:
            for file_path in source.rglob('*'):
                if file_path.is_file():
                    category = self.categorize_file(file_path)
                    target_dir = self.base_directory / category
                    target_dir.mkdir(exist_ok=True)
                    
                    new_path = target_dir / file_path.name
                    if new_path.exists():
                        new_path = self._get_unique_path(new_path)
                    
                    shutil.move(str(file_path), str(new_path))
                    organized.setdefault(category, []).append(new_path)
                    self.logger.info(f'Moved {file_path} to {new_path}')

            return organized
        except Exception as e:
            self.logger.error(f'Error organizing files: {str(e)}')
            raise

    def categorize_file(self, file_path: Union[str, Path]) -> str:
        path = Path(file_path)
        extension = path.suffix.lower()

        # Check predefined categories
        for category, extensions in self.categories.items():
            if extension in extensions:
                return category

        # Analyze content for better categorization
        content_info = self.analyze_content(path)
        if content_info.get('suggested_category'):
            return content_info['suggested_category']

        return 'uncategorized'

    def analyze_content(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(file_path)
        result = {
            'file_type': path.suffix.lower(),
            'size': path.stat().st_size,
            'suggested_category': None
        }

        try:
            # Basic text content analysis
            if path.suffix.lower() in ['.txt', '.log', '.md']:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read(1024)  # Read first 1KB for analysis
                    if any(keyword in content.lower() for keyword in ['error', 'exception', 'failed']):
                        result['suggested_category'] = 'logs'
                    elif any(keyword in content.lower() for keyword in ['import', 'class', 'def', 'function']):
                        result['suggested_category'] = 'code'

            # Add file signature/magic number analysis
            with open(path, 'rb') as f:
                header = f.read(8)  # Read first 8 bytes for file signature
                result['file_signature'] = header.hex()

        except Exception as e:
            self.logger.error(f'Error analyzing content of {path}: {str(e)}')

        return result

    def pattern_recognition(self, directory: Union[str, Path]) -> Dict[str, Any]:
        dir_path = Path(directory)
        patterns = {
            'naming_patterns': {},
            'extension_stats': {},
            'size_distribution': {},
            'timestamp_patterns': {}
        }

        try:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    # Analyze naming patterns
                    name_parts = file_path.stem.split('_')
                    for part in name_parts:
                        patterns['naming_patterns'][part] = patterns['naming_patterns'].get(part, 0) + 1

                    # Extension statistics
                    ext = file_path.suffix.lower()
                    patterns['extension_stats'][ext] = patterns['extension_stats'].get(ext, 0) + 1

                    # Size distribution
                    size = file_path.stat().st_size
                    size_category = self._get_size_category(size)
                    patterns['size_distribution'][size_category] = patterns['size_distribution'].get(size_category, 0) + 1

                    # Timestamp patterns
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    hour = mtime.hour
                    patterns['timestamp_patterns'][hour] = patterns['timestamp_patterns'].get(hour, 0) + 1

            return patterns
        except Exception as e:
            self.logger.error(f'Error in pattern recognition: {str(e)}')
            raise

    def auto_rename(self, file_path: Union[str, Path], pattern: Optional[str] = None) -> Path:
        path = Path(file_path)
        if not pattern:
            # Generate pattern based on file attributes
            timestamp = datetime.fromtimestamp(path.stat().st_mtime)
            pattern = f"{timestamp.strftime('%Y%m%d')}_{path.stem}_{self._generate_short_hash(path)}"

        new_name = f"{pattern}{path.suffix}"
        new_path = path.parent / new_name

        try:
            if new_path.exists():
                new_path = self._get_unique_path(new_path)
            path.rename(new_path)
            self.logger.info(f'Renamed {path} to {new_path}')
            return new_path
        except Exception as e:
            self.logger.error(f'Error renaming {path}: {str(e)}')
            raise

    def backup_file(self, file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
        source = Path(file_path)
        backup_directory = Path(backup_dir) if backup_dir else self.base_directory / 'backups'
        backup_directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_directory / f"{source.stem}_{timestamp}{source.suffix}"

        try:
            shutil.copy2(str(source), str(backup_path))
            self.logger.info(f'Created backup of {source} at {backup_path}')
            return backup_path
        except Exception as e:
            self.logger.error(f'Error creating backup of {source}: {str(e)}')
            raise

    def restore_file(self, backup_path: Union[str, Path], restore_location: Optional[Union[str, Path]] = None) -> Path:
        backup = Path(backup_path)
        if not backup.exists():
            raise FileNotFoundError(f'Backup file {backup} not found')

        restore_path = Path(restore_location) if restore_location else self.base_directory / backup.name

        try:
            if restore_path.exists():
                restore_path = self._get_unique_path(restore_path)
            shutil.copy2(str(backup), str(restore_path))
            self.logger.info(f'Restored {backup} to {restore_path}')
            return restore_path
        except Exception as e:
            self.logger.error(f'Error restoring {backup}: {str(e)}')
            raise

    def monitor_directory(self, directory: Union[str, Path], callback: Optional[Callable[[Path], None]] = None) -> None:
        class FileHandler(FileSystemEventHandler):
            def __init__(self, genfiler: 'GenFiler', callback: Optional[Callable[[Path], None]]):
                self.genfiler = genfiler
                self.callback = callback

            def on_created(self, event):
                if not event.is_directory:
                    path = Path(event.src_path)
                    self.genfiler.logger.info(f'New file detected: {path}')
                    if self.callback:
                        self.callback(path)

        try:
            path = Path(directory)
            event_handler = FileHandler(self, callback)
            observer = Observer()
            observer.schedule(event_handler, str(path), recursive=True)
            observer.start()
            self.logger.info(f'Started monitoring {path}')
            return observer
        except Exception as e:
            self.logger.error(f'Error setting up directory monitoring: {str(e)}')
            raise

    def get_file_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(file_path)
        try:
            stats = path.stat()
            return {
                'name': path.name,
                'extension': path.suffix,
                'size': stats.st_size,
                'created_time': datetime.fromtimestamp(stats.st_ctime),
                'modified_time': datetime.fromtimestamp(stats.st_mtime),
                'accessed_time': datetime.fromtimestamp(stats.st_atime),
                'is_hidden': path.name.startswith('.'),
                'checksum': self._calculate_checksum(path),
                'mime_type': self._get_mime_type(path)
            }
        except Exception as e:
            self.logger.error(f'Error getting metadata for {path}: {str(e)}')
            raise

    def search_files(self, pattern: str, recursive: bool = True) -> List[Path]:
        try:
            results = []
            search_path = self.base_directory

            if recursive:
                glob_pattern = f'**/{pattern}'
            else:
                glob_pattern = pattern

            for file_path in search_path.glob(glob_pattern):
                if file_path.is_file():
                    results.append(file_path)

            self.logger.info(f'Found {len(results)} files matching pattern {pattern}')
            return results
        except Exception as e:
            self.logger.error(f'Error searching files with pattern {pattern}: {str(e)}')
            raise

    def validate_file(self, file_path: Union[str, Path], validation_rules: Optional[Dict[str, Any]] = None) -> bool:
        path = Path(file_path)
        if not validation_rules:
            validation_rules = self.config.get('validation_rules', {
                'max_size': 100 * 1024 * 1024,  # 100MB
                'allowed_extensions': ['.txt', '.pdf', '.doc', '.docx'],
                'min_size': 0
            })

        try:
            if not path.exists():
                self.logger.warning(f'File {path} does not exist')
                return False

            file_size = path.stat().st_size
            extension = path.suffix.lower()

            # Check file size
            if file_size > validation_rules.get('max_size', float('inf')):
                self.logger.warning(f'File {path} exceeds maximum size')
                return False

            if file_size < validation_rules.get('min_size', 0):
                self.logger.warning(f'File {path} is smaller than minimum size')
                return False

            # Check file extension
            if 'allowed_extensions' in validation_rules and extension not in validation_rules['allowed_extensions']:
                self.logger.warning(f'File {path} has invalid extension')
                return False

            return True
        except Exception as e:
            self.logger.error(f'Error validating file {path}: {str(e)}')
            return False

    def process_batch(self, file_list: List[Union[str, Path]], operation: Callable[[Path], Any]) -> Dict[Path, Any]:
        results = {}
        for file_path in file_list:
            path = Path(file_path)
            try:
                results[path] = operation(path)
                self.logger.info(f'Successfully processed {path}')
            except Exception as e:
                self.logger.error(f'Error processing {path}: {str(e)}')
                results[path] = {'error': str(e)}
        return results

    def create_file_structure(self, structure: Dict[str, Any]) -> None:
        def create_structure(base_path: Path, struct: Dict[str, Any]) -> None:
            for name, content in struct.items():
                path = base_path / name
                if isinstance(content, dict):
                    path.mkdir(parents=True, exist_ok=True)
                    create_structure(path, content)
                else:
                    path.write_text(str(content))

        try:
            create_structure(self.base_directory, structure)
            self.logger.info('File structure created successfully')
        except Exception as e:
            self.logger.error(f'Error creating file structure: {str(e)}')
            raise

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_files': 0,
            'total_size': 0,
            'extension_count': {},
            'category_count': {},
            'average_file_size': 0
        }

        try:
            for file_path in self.base_directory.rglob('*'):
                if file_path.is_file():
                    stats['total_files'] += 1
                    size = file_path.stat().st_size
                    stats['total_size'] += size

                    ext = file_path.suffix.lower()
                    stats['extension_count'][ext] = stats['extension_count'].get(ext, 0) + 1