"""Import the eight NHEFS data/layout files from a user-supplied Colab ZIP.

Validate before moving any files into raw/. Format validation is not independent
authentication that the archive came from CDC; record that distinction explicitly.
"""
import hashlib
from pathlib import Path, PurePosixPath
import tempfile
from zipfile import ZipFile

from .workflow import load_original, save_json

MAX_TOTAL_BYTES = 512 * 1024 * 1024


def import_data_zip(archive_path, root):
    archive_path, root = Path(archive_path), Path(root)
    original = load_original()
    required = {config[key] for config in original['WAVES'].values() for key in ('data', 'layout')}
    root.mkdir(parents=True, exist_ok=True)
    destination = root/'raw'
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('raw/ is not empty; use a new work directory to preserve existing input files')
    manifest_path = root/'input_archive_manifest.json'
    if manifest_path.exists():
        raise ValueError('An archive manifest already exists; use a new work directory')
    archive_hash = hashlib.sha256()
    with archive_path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            archive_hash.update(chunk)
    with ZipFile(archive_path) as archive:
        members = {}
        total = 0
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in entry.filename:
                raise ValueError('Unsafe archive member path')
            if entry.is_dir() or path.name not in required:
                continue
            if path.name in members:
                raise ValueError('Duplicate input filename: '+path.name)
            if entry.file_size == 0:
                raise ValueError('Empty input file: '+path.name)
            total += entry.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError('Input archive exceeds the supported uncompressed size')
            members[path.name] = entry
        missing = required-set(members)
        if missing:
            raise ValueError('Archive missing required files: '+', '.join(sorted(missing)))
        with tempfile.TemporaryDirectory(prefix='nhefs-import-', dir=root) as tmp:
            stage = Path(tmp)/'raw'; stage.mkdir()
            files = []
            for name, entry in sorted(members.items()):
                target = stage/name
                digest = hashlib.sha256()
                size = 0
                with archive.open(entry) as source, target.open('wb') as output:
                    for chunk in iter(lambda: source.read(1024*1024), b''):
                        size += len(chunk)
                        if size > entry.file_size:
                            raise ValueError('Unexpected uncompressed input size')
                        digest.update(chunk); output.write(chunk)
                with target.open('rb') as source:
                    header = source.read(1024).lstrip().lower()
                if header.startswith((b'<!doctype', b'<html')):
                    raise ValueError('HTML response instead of NHEFS input: '+name)
                files.append(dict(name=name, bytes=size, sha256=digest.hexdigest(),
                                  expected_source_url=original['BASE']+name))
            waves = {}
            for wave, config in original['WAVES'].items():
                layout = original['parse_layout']((stage/config['layout']).read_text(encoding='utf-8-sig'))
                rows = original['read_wave'](stage/config['data'], layout, config['fields'], config['id'])
                waves[wave] = len(rows)
            # Only validated inputs enter raw/. No ZIP paths are used as output paths.
            if destination.exists():
                destination.rmdir()  # Empty directories only; never remove user data.
            stage.rename(destination)
    manifest = dict(source='user_supplied_archive', archive_sha256=archive_hash.hexdigest(),
                    format_validation='all four waves parsed with their supplied layouts',
                    provenance_note='CDC origin reported by supplier; not independently authenticated by a published checksum',
                    wave_rows=waves, files=files)
    save_json(manifest_path, manifest)
    return manifest
