"""Tiny file-format fixtures only; no real participant/model performance claims."""
import tempfile
from pathlib import Path
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from round3.data_archive import import_data_zip
from round3.workflow import load_original


def tiny_archive(path, omit=None, html=False, duplicate=False):
    original = load_original()
    with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
        for wave, config in original['WAVES'].items():
            position = 1; specs = []; values = []
            for field in config['fields']:
                width = 5 if field == config['id'] else 2
                specs.append(f'@{position} {field} $CHAR{width}.')
                values.append('00001' if width == 5 else ' 1')
                position += width
            data = '<html>Error</html>' if html else ''.join(values)+'\n'
            for name, text in [(config['data'], data), (config['layout'], '\n'.join(specs))]:
                if name != omit:
                    archive.writestr('NHEFS_data/'+name, text)
            if duplicate and wave == '82':
                archive.writestr('other/'+config['data'], data)


class ArchiveTests(unittest.TestCase):
    def test_valid_format_and_manifest_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'input.zip'; tiny_archive(archive)
            manifest = import_data_zip(archive, root/'work')
            self.assertEqual(manifest['wave_rows'], dict.fromkeys(['82', '86', '87', '92'], 1))
            self.assertEqual(len(manifest['files']), 8)
            self.assertEqual(len(list((root/'work'/'raw').iterdir())), 8)
            self.assertIn('not independently authenticated', manifest['provenance_note'])
            with self.assertRaises(ValueError):
                import_data_zip(archive, root/'work')

    def test_missing_required_file_leaves_no_partial_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'input.zip'; tiny_archive(archive, omit='N92int.txt')
            with self.assertRaisesRegex(ValueError, 'missing required'):
                import_data_zip(archive, root/'work')
            self.assertFalse((root/'work'/'raw').exists())

    def test_html_instead_of_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'input.zip'; tiny_archive(archive, html=True)
            with self.assertRaisesRegex(ValueError, 'HTML response'):
                import_data_zip(archive, root/'work')
            self.assertFalse((root/'work'/'raw').exists())

    def test_ambiguous_duplicate_basename_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'input.zip'; tiny_archive(archive, duplicate=True)
            with self.assertRaisesRegex(ValueError, 'Duplicate input'):
                import_data_zip(archive, root/'work')

    def test_unexpected_data_width_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'input.zip'; tiny_archive(archive)
            with ZipFile(archive) as source:
                files = {i.filename: source.read(i) for i in source.infolist()}
            files['NHEFS_data/N82int.txt'] = b'00001\n'
            with ZipFile(archive, 'w') as destination:
                for name, body in files.items(): destination.writestr(name, body)
            with self.assertRaisesRegex(ValueError, 'unexpected width'):
                import_data_zip(archive, root/'work')
            self.assertFalse((root/'work'/'raw').exists())


if __name__ == '__main__':
    unittest.main()
