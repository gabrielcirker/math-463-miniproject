"""Navigation, asset, and original report checks for the public website."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse
import hashlib
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {'report/final-report.pdf': '8b825538f082621999864deaaf0bacb9dc0450c121eec88db4cad3fdcc919fb3'}


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.navigation = []
        self.ids = []
        self.headings = 0
        self.in_navigation = False
    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == 'nav':
            self.in_navigation = attrs.get('aria-label') == 'Main navigation'
        if tag == 'h1':
            self.headings += 1
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'a' and self.in_navigation:
            self.navigation.append(attrs.get('href', ''))
        for key in ('href', 'src', 'data'):
            if key in attrs:
                self.references.append(attrs[key])
    def handle_endtag(self, tag):
        if tag == 'nav':
            self.in_navigation = False


def read_page(path):
    page = Page()
    page.feed(path.read_text())
    return page


class PresentationChecks(unittest.TestCase):
    def test_each_page_has_clear_navigation(self):
        for path in ROOT.rglob('*.html'):
            with self.subTest(page=str(path.relative_to(ROOT))):
                page = read_page(path)
                self.assertEqual(page.headings, 1)
                self.assertEqual(len(page.ids), len(set(page.ids)))
                self.assertGreaterEqual(len(page.navigation), 3)
                for url in page.navigation:
                    self.assertTrue(url)
                    self.assertFalse(urlparse(url).fragment, url)

    def test_links_assets_and_fragments(self):
        for path in ROOT.rglob('*.html'):
            for reference in read_page(path).references:
                url = urlparse(reference)
                if url.scheme or url.netloc:
                    continue
                target = path.parent / unquote(url.path) if url.path else path
                if target.is_dir():
                    target /= 'index.html'
                with self.subTest(page=str(path.relative_to(ROOT)), link=reference):
                    self.assertTrue(target.exists(), str(target))
                    if url.fragment and target.suffix == '.html':
                        self.assertIn(url.fragment, read_page(target).ids)

    def test_original_reports_are_unchanged(self):
        for path, expected in REPORTS.items():
            with self.subTest(report=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
