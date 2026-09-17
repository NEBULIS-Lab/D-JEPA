"""Offline site checks: local URLs, figure slots, brand copies and asset hashes."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/'docs'


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.slots = set()
        self.videos = []
        self.tabs = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'video':
            self.videos.append(attributes)
        if attributes.get('role') == 'tab' or 'data-demo' in attributes:
            self.tabs.append(attributes)
        if 'id' in attributes:
            if attributes['id'] in self.ids:
                raise ValueError(f'duplicate HTML ID: {attributes["id"]}')
            self.ids.add(attributes['id'])
        if 'data-figure' in attributes:
            self.slots.add(attributes['data-figure'])
        for key in ('href', 'src', 'poster'):
            if attributes.get(key):
                self.links.append(attributes[key])


def check():
    page = Page()
    page.feed((SITE/'index.html').read_text())
    if len(page.videos) != 7 or page.tabs:
        raise ValueError('gallery must expose seven videos directly, without task tabs')
    for video in page.videos:
        if not video.get('poster') or video.get('preload') != 'none' or 'controls' not in video:
            raise ValueError('videos need posters, playback controls and no eager preload')
    for link in page.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        if url.path:
            path = (SITE/unquote(url.path)).resolve()
            if SITE.resolve() not in path.parents or not path.is_file():
                raise ValueError(f'nonportable or missing site asset: {link}')
        elif url.fragment and url.fragment not in page.ids:
            raise ValueError(f'unknown section: {link}')
    content = json.loads((SITE/'site-content.json').read_text())
    for name, figure in content['figures'].items():
        if name not in page.slots:
            raise ValueError(f'figure has no matching slot: {name}')
        if figure['src']:
            path = (SITE/figure['src']).resolve()
            if SITE.resolve() not in path.parents or not path.is_file():
                raise ValueError(f'missing final artwork: {name}')
        elif figure['status'] != 'awaiting_author_artwork':
            raise ValueError(f'unexplained empty figure: {name}')
    for record in json.loads((SITE/'ASSET_MANIFEST.json').read_text()):
        path = SITE/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'asset hash changed: {path}')
    for master in (ROOT/'assets/branding').glob('*.svg'):
        if master.read_bytes() != (SITE/'static/images/branding'/master.name).read_bytes():
            raise ValueError(f'brand copy differs: {master.name}')
    print('PASS: local links, sections, artwork, media hashes, logos and directly visible video markup')


if __name__ == '__main__':
    check()
