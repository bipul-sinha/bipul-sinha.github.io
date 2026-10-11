#!/usr/bin/env python3
"""Fetch higher-resolution covers for fiction.json from Google Books.

For each book, looks up the best matching volume on the Google Books API and
downloads its largest available cover image. A new image only replaces the
existing cover when it is taller, so a miss never makes a cover worse.
Books whose cover already came from Google Books are skipped.

Google Books has a daily query quota. When it runs out, the script stops and
records where it got to, so re-running it later continues from there.

Usage:
    python3 scripts/fetch_fiction_covers.py
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'fiction.json')
COVER_DIR = os.path.join(REPO, 'covers', 'fiction')
UA = {'User-Agent': 'Mozilla/5.0 (personal site cover fetch)'}
IMAGE_KEYS = ['extraLarge', 'large', 'medium', 'small', 'thumbnail', 'smallThumbnail']


def norm(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


def slugify(text):
    return re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', text.lower())).strip('-')


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read()


def image_height(path):
    out = subprocess.run(['sips', '-g', 'pixelHeight', path], capture_output=True, text=True).stdout
    m = re.search(r'pixelHeight: (\d+)', out)
    return int(m.group(1)) if m else 0


def find_google_image(book):
    """Return a high-resolution image URL for the book, or None if no match."""
    last = norm(book['author'].split()[-1])
    query = f"intitle:{book['title']}"
    if last:
        query += f" inauthor:{last}"
    url = 'https://www.googleapis.com/books/v1/volumes?' + urllib.parse.urlencode(
        {'q': query, 'maxResults': 5, 'printType': 'books'})
    items = json.loads(get(url)).get('items', [])
    want = norm(book['title'])
    for item in items:
        info = item.get('volumeInfo', {})
        got = norm(info.get('title', ''))
        authors = [norm(a) for a in info.get('authors', [])]
        if not (got == want or got.startswith(want) or want.startswith(got)):
            continue
        if last and not any(last in a for a in authors):
            continue
        links = info.get('imageLinks', {})
        for key in IMAGE_KEYS:
            if key in links:
                # zoom=0 asks for the largest size Google has for this volume.
                img = re.sub(r'&edge=curl', '', links[key])
                img = re.sub(r'zoom=\d', 'zoom=0', img)
                return img.replace('http://', 'https://')
    return None


def main():
    books = json.load(open(DATA, encoding='utf-8'))
    replaced = skipped = missed = 0
    for i, book in enumerate(books):
        if book.get('cover_source') == 'google':
            skipped += 1
            continue
        try:
            img_url = find_google_image(book)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f'Google Books quota exhausted at "{book["title"]}". Re-run tomorrow to continue.')
                break
            print(f'HTTP {e.code} for "{book["title"]}", skipping')
            missed += 1
            continue
        except Exception as e:
            print(f'Error for "{book["title"]}": {e}')
            missed += 1
            continue
        time.sleep(1)
        if not img_url:
            missed += 1
            print(f'no match: {book["title"]} | {book["author"]}')
            continue

        slug = slugify(book['title'])
        tmp = os.path.join(COVER_DIR, f'.{slug}.new.jpg')
        try:
            open(tmp, 'wb').write(get(img_url))
        except Exception as e:
            print(f'download failed for "{book["title"]}": {e}')
            missed += 1
            continue
        dest = os.path.join(COVER_DIR, f'{slug}.jpg')
        old_h = image_height(dest) if os.path.exists(dest) else 0
        new_h = image_height(tmp)
        if new_h > old_h:
            os.replace(tmp, dest)
            book['cover'] = f'covers/fiction/{slug}.jpg'
            book['cover_source'] = 'google'
            replaced += 1
            print(f'{book["title"]}: {old_h}px -> {new_h}px')
        else:
            os.remove(tmp)
            print(f'{book["title"]}: kept existing ({old_h}px >= {new_h}px)')
        time.sleep(1)
        # Save after each book so a quota stop loses nothing.
        json.dump(books, open(DATA, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    json.dump(books, open(DATA, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'replaced: {replaced}, already from Google: {skipped}, not found: {missed}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
