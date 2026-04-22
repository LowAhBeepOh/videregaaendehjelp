#!/usr/bin/env python3
"""
Generate search index with semantic embeddings from guide content.
Run this whenever you update guides: python generate-search-index.py
"""

import os
import json
import re
from html.parser import HTMLParser
from collections import Counter
from pathlib import Path
import math

class TextExtractor(HTMLParser):
    """Extract text from HTML, excluding scripts and styles."""
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.heading_parts = []  # h1, h2, h3
        self.skip = False
        self.title = ""
        self.in_title = False
        self.in_heading = False
        self.heading_level = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self.in_title = True
        elif tag in ('h1', 'h2', 'h3'):
            self.in_heading = True
            self.heading_level = int(tag[1])
        elif tag in ('script', 'style', 'meta', 'link', 'nav', 'footer'):
            self.skip = True

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False
        elif tag in ('h1', 'h2', 'h3'):
            self.in_heading = False
            self.heading_level = 0
        elif tag in ('script', 'style', 'nav', 'footer'):
            self.skip = False

    def handle_data(self, data):
        if self.in_title:
            self.title = data.strip()
        elif self.in_heading and not self.skip:
            # Weight headings by level: h1=5x, h2=3x, h3=2x
            weight = {1: 5, 2: 3, 3: 2}.get(self.heading_level, 1)
            self.heading_parts.extend([data.strip()] * weight)
        elif not self.skip:
            self.text_parts.append(data)

    def get_text(self):
        # Combine body text with weighted headings
        all_text = self.heading_parts + self.text_parts
        return ' '.join(all_text)

    def get_title(self):
        # Clean up title - remove various dash types and " Videregående Hjelp" suffix
        if self.title:
            title = self.title
            # Remove with em dash (—), en dash (–), regular dash (-)
            title = title.replace(' — Videregående Hjelp', '')
            title = title.replace(' – Videregående Hjelp', '')
            title = title.replace(' - Videregående Hjelp', '')
            return title.strip()
        return ""

# Norwegian stop words (Bokmål & Nynorsk)
NORWEGIAN_STOP_WORDS = {
    # Bokmål
    'og', 'i', 'jeg', 'det', 'at', 'en', 'til', 'er', 'som', 'på', 'de', 'med', 'han', 'av', 'for', 'ikke', 'der', 'var', 'meg', 'seg', 'så', 'over', 'fra', 'hun', 'om', 'har', 'ham', 'hans', 'hvor', 'da', 'skulle', 'eller', 'hva', 'dette', 'denne', 'disse', 'hvis', 'sin', 'sitt', 'hennes', 'deres', 'vårt', 'ditt', 'mitt', 'din', 'do', 'vi', 'alle', 'også', 'kan', 'kunne', 'hadde', 'aner', 'anna', 'anno', 'åt', 'være', 'vær', 'værd', 'ville', 'vil', 'viss', 'ja', 'nei', 'nein', 'nej', 'jo', 'du', 'deg',
    # Nynorsk
    'eg', 'ein', 'dei', 'ikkje', 'frå', 'ho', 'honom', 'kvar', 'då', 'kva', 'desse', 'dét', 'hennar', 'deira', 'gjer', 'me', 'dykk',
    # Common Scandinavian
    'her', 'men', 'alltid', 'selvfølgelig', 'mange', 'gjøre', 'gjør', 'gjort',
    # Settings/navigation
    'settings', 'hjem', 'hallo', 'start', 'forside', 'guide', 'guider', 'verktøy', 'tool', 'tools', 'interaktivt', 'interactive', 'søk', 'search', 'navbar', 'footer', 'header', 'menu', 'nav',
}

def extract_text_from_html(filepath):
    """Extract clean text and title from HTML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()

        parser = TextExtractor()
        parser.feed(html_content)
        text = parser.get_text()
        title = parser.get_title()

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text, title
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return "", ""

def tokenize(text):
    """Simple tokenization - split on non-alphanumeric, lowercase."""
    # Keep underscores for Norwegian compound words
    tokens = re.findall(r'\b[a-zæøåA-ZÆØÅ0-9]+\b', text.lower())
    # Filter out stop words and very short words
    return [t for t in tokens if len(t) > 2 and t not in NORWEGIAN_STOP_WORDS]

def build_vocabulary(all_tokens):
    """Build vocabulary from all tokens."""
    vocab = {}
    for i, word in enumerate(sorted(set(all_tokens))):
        vocab[word] = i
    return vocab

def calculate_idf(all_items, vocab):
    """Calculate IDF (Inverse Document Frequency) for each term."""
    N = len(all_items)
    doc_freq = Counter()
    
    for item in all_items:
        unique_tokens = set(item['tokens'])
        for token in unique_tokens:
            if token in vocab:
                doc_freq[token] += 1
    
    # IDF = log(N / df) + 1 (smoothing)
    idf = {}
    for token, df in doc_freq.items():
        idf[token] = math.log(N / df) + 1
    return idf

def create_tfidf_vector(tokens, vocab, idf):
    """Create TF-IDF weighted vector."""
    vector = [0.0] * len(vocab)
    token_count = Counter(tokens)
    total = len(tokens)
    
    for token, count in token_count.items():
        if token in vocab:
            # TF = count / total
            tf = count / total if total > 0 else 0
            # TF-IDF weighting
            vector[vocab[token]] = tf * idf.get(token, 1.0)
    
    return vector

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    if mag1 == 0 or mag2 == 0:
        return 0
    return dot_product / (mag1 * mag2)

def main():
    # Define paths
    guides_dir = Path('Guides')
    tools_dir = Path('Tools')
    interactives_dir = Path('Interactives')

    index_data = []
    all_tokens = []
    all_items = []

    # Process guides
    print("📚 Indexing guides...")
    if guides_dir.exists():
        for html_file in sorted(guides_dir.glob('*.html')):
            text, title = extract_text_from_html(html_file)
            tokens = tokenize(text)
            all_tokens.extend(tokens)

            # Use extracted title, fallback to filename if empty
            if not title:
                title = ' '.join(html_file.stem.split('-')).title()

            item = {
                'title': title,
                'url': str(html_file),
                'type': 'guide',
                'content': text[:500],  # Store first 500 chars for reference
                'tokens': tokens,
                'keywords': list(set(tokens[:20]))  # Most common tokens as keywords
            }
            all_items.append(item)
            print(f"  ✓ {title}")

    # Process tools
    print("🔧 Indexing tools...")
    if tools_dir.exists():
        for html_file in sorted(tools_dir.glob('*.html')):
            text, title = extract_text_from_html(html_file)
            tokens = tokenize(text)
            all_tokens.extend(tokens)

            if not title:
                title = ' '.join(html_file.stem.split('-')).title()

            item = {
                'title': title,
                'url': str(html_file),
                'type': 'tool',
                'content': text[:500],
                'tokens': tokens,
                'keywords': list(set(tokens[:20]))
            }
            all_items.append(item)
            print(f"  ✓ {title}")

    # Process interactives
    print("🎮 Indexing interactives...")
    if interactives_dir.exists():
        for html_file in sorted(interactives_dir.glob('*.html')):
            text, title = extract_text_from_html(html_file)
            tokens = tokenize(text)
            all_tokens.extend(tokens)

            if not title:
                title = ' '.join(html_file.stem.split('-')).title()

            item = {
                'title': title,
                'url': str(html_file),
                'type': 'interactive',
                'content': text[:500],
                'tokens': tokens,
                'keywords': list(set(tokens[:20]))
            }
            all_items.append(item)
            print(f"  ✓ {title}")

    # Build vocabulary
    print("\n🔤 Building vocabulary...")
    vocab = build_vocabulary(all_tokens)
    print(f"   Vocabulary size: {len(vocab)} unique terms")
    
    # Calculate IDF weights
    print("📊 Calculating IDF weights...")
    idf = calculate_idf(all_items, vocab)
    
    # Create TF-IDF vectors for each item
    print("📐 Creating TF-IDF embeddings...")
    for item in all_items:
        vector = create_tfidf_vector(item['tokens'], vocab, idf)
        # Store only non-zero entries to save space (sparse vector)
        sparse_vector = {str(i): round(v, 6) for i, v in enumerate(vector) if v > 0}
        item['vector'] = sparse_vector
        del item['tokens']  # Remove tokens from output

    # Generate final output
    output = {
        'vocab_size': len(vocab),
        'idf': {k: round(v, 6) for k, v in idf.items()},  # Include IDF for query processing
        'items': []
    }

    for item in all_items:
        output_item = {
            'title': item['title'],
            'url': item['url'],
            'type': item['type'],
            'keywords': item['keywords'],
            'vector': item['vector']
        }
        output['items'].append(output_item)

    # Write to search-data.json
    print("\n💾 Writing search-data.json...")
    with open('search-data.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Indexed {len(all_items)} items with {len(vocab)} vocabulary terms")
    print(f"   Output: search-data.json ({os.path.getsize('search-data.json') / 1024:.1f} KB)")

if __name__ == '__main__':
    main()
