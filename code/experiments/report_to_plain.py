"""Format the MP1 Markdown report for the built-in macOS PDF print filter."""

import argparse
from pathlib import Path
import re
import textwrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    paragraphs = source.split('\n\n')
    lines = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if paragraph.startswith('# '):
            title = paragraph[2:].upper()
            lines.extend((title, '=' * min(len(title), 72), ''))
            continue
        if paragraph.startswith('## '):
            title = paragraph[3:]
            lines.extend((title, '-' * min(len(title), 72), ''))
            continue
        paragraph = re.sub(r'\*\*|`', '', paragraph)
        paragraph = ' '.join(paragraph.split())
        lines.extend(textwrap.wrap(paragraph, width=74,
                                   break_long_words=False,
                                   break_on_hyphens=False))
        lines.append('')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines).rstrip() + '\n')


if __name__ == '__main__':
    main()
