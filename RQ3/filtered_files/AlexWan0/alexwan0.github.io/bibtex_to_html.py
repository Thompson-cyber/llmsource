# Generated by ChatGPT

import argparse
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

HTML_TEMPLATE = """
<div id="publications_content" class="row">
    <div class="col paper_thumbnail is-horizontal-align is-vertical-align">
        <img src="[IMAGE PATH]" alt="[IMAGE ALT TEXT]">
    </div>
    <div class="paper col-10">
        <p>
            <b>{title}</b>
        </p>
        <p>
            {authors}
        </p>
        <p>
            <i>{venue} {year}</i>
        </p>
        <p>
            <a href="[LINK]">arXiv</a>
        </p>
    </div>
</div>
"""

def parse_bibtex(bibtex_file, author_name):
    with open(bibtex_file, 'r', encoding='utf-8') as file:
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        bib_database = bibtexparser.load(file, parser=parser)
        entries = bib_database.entries

    html_output = ""
    for entry in entries:
        title = entry.get('title', '')
        authors = entry.get('author', '')
        venue = entry.get('journal', '') or entry.get('booktitle', '')
        year = entry.get('year', '')

        if author_name and author_name in authors:
            authors = authors.replace(author_name, "<u>{}</u>".format(author_name))

        html_output += HTML_TEMPLATE.format(title=title, authors=authors, venue=venue, year=year)

    return html_output

# Create command-line argument parser
parser = argparse.ArgumentParser(description='Parse BibTeX and generate HTML output')
parser.add_argument('bibtex_file', type=str, help='path to the BibTeX file')
parser.add_argument('--author', type=str, help='author name to underline')

# Parse command-line arguments
args = parser.parse_args()
bibtex_file = args.bibtex_file
author_name = args.author

# Process the BibTeX file and generate HTML output
html = parse_bibtex(bibtex_file, author_name)
print(html)
