import sys
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

path = sys.argv[1]
doc = Document(path)

out = []

def iter_block_items(parent):
    from docx.document import Document as _Doc
    if isinstance(parent, _Doc):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc
    for child in parent_elm.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)

for block in iter_block_items(doc):
    if isinstance(block, Paragraph):
        t = block.text.strip()
        style = block.style.name if block.style is not None else ''
        if t:
            out.append(f"[{style}] {t}")
    else:
        out.append("=== TABLE ===")
        for row in block.rows:
            cells = [c.text.strip().replace('\n', ' / ') for c in row.cells]
            out.append(" | ".join(cells))
        out.append("=== END TABLE ===")

sys.stdout.reconfigure(encoding='utf-8')
print("\n".join(out))
