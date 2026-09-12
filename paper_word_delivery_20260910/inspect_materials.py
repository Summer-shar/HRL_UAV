from pathlib import Path
import json
import hashlib
import subprocess
from pypdf import PdfReader
from docx import Document
from docx.oxml.ns import qn
from lxml import etree

root = Path(__file__).resolve().parent
evidence = {}
for name in ('template_reference.pdf', 'qa_test/test.pdf'):
    pdf = PdfReader(root / name)
    out = root / ('qa_template' if name.startswith('template') else 'qa_test')
    out.mkdir(exist_ok=True)
    if not (out/'page-1.png').exists():
        subprocess.run([r'C:\Users\PC\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe',
                        '-r','90','-png',str(root/name),str(out/'page')],check=True)
    evidence[name] = {'pages': len(pdf.pages), 'page_sizes': [list(p.mediabox) for p in pdf.pages]}

doc = Document(root / 'template_reference.docx')
evidence['sections'] = [
    {'width_mm': s.page_width.mm, 'height_mm': s.page_height.mm,
     'top_mm': s.top_margin.mm, 'bottom_mm': s.bottom_margin.mm,
     'left_mm': s.left_margin.mm, 'right_mm': s.right_margin.mm,
     'header_mm': s.header_distance.mm, 'footer_mm': s.footer_distance.mm,
     'columns': etree.tostring(s._sectPr.find(qn('w:cols')),encoding='unicode') if s._sectPr.find(qn('w:cols')) is not None else ''}
    for s in doc.sections]
paragraphs = []
for i,p in enumerate(doc.paragraphs):
    if not p.text.strip(): continue
    paragraphs.append({'i':i, 'text':p.text, 'style':p.style.name,
                       'runs': [{'text':r.text[:40], 'size_pt':r.font.size.pt if r.font.size else None,
                                 'font':r.font.name,'bold':r.font.bold,
                                 'eastAsia':r._element.rPr.rFonts.get(qn('w:eastAsia')) if r._element.rPr is not None and r._element.rPr.rFonts is not None else None}
                                for r in p.runs[:5]]})
evidence['paragraphs'] = paragraphs
evidence['original_template_sha256'] = hashlib.sha256(Path(r'C:\Users\PC\Desktop\《通信学报》正文模版 .doc').read_bytes()).hexdigest()
(root / 'template_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in evidence.items() if k!='paragraphs'}, ensure_ascii=True))
print(json.dumps(paragraphs[:20], ensure_ascii=True))
