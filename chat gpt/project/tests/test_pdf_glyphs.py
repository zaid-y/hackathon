from pathlib import Path
import hashlib
import re
import pytest
from app.pdf_glyphs import _unique_target
from app.text_extractor import DocumentTextExtractor
from app.chunker import DocumentChunker


def test_unknown_or_ambiguous_glyph_is_not_guessed():
    assert _unique_target(set()) is None
    assert _unique_target({0xe48, 0xe49}) is None
    assert _unique_target({0xe49}) == 0xe49


def test_non_pdf_private_use_characters_are_untouched(tmp_path):
    path = tmp_path / 'text.txt'
    text = 'AIT IT2565 120 129 2566 中文 English \ue04a \uf701'
    path.write_text(text, encoding='utf-8')
    assert DocumentTextExtractor().extract(path).pages[0].text == text


@pytest.mark.parametrize('name', ['AIT.pdf', 'DSBA.pdf', 'IT2565.pdf', 'IT_inter2565.pdf'])
def test_real_pdf_thai_glyphs_and_provenance(name):
    path = Path(__file__).resolve().parents[1] / 'data/documents' / name
    if not path.exists():
        pytest.skip('Competition PDF not installed')
    before = hashlib.sha256(path.read_bytes()).digest()
    document = DocumentTextExtractor().extract(path)
    assert hashlib.sha256(path.read_bytes()).digest() == before
    for page in document.pages:
        # Wingdings checkbox symbols are intentionally not treated as Thai.
        assert not re.search('[\ue043-\ue096\uf700-\uf71d]', page.text)
        assert page.document == name and page.page >= 1
        for chunk in DocumentChunker().chunk_page(page):
            assert chunk.text == page.text[chunk.start_char:chunk.end_char]
    if name == 'AIT.pdf':
        careers = next(p.text for p in document.pages if p.page == 4)
        assert 'วิศวกรการเรียนรู้ของเครื่อง' in careers
        assert 'นักวิทยาศาสตร์ด้านการเรียนรู้เชิงลึก' in careers
        assert 'ผู้เชี่ยวชาญด้านการประมวลผลภาษามนุษย์' in careers
        assert '(12) ที่ปรึกษาด้านปัญญาประดิษฐ์' in careers
        intro = next(p.text for p in document.pages if p.page == 2)
        assert '120' in intro and '4ปี' in intro.replace(' ', '')
