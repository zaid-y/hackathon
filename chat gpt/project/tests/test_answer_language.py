import json
import re
from pathlib import Path
import pytest
from app.answer import AnswerService
from app.answer_language import detect_language, LanguageRenderingError
from app.grounding import REFUSAL
from app.models import TextChunk
from app.preferences import AnswerOptions
from app.retriever import BM25Retriever
from app.thailmm import ThaiLLMResponse


class Selector:
    def __init__(self, bad=False): self.bad = bad
    def answer(self, system, user):
        return ThaiLLMResponse('invented 999 credits' if self.bad else json.dumps({
            'evidence_ids': [e['id'] for e in json.loads(user)['evidence']]}))


def service(text, bad=False):
    r = BM25Retriever()
    r.build([TextChunk('test', text, 'AIT.pdf', 2, 1, 0, len(text))])
    return AnswerService(retriever=r, provider=Selector(bad))


SOURCE = 'AIT จำนวนหน่วยกิตรวมตลอดหลักสูตร 120 หน่วยกิต\nหลักสูตรปริญญาตรี 4 ปี'


@pytest.mark.parametrize('q,language', [
    ('AIT มีกี่หน่วยกิต', 'th'), ('AIT credits?', 'en'),
    ('KMITL信息技术学院的人工智能技术专业(AIT)总共需要修满多少学分?学制几年?', 'zh'),
])
def test_current_question_language(q, language):
    assert detect_language(q) == language


@pytest.mark.parametrize('q,heading', [
    ('How many credits and how many years for AIT?', 'English'),
    ('AIT多少学分?学制几年?', '中文'),
])
def test_exact_two_sections_same_facts(q, heading):
    s = service(SOURCE)
    a = s.answer(q, options=AnswerOptions(answer_language='th'))
    thai, foreign = a.answer.split(f'\n\n{heading}:\n')
    assert thai.startswith('ภาษาไทย:\n')
    assert a.grounded and REFUSAL not in a.answer
    assert re.findall(r'\d+', thai) == re.findall(r'\d+', foreign)
    assert '120' in foreign and '4' in foreign
    assert not re.search('[\u0e00-\u0e7f]', foreign)
    assert s.last_debug['final_validated_answer'] == a.answer
    assert len(a.sources) == 1 and a.sources[0].page == 2


def test_thai_once_even_if_saved_preference_is_english():
    a = service(SOURCE).answer('AIT มีกี่หน่วยกิตและเรียนกี่ปี', options=AnswerOptions(answer_language='en'))
    assert not a.answer.startswith('ภาษาไทย:')
    assert 'English:' not in a.answer and '中文:' not in a.answer
    assert '120' in a.answer and '4' in a.answer


@pytest.mark.parametrize('bad', [False, True])
@pytest.mark.parametrize('q,heading', [('AIT多少学分?', '中文'), ('AIT credits?', 'English')])
def test_refusal_two_sections_no_success_or_sources(q, heading, bad):
    s = service(SOURCE if bad else 'ไม่มีข้อมูล', bad)
    a = s.answer(q)
    assert a.answer.startswith('ภาษาไทย:\n'+REFUSAL)
    assert f'\n\n{heading}:\n' in a.answer
    assert not a.grounded and not a.sources and '120' not in a.answer
    assert s.last_debug['final_validated_answer'] == a.answer


@pytest.mark.parametrize('q,heading,missing', [
    ('How many credits and how many years for AIT?', 'English', 'Study duration'),
    ('AIT多少学分?学制几年?', '中文', '学制'),
])
def test_partial_information_same_missing_field(q, heading, missing):
    a = service(SOURCE.split('\n')[0]).answer(q)
    thai, foreign = a.answer.split(f'\n\n{heading}:\n')
    assert 'ข้อมูลที่พบ:' in thai and 'ข้อมูลที่ไม่พบ:' in thai
    assert missing in foreign and '120' in foreign
    assert REFUSAL not in a.answer and '4' not in a.answer


def test_unknown_career_translation_not_silently_dropped_or_invented():
    s = service('AIT อาชีพหลังสำเร็จการศึกษา\n(1) ผู้เชี่ยวชาญเฉพาะตำแหน่งใหม่')
    with pytest.raises(LanguageRenderingError): s.answer('AIT careers?')
    assert s.last_debug['final_validated_answer'] is None
    assert 'verified_translation_unavailable' in s.last_debug['validation_errors']


def test_conflicting_values_in_both_languages_with_sources():
    a = service(SOURCE+'\nจำนวนหน่วยกิตรวมตลอดหลักสูตร 129 หน่วยกิต').answer('AIT credits?')
    thai, foreign = a.answer.split('\n\nEnglish:\n')
    assert 'ขัดแย้ง' in thai and 'Conflicting' in foreign
    for part in (thai, foreign): assert '120' in part and '129' in part and 'AIT.pdf' in part


def test_coop_minimum_weeks_does_not_become_minimum_credits():
    a = service('AIT สหกิจศึกษา\n6(0-45-0)\nระยะเวลาอย่างน้อย16สัปดาห์').answer('AIT co-op credits?')
    assert '6 credits' in a.answer and 'at least 6' not in a.answer


@pytest.mark.parametrize('q,heading', [('AIT毕业生可以从事哪些职业?', '中文'), ('AIT careers?', 'English')])
def test_real_careers_have_same_twelve_entries(q, heading):
    path = Path(__file__).resolve().parents[1]/'data/index/bm25_index.json'
    if not path.exists(): pytest.skip('Competition index not installed')
    s = AnswerService(retriever=BM25Retriever.load(path), provider=Selector())
    a = s.answer(q)
    thai, foreign = a.answer.split(f'\n\n{heading}:\n')
    assert a.grounded
    assert len(re.findall(r'^\(\d+\)',thai,re.M)) == 12
    assert len(re.findall(r'^\(\d+\)',foreign,re.M)) == 12
    assert re.findall(r'\d+',thai) == re.findall(r'\d+',foreign)
    assert not re.search('[\u0e00-\u0e7f]',foreign)
