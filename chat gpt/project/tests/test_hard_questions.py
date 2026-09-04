import json
from pathlib import Path
import pytest
from app.answer import AnswerService
from app.grounding import REFUSAL
from app.models import TextChunk
from app.retriever import BM25Retriever
from app.thailmm import ThaiLLMResponse


class Selector:
    def __init__(self): self.calls = 0
    def answer(self, system, user):
        self.calls += 1
        return ThaiLLMResponse(json.dumps({'evidence_ids':[e['id'] for e in json.loads(user)['evidence']]}))


def service(rows):
    r=BM25Retriever()
    r.build([TextChunk(str(i), text, doc, page, 1, 0, len(text)) for i,(doc,page,text) in enumerate(rows)])
    return AnswerService(retriever=r, provider=Selector())


@pytest.mark.parametrize('q', [
    'อาจารย์ในคณะแนะนำให้อ่านหนังสือเตรียมสอบ แต่ผมอยากได้สรุปเนื้อหาแคลคูลัส 1 ทั้งเทอมแบบละเอียด ช่วยสรุปให้หน่อย',
    'ช่วยสอนแคลคูลัสทั้งภาคการศึกษา', 'Teach me calculus for the whole semester',
    '请讲解微积分整个学期的内容',
])
def test_teaching_not_calendar(q):
    s=service([('AIT.pdf',10,'ใช้ระบบทวิภาค โดยใน1ปีการศึกษาแบ่งออกเป็น2ภาคการศึกษาปกติ\nภาคการศึกษาที่2เดือนธันวาคม–เดือนเมษายน')])
    result=s.answer(q)
    assert not result.grounded and not result.sources
    assert REFUSAL in result.answer and s.provider.calls==0
    assert 'ธันวาคม' not in result.answer


@pytest.mark.parametrize('q,ordered,reversed_', [
    ('AIT ภาคการศึกษาที่2เรียนเดือนไหน', 'ธันวาคม–เมษายน','เมษายน–ธันวาคม'),
    ('AIT semesters?', 'December–April','April–December'),
    ('AIT学期怎么划分?', '十二月–四月','四月–十二月'),
])
def test_source_month_order_and_opening_announcement_exclusion(q, ordered, reversed_):
    s=service([('AIT.pdf',10,'ภาคการศึกษาที่2เดือนธันวาคม–เดือนเมษายน'),
               ('AIT.pdf',4,'หลักสูตรใหม่ กำหนดเปิดสอนเดือนกรกฎาคม2566(ภาคการศึกษาที่1/2566)')])
    a=s.answer(q)
    assert a.grounded and ordered in a.answer and reversed_ not in a.answer
    assert all(c.page==10 for c in a.sources)
    assert '2566' not in a.answer


def credit_rows():
    return [('AIT.pdf',12,'ข.หมวดวิชาเฉพาะ90หน่วยกิต'),
            ('DSBA.pdf',14,'ข.หมวดวิชาเฉพาะ96หน่วยกิต'),
            ('IT2565.pdf',15,'ข.หมวดวิชาเฉพาะ93หน่วยกิต'),
            ('IT_inter2565.pdf',14,'ข.หมวดวิชาเฉพาะ90หน่วยกิต')]


def test_different_program_values_are_not_conflicts_and_sort_descending():
    a=service(credit_rows()).answer('หมวดวิชาเฉพาะของแต่ละหลักสูตรมีกี่หน่วยกิต เรียงลำดับจากมากไปน้อย')
    assert 'พบข้อมูลขัดแย้งกัน' not in a.answer
    assert a.answer.index('DSBA:') < a.answer.index('IT2565:') < a.answer.index('AIT:')
    assert 'IT_inter2565:' in a.answer
    assert len(a.sources)==4


def test_real_intra_program_conflict_keeps_both_values_out_of_definitive_ranking():
    rows=credit_rows()+[('IT_inter2565.pdf',17,'ข.หมวดวิชาเฉพาะ96หน่วยกิต')]
    a=service(rows).answer('这四个专业的专业课程类学分从高到低如何排列?')
    thai, chinese=a.answer.split('\n\n中文:\n')
    for part in (thai,chinese):
        assert part.index('DSBA:') < part.index('IT2565:') < part.index('AIT:') < part.index('IT_inter2565:')
        assert '90' in part.split('IT_inter2565:')[1] and '96' in part.split('IT_inter2565:')[1]
    assert thai.count('พบข้อมูลขัดแย้งกัน')==1
    assert chinese.count('文档信息存在冲突')==1


@pytest.mark.parametrize('q', [
    'หากสนใจสายงานด้าน AI และ Data โดยเฉพาะ ควรเลือกเรียนหลักสูตรใดระหว่าง AIT กับ DSBA และทั้งสองหลักสูตรต่างกันอย่างไร',
    '如果对人工智能和数据方向感兴趣,应该选择AIT还是DSBA?两者有什么区别?',
    'Should I choose AIT or DSBA? What is the difference?',
])
def test_comparison_includes_both_programs_without_invented_recommendation(q):
    s=service([('AIT.pdf',2,'เทคโนโลยีปัญญาประดิษฐ์\nจำนวนหน่วยกิตรวมตลอดหลักสูตร120หน่วยกิต\nหลักสูตรปริญญาตรี4ปี'),
               ('DSBA.pdf',2,'วิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ\nจำนวนหน่วยกิตรวมตลอดหลักสูตร129หน่วยกิต\nหลักสูตรปริญญาตรี4ปี')])
    a=s.answer(q)
    assert a.grounded and {c.document for c in a.sources}=={'AIT.pdf','DSBA.pdf'}
    assert '120' in a.answer and '129' in a.answer
    assert 'ข้อมูลที่ไม่พบ:' in a.answer and REFUSAL not in a.answer
    assert 'พบข้อมูลขัดแย้งกัน' not in a.answer
    if '如果' in q: assert '\n\n中文:\n' in a.answer
    if 'Should' in q: assert '\n\nEnglish:\n' in a.answer


def test_missing_comparison_field_not_borrowed_from_other_program():
    s=service([('AIT.pdf',2,'เทคโนโลยีปัญญาประดิษฐ์\nจำนวนหน่วยกิตรวมตลอดหลักสูตร120หน่วยกิต\nหลักสูตรปริญญาตรี4ปี'),
               ('DSBA.pdf',2,'วิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ\nจำนวนหน่วยกิตรวมตลอดหลักสูตร129หน่วยกิต')])
    a=s.answer('Compare AIT vs DSBA differences')
    assert 'DSBA — Study duration' in a.answer


def test_comparison_includes_additional_program_document_conflict():
    rows=credit_rows()+[('AIT-update.pdf',3,'AIT ข.หมวดวิชาเฉพาะ91หน่วยกิต')]
    a=service(rows).answer('หมวดวิชาเฉพาะของแต่ละหลักสูตรมีกี่หน่วยกิต เรียงลำดับจากมากไปน้อย')
    assert 'AIT-update.pdf' in a.answer and '91' in a.answer
    assert 'พบข้อมูลขัดแย้งกัน' in a.answer


def test_real_corpus_rank_and_comparison():
    path=Path(__file__).resolve().parents[1]/'data/index/bm25_index.json'
    if not path.exists(): pytest.skip('Competition index not installed')
    s=AnswerService(retriever=BM25Retriever.load(path),provider=Selector())
    a=s.answer('这四个专业的专业课程类学分从高到低如何排列?')
    assert a.grounded and '\n\n中文:\n' in a.answer
    assert a.answer.count('พบข้อมูลขัดแย้งกัน')==1
    a=s.answer('如果对人工智能和数据方向感兴趣,应该选择AIT还是DSBA?两者有什么区别?')
    assert a.grounded and '\n\n中文:\n' in a.answer
    assert {c.document for c in a.sources}=={'AIT.pdf','DSBA.pdf'}
