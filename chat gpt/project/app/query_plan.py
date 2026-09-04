"""Bounded local intent/scope planning. No remote query processing."""
import re

PROGRAMS = {
    'AIT.pdf': ('AIT', 'เทคโนโลยีปัญญาประดิษฐ์', 'Artificial Intelligence Technology', '人工智能技术'),
    'DSBA.pdf': ('DSBA', 'วิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ', 'Data Science and Business Analytics', '数据科学与商业分析'),
    'IT2565.pdf': ('IT2565', 'เทคโนโลยีสารสนเทศ', 'Information Technology', '信息技术'),
    'IT_inter2565.pdf': ('IT_inter2565', 'เทคโนโลยีสารสนเทศทางธุรกิจ', 'Business Information Technology', '商业信息技术'),
}


def subject(document):
    if document in PROGRAMS: return PROGRAMS[document][0]
    for data in PROGRAMS.values():
        if re.match(re.escape(data[0])+r'[-_]', document, re.I): return data[0]
    return None


def plan(question):
    q = question.casefold()
    entities = [doc for doc, data in PROGRAMS.items()
                if re.search(r'(?<![a-z0-9_])'+re.escape(data[0].casefold())+r'(?![a-z0-9_])', q)]
    all_programs = any(t in q for t in ('แต่ละหลักสูตร','ทุกหลักสูตร','ทั้ง 4 หลักสูตร','这四个','四个专业','all four','each program','each curriculum'))
    ranking = any(t in q for t in ('มากไปน้อย','高到低','descending','highest to lowest'))
    comparison = len(entities)>1 and any(t in q for t in ('ต่างกัน','เปรียบเทียบ','เลือก','区别','选择','compare','difference','choose',' vs '))
    if all_programs: entities = list(PROGRAMS)
    # An incidental mention of a term/semester in a teaching request does not
    # request the academic calendar. No lecture-summary evidence path exists.
    teaching = any(t in q for t in ('สรุปเนื้อหา','สรุปแคลคูลัส','ช่วยสอน','สอนแคลคูลัส','สรุปให้หน่อย','teach me','summarize calculus','explain calculus','讲解微积分','总结微积分'))
    return {'documents': entities if (all_programs or comparison) else [],
            'ranking': ranking, 'comparison': comparison, 'teaching': teaching,
            'recommendation': comparison and any(t in q for t in ('เลือก','选择','choose','recommend'))}
