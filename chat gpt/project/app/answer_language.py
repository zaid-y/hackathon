"""Thai-first presentation from shared evidence facts, never model translations.

English/Chinese curriculum wording is reviewed and finite. Unknown translations
raise an explicit processing error; they are NOT misreported as missing evidence.
"""
from dataclasses import dataclass
import re
from app.grounding import REFUSAL, LABELS, render_answer


class LanguageRenderingError(RuntimeError):
    pass


def detect_language(question):
    # Detect the current question only, before Thai retrieval aliases are added.
    if re.search('[\u3040-\u30ff]', question): return 'ja'
    if re.search('[\uac00-\ud7af]', question): return 'ko'
    if re.search('[\u3400-\u9fff]', question): return 'zh'
    if re.search('[\u0e00-\u0e7f]', question): return 'th'
    if re.search('[A-Za-z]', question): return 'en'
    return 'th'


TRANSLATIONS = {
    'en': ('English', 'There is not enough relevant information in the provided documents.'),
    'zh': ('中文', '提供的文档中没有足够的相关信息。'),
    'ja': ('日本語', '提供された文書には十分な関連情報がありません。'),
    'ko': ('한국어', '제공된 문서에 관련 정보가 충분하지 않습니다.'),
}


def refusal(language):
    if language == 'th': return REFUSAL
    heading, message = TRANSLATIONS[language]
    return f'ภาษาไทย:\n{REFUSAL}\n\n{heading}:\n{message}'


def compact(text):
    return re.sub(r'\s+', '', text).casefold()


# Thai source wording, English equivalent, Chinese equivalent. No role inferred
# from general knowledge; a complete matching career item is required.
CAREERS = [
    ('นักวิทยาศาสตร์ด้านการเรียนรู้เชิงลึก', 'Deep learning scientist', '深度学习科学家'),
    ('นักวิทยาศาสตร์หรือวิศวกรข้อมูล', 'Data scientist or data engineer', '数据科学家或数据工程师'),
    ('ผู้พัฒนาระบบธุรกิจอัจฉริยะ', 'Business intelligence system developer', '商业智能系统开发人员'),
    ('วิศวกรการเรียนรู้ของเครื่อง', 'Machine learning engineer', '机器学习工程师'),
    ('นักวิจัยและนักวิชาการด้านปัญญาประดิษฐ์', 'Artificial intelligence researcher and academic', '人工智能研究人员和学者'),
    ('วิศวกรวางสถาปัตยกรรมข้อมูลขนาดใหญ่', 'Big data architecture engineer', '大数据架构工程师'),
    ('นักพัฒนาระบบฝังตัวหรือระบบเชื่อมต่อสรรพสิ่ง', 'Embedded systems or Internet of Things developer', '嵌入式系统或物联网开发人员'),
    ('ผู้เชี่ยวชาญด้านการมองเห็นด้วยเครื่องจักร', 'Computer vision specialist', '计算机视觉专家'),
    ('ผู้เชี่ยวชาญด้านการประมวลผลภาษามนุษย์', 'Natural language processing specialist', '自然语言处理专家'),
    ('ผู้ประสานงานหรือผู้จัดการโครงการด้านปัญญาประดิษฐ์', 'Artificial intelligence project coordinator or manager', '人工智能项目协调员或经理'),
    ('ผู้ประสานงานหรือผู้จัดการโครงการวิเคราะห์ข้อมูล', 'Data analysis project coordinator or manager', '数据分析项目协调员或经理'),
    ('ที่ปรึกษาด้านปัญญาประดิษฐ์', 'Artificial intelligence consultant', '人工智能顾问'),
    ('วิศวกรข้อมูล', 'Data engineer', '数据工程师'),
]
SPECIALIZATIONS = [
    ('ด้านการพัฒนาซอฟต์แวร์', 'Software Development', '软件开发'),
    ('ด้านโครงสร้างพื้นฐานเทคโนโลยีสารสนเทศ', 'Information Technology Infrastructure', '信息技术基础设施'),
    ('ด้านสื่อประสมสำหรับการพัฒนาสื่อเชิงโต้ตอบเว็บและเกม', 'Multimedia for Interactive Media, Web and Game Development', '用于交互媒体、网页和游戏开发的多媒体'),
]
FIELD_NAMES = {
    'credits': ('หน่วยกิตรวม', 'Total credits', '总学分'),
    'specialized': ('หน่วยกิตหมวดวิชาเฉพาะ', 'Specialized-course credits', '专业课程学分'),
    'duration': ('ระยะเวลาการศึกษา', 'Study duration', '学制'),
    'co_op': ('หน่วยกิตสหกิจศึกษา', 'Cooperative education credits', '合作教育学分'),
    'careers': ('อาชีพหลังสำเร็จการศึกษา', 'Graduate careers', '毕业生职业'),
    'specializations': ('ความเชี่ยวชาญเฉพาะทาง', 'Specializations', '专业方向'),
    'semesters': ('ภาคการศึกษา', 'Semesters', '学期'),
    'unknown': ('ส่วนอื่นของคำถาม', 'Other requested information', '其他所问信息'),
    'age': ('อายุผู้สมัคร', 'Applicant age', '申请人年龄'),
    'prize': ('รางวัล', 'Prize', '奖项'),
    'date': ('วันที่', 'Date', '日期'),
    'name': ('ชื่อหัวหน้าทีม', 'Team leader name', '队长姓名'),
}
MONTHS = [
    ('มกราคม','January','一月'), ('กุมภาพันธ์','February','二月'),
    ('มีนาคม','March','三月'), ('เมษายน','April','四月'),
    ('พฤษภาคม','May','五月'), ('มิถุนายน','June','六月'),
    ('กรกฎาคม','July','七月'), ('สิงหาคม','August','八月'),
    ('กันยายน','September','九月'), ('ตุลาคม','October','十月'),
    ('พฤศจิกายน','November','十一月'), ('ธันวาคม','December','十二月'),
]


@dataclass(frozen=True)
class Fact:
    facet: str
    texts: tuple[str, str, str]
    document: str
    page: int | None
    value: str | None = None


def facts_for(evidence):
    facts = []
    for e in evidence:
        q, f = compact(e.quote), e.facet
        variants = []
        if f in ('credits', 'specialized', 'duration', 'co_op'):
            values = [e.value] if e.value is not None else []
            if f == 'co_op':
                values = list(dict.fromkeys(re.findall(r'(\d+)\(0[-–]45[-–]0\)', q)))
            for value in values:
                minimum = bool(re.search(r'(?:ไม่น้อยกว่า|อย่างน้อย)'+re.escape(value), q))
                units = ('ปี','years','年') if f == 'duration' else ('หน่วยกิต','credits','学分')
                prefixes = ('อย่างน้อย ', 'at least ', '至少 ') if minimum else ('','','')
                texts = tuple(f'{FIELD_NAMES[f][i]}: {prefixes[i]}{value} {units[i]}' for i in range(3))
                variants.append((texts, value))
        elif f in ('careers', 'specializations'):
            item = re.sub(r'^\(\d+\)', '', q)
            for row in CAREERS if f == 'careers' else SPECIALIZATIONS:
                expected = compact(row[0])
                if f == 'specializations':
                    # The optional parenthesized English name must also match.
                    matched = item in (expected, expected+'('+compact(row[1])+')')
                else:
                    matched = item == expected
                if matched:
                    variants.append((row, None))
                    break
        elif f == 'semesters':
            regular = re.search(r'แบ่งออกเป็น(\d+)ภาคการศึกษาปกติ', q)
            weeks = re.search(r'(ไม่น้อยกว่า|อย่างน้อย)?(\d+)สัปดาห์', q)
            if regular:
                n = regular[1]
                variants.append(((f'ปีการศึกษามี {n} ภาคการศึกษาปกติ', f'The academic year has {n} regular semesters', f'每学年有 {n} 个常规学期'), None))
                if weeks:
                    n = weeks[2]; minimum = bool(weeks[1])
                    variants.append(((f'ภาคการศึกษาปกติ: {"อย่างน้อย " if minimum else ""}{n} สัปดาห์', f'Regular semester: {"at least " if minimum else ""}{n} weeks', f'常规学期：{"至少 " if minimum else ""}{n} 周'), None))
            else:
                months = [row for row in MONTHS if row[0] in q]
                semester = re.search(r'ภาคการศึกษาที่(\d+)', q)
                summer = 'ภาคฤดูร้อน' in q
                if months and (semester or summer):
                    names = ('ภาคฤดูร้อน','Summer semester','暑期学期') if summer else (f'ภาคการศึกษาที่ {semester[1]}',f'Semester {semester[1]}',f'第 {semester[1]} 学期')
                    variants.append((tuple(names[i]+': '+'–'.join(m[i] for m in months) for i in range(3)),None))
                    if weeks:
                        n=weeks[2]
                        variants.append(((f'ภาคฤดูร้อน: {n} สัปดาห์', f'Summer semester: {n} weeks', f'暑期学期：{n} 周'),None))
        if not variants:
            raise LanguageRenderingError('A verified translation is not available for this evidence yet. No unvalidated translation was returned.')
        for texts, value in variants:
            facts.append(Fact(f, tuple(texts), e.result.chunk.document, e.result.chunk.page, value))
    return list(dict.fromkeys(facts))


def present(evidence, facets, language):
    if not evidence:
        return refusal(language)
    if language not in ('th', 'en', 'zh'):
        raise LanguageRenderingError('Verified grounded answers currently support Thai, English and Chinese. This language needs a reviewed translation renderer.')
    try:
        facts = facts_for(evidence)
    except LanguageRenderingError:
        if language == 'th':
            return render_answer(evidence, facets)
        raise
    def section(index):
        lines = []
        for facet in facets:
            records = [r for r in facts if r.facet == facet]
            if not records: continue
            lines.append(FIELD_NAMES[facet][index]+':')
            values = {r.value for r in records if r.value is not None}
            if len(values) > 1:
                lines.append(('พบข้อมูลขัดแย้งกัน — ไม่เลือกค่าใดค่าหนึ่ง:', 'Conflicting evidence — no single value selected:', '文档信息存在冲突，不选择其中任何一个值：')[index])
            groups = {}
            for r in records:
                groups.setdefault(r.texts, []).append(r)
            for number, (texts, sources) in enumerate(groups.items(), 1):
                citations = []
                for r in sources:
                    page = (' หน้า ', ' page ', ' 页 ')[index]+str(r.page) if r.page is not None else ''
                    citations.append(r.document+page)
                bullet = f'({number})' if facet in ('careers', 'specializations') else '-'
                lines.append(f'{bullet} {texts[index]} ({"; ".join(dict.fromkeys(citations))})')
        missing = [FIELD_NAMES[f][index] for f in facets if not any(r.facet == f for r in facts)]
        if missing:
            lines = [('ข้อมูลที่พบ:', 'Information found:', '已找到的信息：')[index], *lines,
                     ('ข้อมูลที่ไม่พบ:', 'Information not found:', '未找到的信息：')[index], *missing]
        return '\n'.join(lines)
    thai = section(0)
    if language == 'th': return thai
    index = 1 if language == 'en' else 2
    return f'ภาษาไทย:\n{thai}\n\n{TRANSLATIONS[language][0]}:\n{section(index)}'
