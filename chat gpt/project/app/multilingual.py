"""Local query expansion: original names, codes, numbers and dates are preserved."""
import re
import unicodedata

ALIASES = (
    (("artificial intelligence technology", "人工智能", "ait"), "เทคโนโลยีปัญญาประดิษฐ์ AIT"),
    (("it2565", "information technology", "信息技术"), "เทคโนโลยีสารสนเทศ IT2565"),
    (("dsba", "data science", "数据科学"), "วิทยาการข้อมูล DSBA"),
    (("credits", "credit hours", "学分"), "หน่วยกิต"),
    (("duration", "how many years", "学制", "几年"), "ระยะเวลาการศึกษา กี่ปี"),
    (("careers", "career", "jobs", "occupations", "职业", "就业"), "อาชีพหลังสำเร็จการศึกษา"),
    (("semester", "semesters", "学期", "เทอม"), "ภาคการศึกษา"),
    (("specializations", "specialisation", "专业方向", "专业领域"), "ความเชี่ยวชาญเฉพาะทาง"),
    (("specialized", "specialised", "专业课程", "专业课"), "หมวดวิชาเฉพาะ"),
    (("cooperative", "co-op", "internship", "实习", "合作教育"), "สหกิจศึกษา"),
)

def normalize_query(question: str) -> str:
    original = unicodedata.normalize("NFC", question).strip()
    lowered = original.casefold()
    additions = []
    for aliases, thai in ALIASES:
        if any((re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", lowered)
                if alias.isascii() else alias in lowered) for alias in aliases):
            additions.append(thai)
    return " ".join([original, *dict.fromkeys(additions)])
