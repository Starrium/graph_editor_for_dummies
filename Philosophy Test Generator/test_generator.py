import re, random
from reportlab.lib.pagesizes import A1
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import mm

# =============================
# CONFIG
# =============================
INPUT_FILE = "mcq_practices.txt"        # Input text file
OUTPUT_FILE = "triethoc_exam_fixed.pdf" # Output PDF
FONT_FILE = r"SVN-Arial 2.ttf"               # Keep your Arial font
NUM_QUESTIONS = 70                 # Number of random questions
# =============================

# Register your existing Arial font (you already fixed this)
pdfmetrics.registerFont(TTFont('VNFont', FONT_FILE))
font_name = 'VNFont'

# Read file
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Remove junk page breaks
junk_patterns = [
    r"Downloaded by .*",
    r"lOMoARcPSD\|\d+",
]
for jp in junk_patterns:
    text = re.sub(jp, "", text)
text = re.sub(r"\n{2,}", "\n", text).strip()

# ============================================
# FIXED REGEX: detect both "1." and "Câu 1:"
# ============================================
q_pattern = re.compile(r"(?m)^(?:Câu\s*)?(\d{1,3})[\.:]\s*")
matches = list(q_pattern.finditer(text))

questions = []
for i, m in enumerate(matches):
    qnum = int(m.group(1))
    start = m.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    qtext = text[start:end].strip()

    # ============================================
    # FIX: split jammed options like "B.C." → "B.\nC."
    # ============================================
    qtext = re.sub(r"([A-D])\.(?=[A-D]\.)", r"\1.\n", qtext)
    qtext = re.sub(r"([A-D])\.(?=[A-D]\s)", r"\1.\n", qtext)

    questions.append((qnum, qtext))

print(f"✅ Detected {len(questions)} questions in file.")

# Select random n questions
if len(questions) < NUM_QUESTIONS:
    raise ValueError("Not enough questions to build full test!")

seed = random.randint(1, 100)
random.seed(seed)
selected = random.sample(questions, NUM_QUESTIONS)

# ============================================
# Generate PDF
# ============================================
c = canvas.Canvas(OUTPUT_FILE, pagesize=A1)
width, height = A1
left = 20 * mm
top = height - 20 * mm
y = top

c.setFont(font_name, 16)
c.drawString(left, y, "ĐỀ LUYỆN TẬP TRIẾT HỌC")
c.setFont(font_name, 11)
c.drawString(left, y - 14, f"Renumbered 1–{NUM_QUESTIONS} (original question numbers in parentheses) Test seed = {seed}")
y -= 36

c.setFont(font_name, 12)
for idx, (orig_num, qtext) in enumerate(selected, start=1):
    if y < 40 * mm:
        c.showPage()
        c.setFont(font_name, 12)
        y = top

    heading = f"{idx}. (Q{orig_num})"
    c.setFont(font_name, 12)
    c.drawString(left, y, heading)
    y -= 14

    # Write question text, line by line
    for line in qtext.splitlines():
        line = line.strip()
        if not line:
            continue
        if y < 25 * mm:
            c.showPage()
            c.setFont(font_name, 12)
            y = top
        c.drawString(left + 6 * mm, y, line)
        y -= 12

    y -= 6

c.save()
print(f"🎉 PDF generated successfully: {OUTPUT_FILE}")
