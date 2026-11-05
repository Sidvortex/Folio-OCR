"""Structured synthetic content: resumes (Indian, US, UK styles) and general
pages (articles, tables, math notes, letters/forms, Indic-script pages).

Content is generated *before* layout so the same document can be drawn by
several layouts; everything visible is decided here (casing, bullets, dashes),
because labels must be exactly the visible text (no CSS text-transform).
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from faker import Faker

from .vocab import (CERTS, CITIES, COMMON_WORDS, COURSES, DEGREES, DOMAINS, FIELDS, FIRST_NAMES,
                    LANGS, LAST_NAMES, LEVELS, MONTHS, SCHOOLS, SKILL_LABELS, SOFT_SKILLS,
                    TECH, TITLES, UNIVERSITIES, ResumeTextSampler, smart_quotes)


def pick(rng: random.Random, seq: Sequence):
    return seq[rng.randrange(len(seq))]


_FAKERS = {}


def faker(locale: str, rng: random.Random) -> Faker:
    f = _FAKERS.get(locale)
    if f is None:
        f = _FAKERS[locale] = Faker(locale)
    f.seed_instance(rng.randrange(1 << 30))
    return f


# ------------------------------------------------------------------ tables
@dataclass
class Cell:
    text: str
    colspan: int = 1
    rowspan: int = 1
    header: bool = False
    bold: bool = False


@dataclass
class Table:
    rows: List[List[Cell]]
    caption: str = ""
    footnote: str = ""
    numeric: bool = False


# ----------------------------------------------------------------- resumes
@dataclass
class Entry:
    left: str                 # role / degree / project name
    right: str = ""           # dates
    sub_left: str = ""        # company / institute / tech stack
    sub_right: str = ""       # location
    bullets: List[str] = field(default_factory=list)
    note: str = ""            # CGPA line, project link ...


@dataclass
class Section:
    title: str
    kind: str                 # entries | bullets | skills | paragraph | table | inline
    entries: List[Entry] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    paragraph: str = ""
    table: Optional[Table] = None
    side: bool = False        # suitable for a sidebar


@dataclass
class ResumeDoc:
    style: str
    name: str
    headline: str
    contacts: List[Tuple[str, str]]     # (kind, value): email/phone/location/url
    sections: List[Section]
    personal: List[Tuple[str, str]] = field(default_factory=list)
    declaration: str = ""
    place: str = ""
    date: str = ""
    photo: bool = False
    title_line: str = ""                 # "CURRICULUM VITAE" / "RESUME"
    footer: str = ""


class ResumeGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.s = ResumeTextSampler(rng)

    # ---------------------------------------------------------- atoms
    def dash(self) -> str:
        return pick(self.rng, [" – ", " – ", " – ", " - ", " — ", " to "])

    ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def month_year(self, y: int, fmt: int) -> str:
        m = self.rng.randrange(12)
        ab = "Sept" if m == 8 and self.rng.random() < 0.3 else self.ABBR[m]
        if fmt == 0:
            return f"{ab} {y}"
        if fmt == 1:
            return f"{MONTHS[m]} {y}"
        if fmt == 2:
            return f"{m + 1:02d}/{y}"
        if fmt == 3:
            return f"{ab}. {y}" if len(MONTHS[m]) > len(ab) else f"{ab} {y}"
        return str(y)

    def span(self, start: int, years: int, current: bool, fmt: int) -> str:
        a = self.month_year(start, fmt)
        if current:
            b = pick(self.rng, ["Present", "Present", "Current", "Now"])
        else:
            b = self.month_year(start + years, fmt)
        return a + self.dash() + b

    def person(self, style: str) -> Tuple[str, str]:
        loc = {"in": "en_IN", "us": "en_US", "uk": "en_GB"}[style]
        if style == "in" and self.rng.random() < 0.6:
            first, last = pick(self.rng, FIRST_NAMES[:62]), pick(self.rng, LAST_NAMES[:75])
        else:
            f = faker(loc, self.rng)
            first, last = f.first_name(), f.last_name()
        if self.rng.random() < 0.12:
            first = f"{first} {pick(self.rng, string.ascii_uppercase)}."
        return first, last

    def handle(self, first: str, last: str) -> str:
        f = "".join(c for c in first.split()[0].lower() if c.isalpha())
        l = "".join(c for c in last.lower() if c.isalpha())
        num = str(self.rng.randrange(1, 999)) if self.rng.random() < 0.4 else ""
        return pick(self.rng, [f"{f}{l}", f"{f}.{l}", f"{f}-{l}", f"{f}_{l}", f"{f[0]}{l}", f"{l}{f[:3]}"]) + num

    def contacts(self, style: str, first: str, last: str) -> List[Tuple[str, str]]:
        h = self.handle(first, last)
        dom = pick(self.rng, ["gmail.com"] * 5 + ["outlook.com", "yahoo.com", "hotmail.com", "proton.me",
                                                  "icloud.com", "yahoo.co.in" if style == "in" else "mail.com"])
        out = [("email", f"{h}@{dom}")]
        d = lambda n: "".join(str(self.rng.randrange(10)) for _ in range(n))
        if style == "in":
            first_d = str(self.rng.choice([6, 7, 8, 9]))
            phone = pick(self.rng, [f"+91 {first_d}{d(4)} {d(5)}", f"+91-{first_d}{d(9)}", f"+91 {first_d}{d(9)}",
                                    f"{first_d}{d(9)}", f"+91 {first_d}{d(2)}-{d(3)}-{d(4)}", f"(+91) {first_d}{d(9)}"])
        elif style == "us":
            phone = pick(self.rng, [f"({d(3)}) {d(3)}-{d(4)}", f"+1 {d(3)}-{d(3)}-{d(4)}", f"{d(3)}-{d(3)}-{d(4)}",
                                    f"+1 ({d(3)}) {d(3)}-{d(4)}", f"{d(3)}.{d(3)}.{d(4)}"])
        else:
            phone = pick(self.rng, [f"+44 7{d(3)} {d(6)}", f"07{d(3)} {d(6)}", f"+44 (0)7{d(3)} {d(3)} {d(3)}"])
        out.append(("phone", phone))
        if style == "in":
            city = pick(self.rng, CITIES[:22])
            loc = f"{city}, {pick(self.rng, ['India', 'Uttar Pradesh', 'Karnataka', 'Telangana', 'Maharashtra', 'Tamil Nadu', 'Delhi', 'Haryana'])}"
        else:
            f = faker("en_US" if style == "us" else "en_GB", self.rng)
            loc = f"{f.city()}, {f.state_abbr() if style == 'us' else 'UK'}"
        out.append(("location", loc))
        links = [f"linkedin.com/in/{h}", f"github.com/{h}", f"{h}.github.io", f"www.{h}.dev",
                 f"kaggle.com/{h}", f"leetcode.com/u/{h}", f"behance.net/{h}"]
        for u in self.rng.sample(links[:4] if self.rng.random() < 0.7 else links, self.rng.randint(0, 3)):
            if self.rng.random() < 0.12:
                u = "https://" + u
            out.append(("url", u))
        return out

    def bullet(self) -> str:
        b = self.s.bullet()
        return smart_quotes(b) if self.rng.random() < 0.2 else b

    def company(self, style: str) -> str:
        if style != "in" and self.rng.random() < 0.5:
            return faker("en_US", self.rng).company()
        return self.s.company()

    def place(self, style: str) -> str:
        if self.rng.random() < 0.15:
            return pick(self.rng, ["Remote", "Hybrid", "Remote (India)", "On-site"])
        if style == "in":
            return pick(self.rng, CITIES[:22]) + (", India" if self.rng.random() < 0.3 else "")
        f = faker("en_US" if style == "us" else "en_GB", self.rng)
        return f"{f.city()}, {f.state_abbr()}" if style == "us" else f.city()

    # ------------------------------------------------------- sections
    def experience(self, style: str, n: int, start_year: int, date_fmt: int, intern: bool) -> List[Entry]:
        out, y = [], start_year
        for i in range(n):
            years = self.rng.randint(0, 3) if not intern else 0
            current = i == 0 and not intern and self.rng.random() < 0.6
            title = pick(self.rng, TITLES) if not intern else pick(self.rng, [t for t in TITLES if "Intern" in t or "Trainee" in t] or TITLES)
            e = Entry(left=title, right=self.span(y - years, max(1, years), current, date_fmt),
                      sub_left=self.company(style), sub_right=self.place(style) if self.rng.random() < 0.7 else "",
                      bullets=[self.bullet() for _ in range(self.rng.randint(2, 5 if not intern else 3))])
            out.append(e)
            y -= years + 1
        return out

    def education(self, style: str, fresher: bool, grad_year: int, date_fmt: int) -> List[Entry]:
        deg = pick(self.rng, DEGREES)
        fld = pick(self.rng, FIELDS)
        uni = pick(self.rng, UNIVERSITIES[:46] if style == "in" else UNIVERSITIES[44:] + UNIVERSITIES[:4])
        if style == "in":
            score = pick(self.rng, [f"CGPA: {self.rng.uniform(6.5, 9.8):.2f}/10", f"CGPA: {self.rng.uniform(6.5, 9.8):.2f}",
                                    f"Percentage: {self.rng.uniform(62, 95):.1f}%", f"SGPA: {self.rng.uniform(7, 9.9):.2f}"])
        else:
            score = pick(self.rng, [f"GPA: {self.rng.uniform(3.0, 4.0):.2f}/4.0", f"GPA {self.rng.uniform(3.0, 4.0):.1f}",
                                    "First Class Honours", "Upper Second-Class Honours (2:1)", "Magna Cum Laude", ""])
        span = self.span(grad_year - 4, 4, fresher and self.rng.random() < 0.5, date_fmt) if self.rng.random() < 0.8 \
            else pick(self.rng, ["Expected ", "Graduating "]) + str(grad_year) if fresher else str(grad_year)
        ent = [Entry(left=f"{deg} in {fld}" if self.rng.random() < 0.6 else f"{deg}, {fld}", right=span,
                     sub_left=uni, sub_right=self.place(style) if self.rng.random() < 0.5 else "",
                     note=score)]
        if fresher and self.rng.random() < 0.6 and style == "in":
            for lvl, yr in (("Class XII", grad_year - 4), ("Class X", grad_year - 6)):
                board = pick(self.rng, ["CBSE", "ICSE", "ISC", "State Board", "UP Board", "BIEAP"])
                ent.append(Entry(left=f"{lvl} ({board})", right=str(yr), sub_left=pick(self.rng, SCHOOLS),
                                 note=f"Percentage: {self.rng.uniform(70, 98):.1f}%"))
        if self.rng.random() < 0.3 and fresher:
            ent[0].bullets = ["Relevant Coursework: " + ", ".join(self.rng.sample(COURSES, self.rng.randint(3, 6)))]
        return ent

    def skills(self, sidebar: bool) -> List[Tuple[str, str]]:
        labels = self.rng.sample(SKILL_LABELS, self.rng.randint(2, 5))
        sep = pick(self.rng, [", ", ", ", " | ", " • ", " · ", "; "])
        out = []
        for lab in labels:
            pool = SOFT_SKILLS if lab == "Soft Skills" else TECH
            out.append((lab, sep.join(self.rng.sample(pool, self.rng.randint(3, 6 if sidebar else 9)))))
        return out

    def projects(self, n: int, date_fmt: int) -> List[Entry]:
        out = []
        for _ in range(n):
            name = pick(self.rng, ["Resume Parser", "Smart Attendance System", "Crop Disease Detector",
                                   "Stock Price Predictor", "Campus Connect", "Expense Tracker", "Landslide Early Warning System",
                                   "Code Reader", "Media Bias Estimator", "WiFi Room Mapper", "Hinglish Speech Recognizer",
                                   "Desk Robot", "Movie Recommender", "Fake News Classifier", "Traffic Sign Recognition",
                                   "Personal Portfolio", "Realtime Chat App", "E-commerce Backend", "PDF Q&A Bot",
                                   "Diagnostic Prediction Suite", "Candidate Ranking App", "Vision Hub", "Keyword Spotter"])
            tech = ", ".join(self.rng.sample(TECH, self.rng.randint(2, 4)))
            right = self.month_year(self.rng.randint(2019, 2026), date_fmt) if self.rng.random() < 0.5 else ""
            link = f"github.com/{self.s.handle()}/{name.lower().replace(' ', '-')}" if self.rng.random() < 0.35 else ""
            out.append(Entry(left=name, right=right, sub_left=tech, note=link,
                             bullets=[self.bullet() for _ in range(self.rng.randint(1, 3))]))
        return out

    def summary(self) -> str:
        parts = [self.s.summary()]
        if self.rng.random() < 0.6:
            parts.append(self.s.summary())
        return " ".join(parts)

    def edu_table(self, grad_year: int) -> Table:
        hdr = [Cell(h, header=True) for h in pick(self.rng, [
            ["Qualification", "Institute", "Board/University", "Year", "CGPA/%"],
            ["Degree", "Institution", "Year of Passing", "Score"],
            ["Examination", "School/College", "Board", "Year", "Percentage"]])]
        rows = [hdr]
        quals = [(pick(self.rng, ["B.Tech (CSE)", "B.Tech (IT)", "B.E. (ECE)", "BCA", "B.Sc. (CS)"]), grad_year, f"{self.rng.uniform(6.5, 9.6):.2f} CGPA"),
                 ("Class XII", grad_year - 4, f"{self.rng.uniform(65, 97):.1f}%"),
                 ("Class X", grad_year - 6, f"{self.rng.uniform(70, 98):.1f}%")]
        for q, y, sc in quals:
            vals = {"Qualification": q, "Degree": q, "Examination": q,
                    "Institute": pick(self.rng, UNIVERSITIES[:46]) if q not in ("Class XII", "Class X") else pick(self.rng, SCHOOLS),
                    "Institution": pick(self.rng, UNIVERSITIES[:46]) if q not in ("Class XII", "Class X") else pick(self.rng, SCHOOLS),
                    "School/College": pick(self.rng, SCHOOLS), "Board/University": pick(self.rng, ["AKTU", "VTU", "Anna University", "CBSE", "ICSE", "JNTUH", "Mumbai University"]),
                    "Board": pick(self.rng, ["CBSE", "ICSE", "State Board"]), "Year": str(y), "Year of Passing": str(y),
                    "CGPA/%": sc, "Score": sc, "Percentage": sc}
            rows.append([Cell(vals[c.text]) for c in hdr])
        return Table(rows)

    def personal(self, first: str, last: str) -> List[Tuple[str, str]]:
        f = faker("en_IN", self.rng)
        out = [("Date of Birth", f"{self.rng.randint(1, 28):02d}/{self.rng.randint(1, 12):02d}/{self.rng.randint(1996, 2006)}"),
               ("Gender", pick(self.rng, ["Male", "Female"])), ("Nationality", "Indian"),
               ("Languages Known", ", ".join(self.rng.sample(LANGS[:11], self.rng.randint(2, 4)))),
               ("Father's Name", f"{f.first_name_male()} {last}"),
               ("Address", f.address().replace("\n", ", "))]
        self.rng.shuffle(out)
        return out[: self.rng.randint(3, 6)]

    # ------------------------------------------------------------ main
    def generate(self) -> ResumeDoc:
        rng = self.rng
        style = rng.choices(["in", "us", "uk"], weights=[0.55, 0.3, 0.15])[0]
        first, last = self.person(style)
        name = f"{first} {last}"
        name = name.upper() if rng.random() < 0.4 else name
        fresher = rng.random() < 0.45
        grad = rng.randint(2022, 2028) if fresher else rng.randint(2008, 2021)
        date_fmt = rng.choices([0, 1, 2, 3, 4], weights=[5, 2, 2, 1, 1])[0]
        headline = pick(rng, TITLES) if not fresher else pick(rng, [
            f"{pick(rng, DEGREES)} {pick(rng, ['Student', 'Undergraduate', 'Graduate'])}", "Aspiring " + pick(rng, TITLES),
            pick(rng, TITLES)])
        if rng.random() < 0.25:
            headline = ""
        caps = rng.random() < 0.55        # section titles in capitals
        def T(t):
            return t.upper() if caps else t

        secs: List[Section] = []
        if rng.random() < 0.75:
            secs.append(Section(T(pick(rng, ["Summary", "Professional Summary", "Profile", "About Me"]) if not fresher
                                  else pick(rng, ["Objective", "Career Objective", "Summary"])), "paragraph", paragraph=self.summary()))
        n_jobs = rng.randint(0, 2) if fresher else rng.randint(1, 4)
        if n_jobs:
            secs.append(Section(T(pick(rng, ["Internships", "Experience"]) if fresher else pick(rng, ["Experience", "Work Experience", "Professional Experience"])),
                                "entries", entries=self.experience(style, n_jobs, rng.randint(2022, 2026), date_fmt, fresher)))
        if style == "in" and rng.random() < 0.3:
            secs.append(Section(T(pick(rng, ["Academic Qualifications", "Education", "Educational Qualification"])), "table", table=self.edu_table(grad)))
        else:
            secs.append(Section(T("Education"), "entries", entries=self.education(style, fresher, grad, date_fmt)))
        if rng.random() < 0.85:
            secs.append(Section(T(pick(rng, ["Skills", "Technical Skills", "Core Competencies", "Tools & Technologies"])), "skills",
                                pairs=self.skills(False), side=True))
        n_proj = rng.randint(1, 4) if fresher else rng.randint(0, 2)
        if n_proj:
            secs.append(Section(T(pick(rng, ["Projects", "Academic Projects", "Personal Projects", "Key Projects"])), "entries",
                                entries=self.projects(n_proj, date_fmt)))
        if rng.random() < 0.55:
            secs.append(Section(T(pick(rng, ["Certifications", "Courses & Certifications", "Licenses & Certifications"])), "bullets",
                                items=rng.sample(CERTS, rng.randint(2, 4)), side=True))
        if rng.random() < 0.5:
            ach = [self.s.cert() if rng.random() < 0.5 else self.bullet() for _ in range(rng.randint(2, 4))]
            secs.append(Section(T(pick(rng, ["Achievements", "Awards & Achievements", "Honors and Awards"])), "bullets", items=ach))
        if rng.random() < 0.4:
            secs.append(Section(T(pick(rng, ["Positions of Responsibility", "Leadership", "Extracurricular Activities", "Volunteering"])),
                                "bullets", items=[self.bullet() for _ in range(rng.randint(1, 3))]))
        if rng.random() < 0.45:
            langs = rng.sample(LANGS, rng.randint(2, 4))
            if rng.random() < 0.6:
                langs = [f"{l} ({pick(rng, LEVELS).strip('()')})" for l in langs]
            secs.append(Section(T("Languages"), "inline", items=langs, side=True))
        if rng.random() < 0.35:
            secs.append(Section(T(pick(rng, ["Interests", "Hobbies"])), "inline",
                                items=rng.sample(["Chess", "Reading", "Badminton", "Photography", "Cricket", "Hiking",
                                                  "Music", "Open-source", "Robotics", "Blogging", "Travel", "Sketching"], rng.randint(2, 4)),
                                side=True))
        doc = ResumeDoc(style=style, name=name, headline=headline, contacts=self.contacts(style, first, last), sections=secs)
        if style == "in" and rng.random() < 0.35:
            doc.personal = self.personal(first, last)
            doc.title_line = pick(rng, ["CURRICULUM VITAE", "RESUME", "Curriculum Vitae", "BIO-DATA"]) if rng.random() < 0.6 else ""
            if rng.random() < 0.7:
                doc.declaration = pick(rng, [
                    "I hereby declare that the information furnished above is true to the best of my knowledge and belief.",
                    "I hereby declare that all the details mentioned above are true and correct to the best of my knowledge.",
                    "I do hereby declare that the above information is true to the best of my knowledge."])
                doc.place = pick(rng, CITIES[:22])
                doc.date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2023, 2026)}"
        doc.photo = rng.random() < (0.3 if style == "in" else 0.08)
        if rng.random() < 0.08:
            doc.footer = pick(rng, ["Page 1 of 1", "Page 1", f"{name.title()} – Résumé", "References available upon request"])
        return doc


# ------------------------------------------------------------ general pages
LATEX_FORMULAS = [
    r"E = mc^2", r"a^2 + b^2 = c^2", r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
    r"\int_{0}^{\infty} e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}",
    r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
    r"\mathbf{y} = W\mathbf{x} + \mathbf{b}", r"P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}",
    r"\hat{\beta} = (X^\top X)^{-1} X^\top y", r"\mathcal{L} = -\sum_{c=1}^{C} y_c \log \hat{y}_c",
    r"x_{t+1} = x_t - \eta \nabla f(x_t)", r"\sigma(z) = \frac{1}{1 + e^{-z}}",
    r"\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V",
    r"\det(A - \lambda I) = 0", r"\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e",
    r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}", r"\binom{n}{k} = \frac{n!}{k!\,(n-k)!}",
    r"\|x\|_2 = \sqrt{\sum_{i=1}^{d} x_i^2}", r"\int_a^b f'(x)\,dx = f(b) - f(a)",
    r"R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}",
    r"A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}", r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
    r"e^{i\pi} + 1 = 0", r"\mathbb{E}[X] = \sum_{x} x\, p(x)", r"\mathrm{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2",
    r"F_1 = \frac{2 \cdot \mathrm{precision} \cdot \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}",
    r"\theta^{*} = \arg\min_{\theta} \sum_{i=1}^{N} \ell(f_\theta(x_i), y_i) + \lambda \|\theta\|_2^2",
    r"Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]",
    r"\phi(x) = \sum_{k=0}^{\infty} \frac{x^k}{k!}", r"\oint_C \mathbf{F} \cdot d\mathbf{r} = 0",
    r"y'' + 2\zeta\omega_n y' + \omega_n^2 y = 0",
]
INLINE_MATH = [r"x_i", r"\alpha = 0.05", r"O(n \log n)", r"p < 0.01", r"\mathbb{R}^d", r"k", r"\theta", r"n = 128",
               r"\lambda", r"f(x)", r"\mu \pm \sigma", r"t \in [0, T]", r"\Delta t", r"x^2", r"\epsilon"]

HINDI_WORDS = ("और के में है यह से को पर भारत सरकार विकास शिक्षा छात्र विश्वविद्यालय परियोजना प्रणाली तकनीक कार्य "
               "अनुभव समाज देश लोग समय वर्ष नया पुराना महत्वपूर्ण जानकारी सेवा योजना उद्योग विज्ञान अनुसंधान प्रबंधन "
               "विभाग अधिकारी नागरिक क्षेत्र राज्य जिला गाँव शहर जल स्वास्थ्य कृषि उत्पादन बाजार व्यापार आर्थिक "
               "सामाजिक सांस्कृतिक इतिहास भाषा साहित्य कला संगीत खेल प्रतियोगिता परिणाम रिपोर्ट अध्ययन").split()
TAMIL_WORDS = ("மற்றும் இந்த ஒரு தமிழ் அரசு கல்வி மாணவர் திட்டம் தொழில்நுட்பம் வேலை அனுபவம் நாடு மக்கள் "
               "நேரம் ஆண்டு புதிய முக்கியமான தகவல் சேவை தொழில் அறிவியல் ஆராய்ச்சி மேலாண்மை துறை அலுவலர் "
               "மாநிலம் மாவட்டம் கிராமம் நகரம் தண்ணீர் சுகாதாரம் விவசாயம் உற்பத்தி சந்தை வணிகம் பொருளாதார "
               "சமூக கலாச்சார வரலாறு மொழி இலக்கியம் கலை இசை விளையாட்டு போட்டி முடிவு அறிக்கை படிப்பு").split()


@dataclass
class GenBlock:
    """A general-page block before layout."""
    cat: str                  # schema category
    text: str = ""            # plain text (may contain **bold**/*italic*)
    level: int = 0            # heading level for titles
    items: List[str] = field(default_factory=list)   # list items
    ordered: bool = False
    latex: str = ""           # formula
    table: Optional[Table] = None
    inline_math: List[str] = field(default_factory=list)


@dataclass
class GeneralDoc:
    kind: str
    header: str = ""
    footer: str = ""
    page_no: str = ""
    title: str = ""
    subtitle: str = ""
    columns: int = 1
    blocks: List[GenBlock] = field(default_factory=list)
    script: str = "latn"
    footnotes: List[str] = field(default_factory=list)


class GeneralGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def sentence(self, words: Tuple[int, int] = (8, 22)) -> str:
        f = faker("en_US", self.rng)
        s = f.sentence(nb_words=self.rng.randint(*words), variable_nb_words=False)
        if self.rng.random() < 0.25:
            ws = s[:-1].split()
            ws.insert(self.rng.randrange(len(ws)), pick(self.rng, TECH + DOMAINS + COMMON_WORDS))
            s = " ".join(ws) + "."
        return s

    def paragraph(self, n: Tuple[int, int] = (3, 7), math: float = 0.0) -> Tuple[str, List[str]]:
        sents = [self.sentence() for _ in range(self.rng.randint(*n))]
        inl = []
        if math and self.rng.random() < math:
            k = self.rng.randrange(len(sents))
            m = pick(self.rng, INLINE_MATH)
            inl.append(m)
            ws = sents[k].split()
            ws.insert(self.rng.randrange(1, len(ws)), "\u0000")
            sents[k] = " ".join(ws)
        if self.rng.random() < 0.15:
            k = self.rng.randrange(len(sents))
            ws = sents[k].split()
            i = self.rng.randrange(len(ws))
            ws[i] = ("**" + ws[i] + "**") if self.rng.random() < 0.5 else ("*" + ws[i] + "*")
            sents[k] = " ".join(ws)
        return " ".join(sents), inl

    def number_table(self, rows: int, cols: int, merged: bool) -> Table:
        rng = self.rng
        year0 = rng.randint(2015, 2024)
        kind = rng.randrange(3)
        if kind == 0:
            labels = rng.sample(["Revenue", "Cost of revenue", "Gross profit", "Operating expenses", "EBITDA", "Net income",
                                 "Total assets", "Cash and equivalents", "Employees", "R&D expense", "Capex", "Free cash flow"], rows)
            cols_t = [str(year0 + i) for i in range(cols)]
            def val():
                v = rng.uniform(-500, 9000)
                return f"({abs(v):,.1f})" if v < 0 and rng.random() < 0.5 else f"{v:,.1f}"
        elif kind == 1:
            labels = rng.sample(["Baseline", "Ours", "Ours (no RL)", "BERT-base", "ResNet-50", "ViT-B/16", "LSTM", "XGBoost",
                                 "Random Forest", "SVM", "CRNN", "Tesseract", "Ensemble", "Linear"], rows)
            cols_t = rng.sample(["Accuracy", "F1", "Precision", "Recall", "AUC", "CER", "WER", "Latency (ms)", "Params (M)"], cols)
            def val():
                return f"{rng.uniform(0, 100):.2f}" if rng.random() < 0.8 else f"{rng.uniform(0.1, 9):.3f}"
        else:
            labels = rng.sample(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], min(rows, 7))
            cols_t = [f"{9 + i}:00" for i in range(cols)]
            def val():
                return pick(rng, COURSES[:12] + ["Lab", "Break", "—", "Seminar", "Tutorial"])
        corner = pick(rng, ["", "Item", "Metric", "Model", "Day", "(in ₹ crore)", "(USD millions)"])
        header = [Cell(corner, header=True)] + [Cell(c, header=True) for c in cols_t]
        trs = []
        if merged and cols >= 4 and cols % 2 == 0 and kind != 2:
            top = [Cell("", header=True, rowspan=2)] + [Cell(g, colspan=2, header=True) for g in
                                                         rng.sample(["Validation", "Test", "FY24", "FY25", "Domestic", "Export", "Q1–Q2", "Q3–Q4"], cols // 2)]
            sub = [Cell(c, header=True) for c in cols_t]
            trs = [top, sub]
        else:
            trs = [header]
        for lab in labels:
            trs.append([Cell(lab)] + [Cell(val()) for _ in cols_t])
        return Table(trs, numeric=kind != 2)

    def generate(self, kind: str) -> GeneralDoc:
        rng = self.rng
        f = faker("en_US", rng)
        if kind == "article":
            doc = GeneralDoc("article", columns=rng.choice([1, 2, 2]),
                             header=f"{pick(rng, ['Journal of', 'Proceedings of the', 'Transactions on', 'Review of'])} "
                                    f"{pick(rng, ['Applied', 'Computational', 'Data', 'Intelligent', 'Geospatial'])} "
                                    f"{pick(rng, ['Systems', 'Learning', 'Science', 'Engineering'])}, Vol. {rng.randint(1, 60)}, No. {rng.randint(1, 12)}",
                             page_no=str(rng.randint(1, 400)))
            doc.title = f.catch_phrase().title() + ": " + pick(rng, ["A Study", "An Empirical Analysis", "Methods and Results",
                                                                    "A Survey", "Towards Robust Systems"])
            doc.subtitle = ", ".join(f.name() for _ in range(rng.randint(1, 4)))
            n_sec = rng.randint(2, 4)
            doc.blocks.append(GenBlock("text", "**Abstract.** " + self.paragraph((3, 5))[0]))
            for s in range(n_sec):
                doc.blocks.append(GenBlock("title", f"{s + 1}. {f.bs().title()}", level=2))
                for _ in range(rng.randint(1, 3)):
                    txt, inl = self.paragraph(math=0.3)
                    doc.blocks.append(GenBlock("text", txt, inline_math=inl))
                if rng.random() < 0.3:
                    doc.blocks.append(GenBlock("formula", latex=pick(rng, LATEX_FORMULAS)))
                if rng.random() < 0.2:
                    doc.blocks.append(GenBlock("text", items=[self.sentence((5, 12)) for _ in range(rng.randint(2, 4))],
                                               ordered=rng.random() < 0.4))
            if rng.random() < 0.6:
                doc.footnotes = [f"{i + 1} " + self.sentence((6, 14)) for i in range(rng.randint(1, 2))]
            return doc
        if kind == "table":
            doc = GeneralDoc("table", header=f.company() + " – " + pick(rng, ["Annual Report", "Quarterly Results", "Benchmark Report", "Timetable"]),
                             page_no=f"Page {rng.randint(1, 60)}" if rng.random() < 0.6 else str(rng.randint(1, 60)))
            doc.title = pick(rng, ["Financial Highlights", "Results Overview", "Experimental Results", "Summary of Findings",
                                   "Weekly Schedule", "Operating Metrics"])
            doc.blocks.append(GenBlock("text", self.paragraph((2, 4))[0]))
            for t in range(rng.randint(2, 4)):
                if t and rng.random() < 0.5:
                    doc.blocks.append(GenBlock("title", f"{t + 1}. {f.bs().title()}", level=2))
                tbl = self.number_table(rng.randint(3, 9), rng.choice([3, 4, 4, 5, 6]), merged=rng.random() < 0.4)
                tbl.caption = f"Table {t + 1}: " + self.sentence((5, 10))[:-1]
                if rng.random() < 0.4:
                    tbl.footnote = pick(rng, ["Note: ", "Source: ", "* "]) + self.sentence((6, 12))
                doc.blocks.append(GenBlock("table", table=tbl))
                if rng.random() < 0.6:
                    doc.blocks.append(GenBlock("text", self.paragraph((2, 4))[0]))
            return doc
        if kind == "math":
            doc = GeneralDoc("math", header=pick(rng, ["Lecture Notes", "Problem Set", "Course Handout"]) + f" {rng.randint(1, 14)}",
                             page_no=str(rng.randint(1, 30)))
            doc.title = pick(rng, ["Gradient Descent", "Probability Review", "Linear Regression", "Heat Equation",
                                   "Attention Mechanisms", "Eigenvalues", "Reinforcement Learning Basics"])
            for _ in range(rng.randint(3, 6)):
                txt, inl = self.paragraph((2, 4), math=0.8)
                doc.blocks.append(GenBlock("text", txt, inline_math=inl))
                doc.blocks.append(GenBlock("formula", latex=pick(rng, LATEX_FORMULAS)))
            return doc
        if kind == "letter":
            doc = GeneralDoc("letter", header=f.company() + " · " + f.address().replace("\n", ", "))
            doc.blocks.append(GenBlock("text", f.date(pattern=pick(rng, ["%d %B %Y", "%B %d, %Y", "%d/%m/%Y"]))))
            doc.blocks.append(GenBlock("text", f"{f.name()}\n{f.street_address()}\n{f.city()}, {f.postcode()}"))
            doc.blocks.append(GenBlock("text", f"**Subject:** {f.bs().capitalize()}"))
            doc.blocks.append(GenBlock("text", pick(rng, ["Dear Sir/Madam,", f"Dear {f.first_name()},", "To Whom It May Concern,"])))
            for _ in range(rng.randint(2, 4)):
                doc.blocks.append(GenBlock("text", self.paragraph((2, 5))[0]))
            doc.blocks.append(GenBlock("text", pick(rng, ["Sincerely,", "Yours faithfully,", "Best regards,"]) + "\n" + f.name()))
            doc.footer = f.phone_number() + " | " + f.email()
            return doc
        if kind == "indic":
            script = rng.choice(["deva", "taml", "beng"])
            doc = GeneralDoc("indic", script=script, page_no=str(rng.randint(1, 99)))
            words = {"deva": HINDI_WORDS, "taml": TAMIL_WORDS}.get(script)
            def isent(n):
                if script == "beng":
                    return faker("bn_BD", rng).sentence(nb_words=n)
                ws = [pick(rng, words) for _ in range(n)]
                return " ".join(ws) + ("।" if script == "deva" else ".")
            doc.title = isent(rng.randint(3, 6))[:-1]
            for _ in range(rng.randint(3, 6)):
                doc.blocks.append(GenBlock("text", " ".join(isent(rng.randint(6, 14)) for _ in range(rng.randint(2, 5)))))
                if rng.random() < 0.3:
                    loc = {"deva": "hi_IN", "taml": "ta_IN", "beng": "bn_BD"}[script]
                    doc.blocks.append(GenBlock("text", items=[faker(loc, rng).name() + " – " + faker(loc, rng).city()
                                                              for _ in range(rng.randint(2, 4))]))
            return doc
        raise ValueError(f"unknown general page kind {kind!r}")
