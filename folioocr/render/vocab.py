"""Resume vocabulary and line sampler, copied from ResumeOCR-Lite (resumeocr/corpus.py).

Original docstring:

Resume-style text sampler for synthetic training lines.

The goal is coverage, not realism of whole documents: every line type that
shows up on a resume (headings, contact strips, date ranges, degrees, bullet
points, skill lists, URLs, phone numbers), in the casings and typography that
Word/Docs/LaTeX produce, plus random words and random character strings so the
BiLSTM learns characters rather than memorising phrases.

Optionally mix in your own text with ``--corpus file.txt`` (one sentence or
paragraph per line), e.g. text extracted from public resume datasets.
"""
from __future__ import annotations

import random
import string
from typing import Callable, List, Optional, Sequence, Tuple

from ..textnorm import BULLET_GLYPHS

FIRST_NAMES = """Aarav Vivaan Aditya Arjun Sai Reyansh Krishna Ishaan Rohan Kabir Advait Ayaan
Atharv Dhruv Harsh Nikhil Varun Akash Deepak Manish Ravi Suresh Ramesh Siddharth Rahul Amit
Vikram Karthik Pranav Yash Tushar Gaurav Abhishek Ankit Saurabh Mohit Rajat Kunal Tanmay Om
Priya Ananya Diya Isha Kavya Meera Neha Pooja Riya Sneha Tanvi Aditi Shreya Aishwarya Nandini
Gayatri Lakshmi Divya Swati Anjali Pallavi Bhavya Sanya Kritika Muskan Simran Harshita Ira
James Emily Michael Sarah David Jessica Daniel Olivia Matthew Sophia Andrew Emma Joshua Ava
Christopher Mia Ryan Grace Ethan Chloe Nathan Hannah Benjamin Lily Samuel Zoe Lucas Ella Noah
Liam Mason Logan Jacob Aiden Elena Laura Maria Ana Carlos Diego Juan Luis Sofia Mateo Lucia
Wei Li Jun Hao Yuki Kenji Haruto Min-jun Seo-yeon Ji-woo Thanh Linh Minh Omar Fatima Aisha
Yusuf Zara Hassan Ali Leila Tariq Noor Mohammed Ibrahim Amara Kwame Chidi Ngozi Tunde""".split()

LAST_NAMES = """Sharma Verma Gupta Kumar Singh Patel Reddy Rao Nair Iyer Menon Das Bose Chatterjee
Banerjee Mukherjee Joshi Kulkarni Deshpande Mehta Shah Agarwal Jain Mishra Pandey Tiwari Yadav
Chauhan Rathore Naidu Pillai Kapoor Malhotra Khanna Bhatia Sethi Ahmed Khan Siddiqui Ansari
Fernandes D'Souza Ravada Srinivasan Subramanian Krishnan Venkatesh Bhattacharya Saxena Dubey
Choudhary Thakur Goswami Hegde Shetty Gowda Patil Pawar Jadhav Bhosale Chopra Arora Ahuja Grover
Smith Johnson Williams Brown Jones Garcia Miller Davis Wilson Anderson Taylor Thomas Moore
Martin Lee Clark Lewis Walker Hall Young King Wright Scott Green Baker Adams Nelson Carter
Mitchell Roberts Turner Phillips Campbell Parker Evans Edwards Collins Stewart Morris Murphy
Cook Rogers Morgan Cooper Peterson Reed Bailey Bell Kelly Howard Ward Cox Richardson Wood Watson
Brooks Bennett Gray Hughes Price Sanders Myers Long Ross Foster O'Brien McDonald MacLeod
Zhang Wang Liu Chen Yang Huang Zhao Wu Kim Park Choi Nguyen Tran Tanaka Sato Suzuki Takahashi
Hernandez Lopez Gonzalez Rodriguez Martinez Silva Santos Rossi Muller Schmidt Novak Kowalski
Okafor Mensah Haddad Rahman Hussain Qureshi""".split()

COMPANIES = """Google|Microsoft|Amazon|Amazon Web Services|Meta|Apple|Netflix|Adobe|Oracle|IBM|Intel
|NVIDIA|Qualcomm|Cisco Systems|Salesforce|SAP Labs India|VMware|ServiceNow|Atlassian|Uber|LinkedIn
|Infosys|Tata Consultancy Services|TCS|Wipro|HCLTech|Tech Mahindra|Accenture|Deloitte|Capgemini
|Cognizant|LTIMindtree|Mphasis|Persistent Systems|Zensar Technologies|Mu Sigma|Fractal Analytics
|Tiger Analytics|Flipkart|Paytm|Zomato|Swiggy|Razorpay|Freshworks|Zoho Corporation|PhonePe|Meesho
|CRED|Myntra|Nykaa|Ola|BYJU'S|Reliance Jio|Samsung R&D Institute|Goldman Sachs|JPMorgan Chase & Co.
|Morgan Stanley|American Express|Walmart Global Tech|Barclays|HSBC|Deutsche Bank|Bosch|Siemens
|Honeywell|Philips|Ernst & Young|KPMG|PwC|McKinsey & Company|Boston Consulting Group|ISRO|DRDO
|C-DAC|NIC|Indian Railways|Bharat Electronics Limited|Larsen & Toubro|Mahindra & Mahindra""".replace("\n", "").split("|")

COMPANY_PARTS = (
    "Blue Nimbus Quant Vertex Apex Bright Silver Red Pine Cedar Delta Nova Orbit Pixel Data Cloud "
    "Neural Logic Spark Prism Zen Tera Alpha Beta Insight Cognito Ever Green Swift Iron Bridge Peak "
    "Stack Byte Code Signal Harbor Maple Lotus Indus Ganga Saffron Kavach Sutra Vayu Agni".split())
COMPANY_SUFFIX = ["Analytics", "Labs", "Technologies", "Solutions", "Systems", "Software",
                  "Consulting", "Networks", "Robotics", "AI", "Digital", "Infotech", "Health",
                  "Fintech", "Energy", "Studios", "Dynamics", "Ventures"]
COMPANY_LEGAL = ["Pvt. Ltd.", "Private Limited", "Inc.", "LLC", "LLP", "Ltd.", "GmbH", "Corp.", "", "", ""]

UNIVERSITIES = """Indian Institute of Technology Delhi|IIT Bombay|IIT Madras|IIT Kanpur|IIT Kharagpur
|IIT Roorkee|IIT Guwahati|IIT Hyderabad|IIT (BHU) Varanasi|NIT Tiruchirappalli|NIT Warangal
|NIT Karnataka, Surathkal|NIT Calicut|BITS Pilani|BITS Pilani, Goa Campus|IIIT Hyderabad
|IIIT Bangalore|IIIT Delhi|Delhi Technological University|Netaji Subhas University of Technology
|Jadavpur University|Anna University|VIT Vellore|SRM Institute of Science and Technology
|Manipal Institute of Technology|Amity University, Noida|Maharishi University of Information Technology
|Galgotias University|Sharda University|Chandigarh University|Lovely Professional University
|Thapar Institute of Engineering and Technology|PES University|RV College of Engineering
|College of Engineering, Pune|Jamia Millia Islamia|Aligarh Muslim University|University of Delhi
|Banaras Hindu University|University of Mumbai|Savitribai Phule Pune University|Osmania University
|Massachusetts Institute of Technology|Stanford University|Carnegie Mellon University
|University of California, Berkeley|Georgia Institute of Technology|University of Toronto
|National University of Singapore|Technical University of Munich|University of Michigan
|University of Illinois Urbana-Champaign|Imperial College London|University of Edinburgh
|Arizona State University|Purdue University|University of Texas at Austin|ETH Zurich""".replace("\n", "").split("|")

SCHOOLS = ["Delhi Public School, R.K. Puram", "Kendriya Vidyalaya No. 2", "St. Xavier's High School",
           "Ryan International School", "DAV Public School", "Jawahar Navodaya Vidyalaya",
           "Sri Chaitanya Junior College", "Narayana Junior College", "Bishop Cotton Boys' School",
           "Modern School, Barakhamba Road", "Army Public School", "Amity International School"]

DEGREES = ["B.Tech", "B.Tech.", "B.E.", "B.Sc.", "B.S.", "BCA", "MCA", "M.Tech", "M.E.", "M.Sc.",
           "M.S.", "MBA", "PGDM", "Ph.D.", "Diploma", "Bachelor of Technology", "Bachelor of Science",
           "Master of Science", "Master of Technology", "Bachelor of Engineering", "Integrated M.Tech"]
FIELDS = ["Computer Science and Engineering", "Computer Science", "Information Technology",
          "Electronics and Communication Engineering", "Electrical Engineering",
          "Mechanical Engineering", "Data Science", "Artificial Intelligence and Machine Learning",
          "Computer Science and Engineering (Data Science)", "Mathematics and Computing",
          "Statistics", "Physics", "Economics", "Business Administration", "Civil Engineering",
          "Chemical Engineering", "Biotechnology", "Computer Applications", "Applied Mathematics",
          "Information Systems", "Cyber Security", "Electronics and Instrumentation"]

TITLES = ["Software Engineer", "Senior Software Engineer", "Software Development Engineer",
          "SDE-1", "SDE II", "Machine Learning Engineer", "ML Engineer Intern", "Data Scientist",
          "Data Analyst", "Data Engineer", "Backend Developer", "Frontend Developer",
          "Full Stack Developer", "DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer",
          "Research Intern", "Research Assistant", "Teaching Assistant", "Product Manager",
          "Associate Product Manager", "Business Analyst", "UI/UX Designer", "QA Engineer",
          "Embedded Systems Engineer", "Computer Vision Engineer", "NLP Engineer", "AI Engineer",
          "Technical Lead", "Engineering Manager", "Summer Intern", "Software Engineering Intern",
          "Project Trainee", "Graduate Engineer Trainee", "Systems Engineer", "Analyst",
          "Associate Consultant", "Security Analyst", "Android Developer", "iOS Developer",
          "Game Developer", "Blockchain Developer", "Quantitative Analyst", "Robotics Intern"]

CITIES = ["Noida", "New Delhi", "Gurugram", "Bengaluru", "Bangalore", "Hyderabad", "Pune", "Mumbai",
          "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Kochi", "Indore",
          "Bhubaneswar", "Visakhapatnam", "Coimbatore", "Nagpur", "Mysuru", "Thiruvananthapuram",
          "San Francisco", "Seattle", "New York", "Austin", "Boston", "London", "Berlin", "Toronto",
          "Singapore", "Dublin", "Sydney", "Dubai", "Amsterdam", "Zurich", "Tokyo"]
REGIONS = ["Uttar Pradesh", "UP", "India", "Karnataka", "Telangana", "Maharashtra", "Tamil Nadu",
           "West Bengal", "Rajasthan", "Delhi", "Haryana", "Kerala", "Gujarat", "CA", "WA", "NY",
           "TX", "MA", "UK", "Germany", "Canada", "USA", "Ontario", "Andhra Pradesh"]

TECH = """Python|C++|C|Java|JavaScript|TypeScript|Go|Rust|Kotlin|Swift|R|MATLAB|SQL|Bash|Scala|PHP
|Ruby|C#|Dart|Julia|HTML|CSS|HTML/CSS|React|React.js|Next.js|Vue.js|Angular|Node.js|Express.js
|Django|Flask|FastAPI|Spring Boot|.NET|Tailwind CSS|Redux|GraphQL|REST APIs|gRPC|PyTorch
|TensorFlow|Keras|scikit-learn|XGBoost|LightGBM|Pandas|NumPy|SciPy|Matplotlib|Seaborn|Plotly
|OpenCV|Hugging Face|Transformers|LangChain|LlamaIndex|spaCy|NLTK|YOLOv8|ONNX|TensorRT
|MLflow|Kubeflow|Airflow|Spark|PySpark|Hadoop|Kafka|Flink|dbt|Snowflake|BigQuery|Redshift
|PostgreSQL|MySQL|SQLite|MongoDB|Redis|Cassandra|DynamoDB|Elasticsearch|Neo4j|Firebase|Supabase
|Docker|Kubernetes|Helm|Terraform|Ansible|Jenkins|GitHub Actions|GitLab CI|CI/CD|Linux|Git
|AWS|AWS (EC2, S3, Lambda)|GCP|Azure|Vertex AI|SageMaker|Power BI|Tableau|Excel|Looker
|Figma|Jira|Postman|Prometheus|Grafana|Nginx|Celery|RabbitMQ|WebSockets|Socket.IO|OAuth 2.0
|JWT|Selenium|Pytest|JUnit|Jest|Arduino|Raspberry Pi|ESP32|ROS|Unity|Unreal Engine|Blender
|LaTeX|Streamlit|Gradio|OpenAI API|RAG|LLMs|CNNs|RNNs|LSTMs|GANs|Transformers|BERT|GPT-4
|Llama 3|Stable Diffusion|CUDA|OpenMP|MPI|Verilog|VHDL|Embedded C|FreeRTOS|Qt""".replace("\n", "").split("|")

VERBS = ["Developed", "Designed", "Implemented", "Built", "Led", "Optimized", "Automated",
         "Deployed", "Engineered", "Architected", "Created", "Reduced", "Improved", "Migrated",
         "Analyzed", "Trained", "Fine-tuned", "Integrated", "Collaborated with", "Mentored",
         "Spearheaded", "Streamlined", "Refactored", "Launched", "Maintained", "Researched",
         "Presented", "Published", "Containerized", "Scaled", "Benchmarked", "Debugged",
         "Prototyped", "Wrote", "Configured", "Monitored", "Coordinated", "Organized", "Delivered",
         "Owned", "Revamped", "Modernized", "Accelerated", "Secured", "Visualized", "Curated"]
OBJECTS = ["a real-time data pipeline", "REST APIs", "a microservices backend", "an ETL pipeline",
           "a CNN-based image classifier", "a recommendation engine", "dashboards", "a chatbot",
           "an end-to-end ML pipeline", "a fraud detection model", "a payment gateway integration",
           "a CI/CD pipeline", "unit and integration tests", "a mobile app", "a web application",
           "a search service", "an OCR system for invoices", "a resume parser", "a feature store",
           "an anomaly detection system", "a demand forecasting model", "a customer churn model",
           "a sentiment analysis model", "a document classification service", "a GraphQL gateway",
           "a caching layer", "an internal admin portal", "a data warehouse", "batch jobs",
           "monitoring and alerting", "a load balancer configuration", "an authentication module",
           "the onboarding flow", "a Kubernetes cluster", "a landslide early-warning system",
           "a face recognition attendance system", "an IoT sensor network", "a desk robot",
           "a speech-to-text service", "a keyword spotting model", "a LoRA fine-tuning pipeline",
           "a retrieval-augmented generation (RAG) system", "a time-series forecasting service",
           "a PDF report generator", "an A/B testing framework", "a hackathon project"]
METRICS = ["latency", "inference time", "p95 latency", "page load time", "build time",
           "cloud costs", "memory usage", "error rate", "manual effort", "processing time",
           "false positives", "deployment time", "query time", "churn", "downtime"]
RESULT_TEMPLATES = [
    "reducing {metric} by {pct}%", "cutting {metric} by {pct}%", "improving accuracy from {a}% to {b}%",
    "achieving {b}% accuracy", "serving {n}+ daily active users", "processing {n}M+ records per day",
    "saving ${k}K annually", "saving ₹{l} lakh per year", "increasing throughput by {x}x",
    "boosting F1-score to {f}", "handling {n}K requests/sec", "used by {n}+ students",
    "with {pct}% test coverage", "within {w} weeks", "ahead of schedule", "across {n} teams",
    "ranked {r} out of {n2} teams", "resulting in a {pct}% increase in engagement",
    "reducing {metric} from {s1}s to {s2}s", "achieving a {f} mAP@0.5", "with {n} contributors",
]
HEADINGS = ["Education", "Experience", "Work Experience", "Professional Experience", "Projects",
            "Academic Projects", "Personal Projects", "Skills", "Technical Skills", "Core Competencies",
            "Certifications", "Achievements", "Awards & Achievements", "Honors and Awards",
            "Publications", "Research Experience", "Leadership", "Positions of Responsibility",
            "Extracurricular Activities", "Volunteering", "Summary", "Professional Summary",
            "Objective", "Career Objective", "Internships", "Relevant Coursework", "Coursework",
            "Languages", "Interests", "Hobbies", "References", "Contact", "Profile", "About Me",
            "Training", "Workshops", "Hackathons", "Open Source Contributions", "Declaration",
            "Personal Details", "Strengths", "Tools & Technologies"]
COURSES = ["Data Structures and Algorithms", "Operating Systems", "Database Management Systems",
           "Computer Networks", "Machine Learning", "Deep Learning", "Natural Language Processing",
           "Computer Vision", "Probability and Statistics", "Linear Algebra", "Discrete Mathematics",
           "Object-Oriented Programming", "Software Engineering", "Compiler Design",
           "Theory of Computation", "Distributed Systems", "Cloud Computing", "Big Data Analytics",
           "Information Retrieval", "Reinforcement Learning", "Digital Image Processing",
           "Computer Architecture", "Cryptography", "Data Mining", "Optimization"]
CERTS = ["AWS Certified Cloud Practitioner", "AWS Certified Solutions Architect – Associate",
         "Google Cloud Professional Data Engineer", "Microsoft Certified: Azure Fundamentals (AZ-900)",
         "TensorFlow Developer Certificate", "Deep Learning Specialization – Coursera",
         "Machine Learning by Andrew Ng (Stanford Online)", "NPTEL: Python for Data Science (Elite)",
         "CKA: Certified Kubernetes Administrator", "Oracle Certified Java Programmer",
         "Cisco CCNA", "CompTIA Security+", "Meta Front-End Developer Certificate",
         "IBM Data Science Professional Certificate", "Smart India Hackathon 2024 – Finalist",
         "Winner, Hack36 (MNNIT Allahabad)", "Google Summer of Code 2025 Contributor",
         "Codeforces Expert (max rating 1712)", "LeetCode Knight – 1900+ rating",
         "Kaggle Notebooks Expert", "Dean's List (2023–24)", "Merit Scholarship, 2 years",
         "JEE Advanced AIR 4,512", "Top 5% in Flipkart GRiD 6.0", "NASA Space Apps Challenge – Nominee"]
SKILL_LABELS = ["Languages", "Programming Languages", "Frameworks", "Libraries", "Tools",
                "Developer Tools", "Databases", "Cloud", "Cloud/DevOps", "ML/AI", "Web Technologies",
                "Technologies", "Soft Skills", "Concepts", "Platforms", "Operating Systems"]
SOFT_SKILLS = ["Leadership", "Teamwork", "Communication", "Problem Solving", "Time Management",
               "Critical Thinking", "Public Speaking", "Adaptability", "Mentoring", "Agile/Scrum"]
LANGS = ["English", "Hindi", "Telugu", "Tamil", "Kannada", "Marathi", "Bengali", "Gujarati",
         "Punjabi", "Malayalam", "Urdu", "French", "German", "Spanish", "Japanese"]
LEVELS = ["Native", "Fluent", "Professional", "Intermediate", "Basic", "Conversational", "(C1)", "(B2)"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
MONTHS_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec"]
SUMMARIES = [
    "{deg} student in {field} with hands-on experience in {t1}, {t2} and {t3}.",
    "Results-driven {title} with {n}+ years of experience building scalable {domain} systems.",
    "Passionate about {domain}, open-source software and solving real-world problems with data.",
    "Seeking a {title} role where I can apply my skills in {t1} and {t2}.",
    "Detail-oriented engineer experienced in designing, deploying and monitoring {domain} applications.",
    "Final-year undergraduate interested in {domain}, systems programming and applied research.",
]
DOMAINS = ["machine learning", "backend", "distributed", "data", "computer vision", "NLP", "cloud",
           "full-stack", "embedded", "mobile", "fintech", "healthcare AI", "geospatial", "MLOps"]

COMMON_WORDS = """the of and to in for on with as by at from that this is are was were be been
be have has had will would can could should may might must not no yes all any each every both
few more most other some such only own same so than too very just also well even back still way
work team project system data model user users service services product customer business
research development design analysis performance quality support management process report
application applications platform tool tools technology software hardware network security
experience skills knowledge ability responsible handled worked using used use making made built
new first last long great little high large small next early young important public private
good best better right left real full free open key main major general specific technical
strong excellent effective efficient successful complex simple multiple various several key
including within across between through during before after over under about against among
students faculty university college school department lab club society chapter committee event
events workshop seminar conference paper journal thesis course semester year years month months
week weeks day days time hours minutes members people clients partners stakeholders managers
engineers developers designers analysts teams company organization startup industry market
sales revenue growth cost costs budget plan plans strategy goals targets results impact value
features feature release releases version versions code review reviews testing tests bugs issues
documentation requirements specifications deployment production staging environment pipeline
pipelines database databases queries query server servers client frontend backend api apis web
mobile cloud infrastructure scalable reliable secure robust real-time automated manual daily
weekly monthly annual national international state district regional global local remote onsite
hybrid full-time part-time intern internship volunteer member lead head coordinator president
secretary treasurer captain winner finalist participant speaker mentor trainer author co-author
""".split()


def _r(rng: random.Random, seq: Sequence):
    return seq[rng.randrange(len(seq))]


class ResumeTextSampler:
    """Draws one line of resume-like text per call."""

    def __init__(self, rng: Optional[random.Random] = None, extra_lines: Optional[List[str]] = None,
                 extra_weight: float = 0.2):
        self.rng = rng or random.Random()
        self.extra_lines = [l.strip() for l in (extra_lines or []) if l.strip()]
        gens: List[Tuple[float, Callable[[], str]]] = [
            (5, self.heading), (2.5, self.name), (8, self.contact), (6, self.job_line),
            (6, self.date_range), (2.5, self.location), (7, self.education), (28, self.bullet),
            (8, self.skills), (5, self.project), (3, self.cert), (4, self.summary),
            (2, self.languages), (6, self.random_words), (5, self.random_chars), (2, self.coursework),
        ]
        if self.extra_lines:
            total = sum(w for w, _ in gens)
            gens.append((total * extra_weight / (1 - extra_weight), self.from_corpus))
        self._weights = [w for w, _ in gens]
        self._gens = [g for _, g in gens]

    # ---------------------------------------------------------------- pieces
    def person(self) -> Tuple[str, str]:
        return _r(self.rng, FIRST_NAMES), _r(self.rng, LAST_NAMES)

    def handle(self) -> str:
        f, l = self.person()
        f, l = f.lower().replace("'", ""), l.lower().replace("'", "")
        pat = self.rng.randrange(6)
        num = str(self.rng.randrange(1, 9999)) if self.rng.random() < 0.4 else ""
        return [f"{f}{l}", f"{f}.{l}", f"{f}-{l}", f"{f}_{l}", f"{f[0]}{l}", f"{l}{f[:3]}"][pat] + num

    def email(self) -> str:
        dom = _r(self.rng, ["gmail.com", "gmail.com", "gmail.com", "outlook.com", "yahoo.com",
                            "yahoo.co.in", "hotmail.com", "protonmail.com", "icloud.com",
                            "muit.ac.in", "iitd.ac.in", "student.university.edu", "company.com"])
        return f"{self.handle()}@{dom}"

    def phone(self) -> str:
        d = lambda n: "".join(str(self.rng.randrange(10)) for _ in range(n))
        first = str(self.rng.choice([6, 7, 8, 9]))
        return _r(self.rng, [
            f"+91 {first}{d(4)} {d(5)}", f"+91-{first}{d(9)}", f"+91 {first}{d(9)}", f"{first}{d(9)}",
            f"({d(3)}) {d(3)}-{d(4)}", f"+1 {d(3)}-{d(3)}-{d(4)}", f"+1 ({d(3)}) {d(3)}-{d(4)}",
            f"+44 {d(2)} {d(4)} {d(4)}", f"0{first}{d(4)} {d(5)}", f"+91 {first}{d(2)}-{d(3)}-{d(4)}",
        ])

    def url(self) -> str:
        h = self.handle()
        u = _r(self.rng, [f"linkedin.com/in/{h}", f"github.com/{h}", f"{h}.github.io",
                          f"www.{h}.dev", f"kaggle.com/{h}", f"leetcode.com/u/{h}",
                          f"medium.com/@{h}", f"{h}.netlify.app", f"behance.net/{h}",
                          f"github.com/{h}/{_r(self.rng, ['resume-parser', 'ml-notes', 'dotfiles', 'ocr-lite'])}"])
        if self.rng.random() < 0.15:
            u = "https://" + u
        return u

    def year(self, lo=2008, hi=2028) -> int:
        return self.rng.randrange(lo, hi)

    def month(self) -> str:
        return _r(self.rng, MONTHS) if self.rng.random() < 0.35 else _r(self.rng, MONTHS_ABBR)

    def dash(self) -> str:
        return _r(self.rng, [" – ", " – ", " - ", " - ", "–", "-", " — ", " to ", " — "])

    def company(self) -> str:
        if self.rng.random() < 0.55:
            return _r(self.rng, COMPANIES)
        name = f"{_r(self.rng, COMPANY_PARTS)}{_r(self.rng, ['', '', ' ', ''])}{_r(self.rng, COMPANY_PARTS).lower() if self.rng.random() < 0.3 else ''}"
        return " ".join(x for x in [name.strip(), _r(self.rng, COMPANY_SUFFIX), _r(self.rng, COMPANY_LEGAL)] if x)

    def place(self) -> str:
        return f"{_r(self.rng, CITIES)}, {_r(self.rng, REGIONS)}" if self.rng.random() < 0.8 else _r(self.rng, ["Remote", "Hybrid", "On-site", "Remote (India)"])

    # ----------------------------------------------------------- line types
    def heading(self) -> str:
        h = _r(self.rng, HEADINGS)
        return h.upper() if self.rng.random() < 0.55 else h

    def name(self) -> str:
        f, l = self.person()
        n = f"{f} {_r(self.rng, string.ascii_uppercase)}. {l}" if self.rng.random() < 0.15 else f"{f} {l}"
        return n.upper() if self.rng.random() < 0.5 else n

    def contact(self) -> str:
        parts = [self.phone(), self.email(), self.url(), self.url(), self.place()]
        self.rng.shuffle(parts)
        k = self.rng.randint(1, 4)
        sep = _r(self.rng, [" | ", " | ", " • ", " · ", "  ", ", ", " / ", " – "])
        return sep.join(parts[:k])

    def date_range(self) -> str:
        y1 = self.year()
        y2 = y1 + self.rng.randint(0, 4)
        end = _r(self.rng, ["Present", "Current", "Now", "Ongoing", f"{self.month()} {y2}", f"{y2}"])
        fmt = self.rng.randrange(7)
        if fmt == 0:
            s = f"{self.month()} {y1}{self.dash()}{end}"
        elif fmt == 1:
            s = f"{y1}{self.dash()}{y2}"
        elif fmt == 2:
            s = f"{self.rng.randint(1, 12):02d}/{y1}{self.dash()}{self.rng.randint(1, 12):02d}/{y2}"
        elif fmt == 3:
            s = f"{self.month()} '{y1 % 100:02d}{self.dash()}{self.month()} '{y2 % 100:02d}"
        elif fmt == 4:
            s = _r(self.rng, ["Expected", "Graduating", "Expected:", "Batch of"]) + f" {self.month()} {y2}"
        elif fmt == 5:
            s = _r(self.rng, ["Summer", "Winter", "Fall", "Spring"]) + f" {y1}"
        else:
            s = f"{self.month()} {y1}"
        return s

    def location(self) -> str:
        return self.place()

    def job_line(self) -> str:
        t, c = _r(self.rng, TITLES), self.company()
        fmt = self.rng.randrange(6)
        if fmt == 0:
            return f"{t}, {c}"
        if fmt == 1:
            return f"{t} | {c}"
        if fmt == 2:
            return f"{c} — {t}"
        if fmt == 3:
            return f"{t} at {c}"
        if fmt == 4:
            return f"{c}{self.dash()}{self.place()}"
        if self.rng.random() < 0.5:     # right-aligned date after a tab stop
            return f"{_r(self.rng, [t, c])}{' ' * self.rng.randint(3, 14)}{self.date_range()}"
        return f"{t} ({self.date_range()})"

    def education(self) -> str:
        fmt = self.rng.randrange(8)
        deg, field = _r(self.rng, DEGREES), _r(self.rng, FIELDS)
        if fmt == 0:
            return f"{deg} in {field}"
        if fmt == 1:
            return f"{deg}, {field}"
        if fmt == 2:
            return _r(self.rng, UNIVERSITIES)
        if fmt == 3:
            return f"{_r(self.rng, UNIVERSITIES)}, {_r(self.rng, CITIES)}"
        if fmt == 4:
            return _r(self.rng, [f"CGPA: {self.rng.uniform(6.5, 9.9):.2f}/10", f"CGPA {self.rng.uniform(6.5, 9.9):.1f}",
                                 f"GPA: {self.rng.uniform(2.8, 4.0):.2f}/4.0", f"Percentage: {self.rng.uniform(60, 99):.1f}%",
                                 f"SGPA: {self.rng.uniform(7, 10):.2f}", f"{self.rng.uniform(70, 99):.1f}%"])
        if fmt == 5:
            return _r(self.rng, ["Class XII", "Class X", "Senior Secondary (XII)", "Secondary (X)",
                                 "HSC", "SSC", "Intermediate"]) + f" ({_r(self.rng, ['CBSE', 'ICSE', 'State Board', 'ISC', 'BIEAP'])})" \
                + f"{self.dash()}{self.rng.uniform(70, 99):.1f}%"
        if fmt == 6:
            return _r(self.rng, SCHOOLS)
        return f"{deg} ({field}){self.dash()}{self.date_range()}"

    def result(self) -> str:
        t = _r(self.rng, RESULT_TEMPLATES)
        a = self.rng.randint(60, 90)
        return t.format(metric=_r(self.rng, METRICS), pct=self.rng.randint(5, 85), a=a,
                        b=min(99, a + self.rng.randint(1, 15)), n=self.rng.choice([5, 10, 25, 50, 100, 200, 500, 1000, 12, 40]),
                        k=self.rng.randint(5, 250), l=self.rng.randint(1, 40), x=self.rng.choice([2, 3, 4, 5, 10, 1.5, 2.5]),
                        f=f"{self.rng.uniform(0.6, 0.98):.2f}", w=self.rng.randint(2, 12),
                        r=self.rng.choice(["1st", "2nd", "3rd", "4th", "Top 10", "#1"]), n2=self.rng.choice([150, 300, 1200, 4500]),
                        s1=self.rng.randint(3, 30), s2=round(self.rng.uniform(0.2, 2.9), 1))

    def bullet(self) -> str:
        v, o = _r(self.rng, VERBS), _r(self.rng, OBJECTS)
        techs = self.rng.sample(TECH, self.rng.randint(1, 3))
        tech = techs[0] if len(techs) == 1 else ", ".join(techs[:-1]) + " and " + techs[-1]
        fmt = self.rng.randrange(5)
        if fmt == 0:
            s = f"{v} {o} using {tech}, {self.result()}"
        elif fmt == 1:
            s = f"{v} {o} with {tech}; {self.result()}"
        elif fmt == 2:
            s = f"{v} {o} in {tech} {_r(self.rng, ['for', 'to support', 'enabling'])} {_r(self.rng, COMMON_WORDS)} {_r(self.rng, COMMON_WORDS)}"
        elif fmt == 3:
            s = f"{v} {o} {self.result()} and {self.result()}"
        else:
            s = f"{v} {o} ({tech})"
        if self.rng.random() < 0.6:
            s += "."
        return s

    def skills(self) -> str:
        label = _r(self.rng, SKILL_LABELS)
        pool = SOFT_SKILLS if label == "Soft Skills" else TECH
        items = self.rng.sample(pool, self.rng.randint(2, 8))
        sep = _r(self.rng, [", ", ", ", " | ", " • ", " · ", "; ", " / "])
        body = sep.join(items)
        return f"{label}: {body}" if self.rng.random() < 0.8 else body

    def project(self) -> str:
        name = _r(self.rng, ["Resume Parser", "Smart Attendance System", "Crop Disease Detector",
                             "Stock Price Predictor", "Campus Connect", "Expense Tracker",
                             "Landslide Early Warning System", "Code Reader", "Media Bias Estimator",
                             "WiFi Room Mapper", "Hinglish Speech Recognizer", "Desk Robot JINX",
                             "Movie Recommender", "Fake News Classifier", "Traffic Sign Recognition",
                             "Personal Portfolio", "Chat App", "E-commerce Backend", "PDF Q&A Bot"])
        techs = ", ".join(self.rng.sample(TECH, self.rng.randint(1, 4)))
        fmt = self.rng.randrange(4)
        if fmt == 0:
            return f"{name} | {techs} | {self.url()}"
        if fmt == 1:
            return f"{name} ({techs})"
        if fmt == 2:
            return f"{name}{self.dash()}{self.date_range()}"
        return f"{name}: {_r(self.rng, ['A', 'An'])} {_r(self.rng, ['web app', 'tool', 'system', 'model', 'library'])} for {_r(self.rng, COMMON_WORDS)} {_r(self.rng, COMMON_WORDS)}"

    def cert(self) -> str:
        c = _r(self.rng, CERTS)
        if self.rng.random() < 0.3:
            c += f" ({self.year(2018, 2027)})"
        return c

    def summary(self) -> str:
        t1, t2, t3 = self.rng.sample(TECH, 3)
        return _r(self.rng, SUMMARIES).format(deg=_r(self.rng, DEGREES), field=_r(self.rng, FIELDS),
                                              t1=t1, t2=t2, t3=t3, title=_r(self.rng, TITLES).lower(),
                                              n=self.rng.randint(1, 12), domain=_r(self.rng, DOMAINS))

    def languages(self) -> str:
        langs = self.rng.sample(LANGS, self.rng.randint(1, 4))
        if self.rng.random() < 0.5:
            langs = [f"{l} ({_r(self.rng, LEVELS).strip('()')})" for l in langs]
        return "Languages: " + ", ".join(langs) if self.rng.random() < 0.5 else ", ".join(langs)

    def coursework(self) -> str:
        return ("Relevant Coursework: " if self.rng.random() < 0.5 else "") + ", ".join(self.rng.sample(COURSES, self.rng.randint(2, 6)))

    def random_words(self) -> str:
        words = []
        for _ in range(self.rng.randint(1, 14)):
            src = self.rng.random()
            if src < 0.6:
                w = _r(self.rng, COMMON_WORDS)
            elif src < 0.75:
                w = _r(self.rng, TECH)
            elif src < 0.85:
                w = _r(self.rng, FIRST_NAMES + LAST_NAMES)
            else:
                w = _r(self.rng, VERBS).split()[0].lower()
            r = self.rng.random()
            if r < 0.15:
                w = w.capitalize()
            elif r < 0.2:
                w = w.upper()
            if self.rng.random() < 0.12:
                w += _r(self.rng, [",", ".", ";", ":", "!", "?", ")", "%"])
            words.append(w)
        return " ".join(words)

    def random_chars(self) -> str:
        fmt = self.rng.randrange(7)
        if fmt == 0:   # identifiers, roll numbers
            return _r(self.rng, ["ID: ", "Roll No. ", "Reg. No.: ", "Enrollment: ", ""]) + \
                f"{self.rng.randint(2018, 2027)}{_r(self.rng, ['CSE', 'CS', 'IT', 'ECE', 'DS', 'AI'])}{self.rng.randint(1, 1999):04d}"
        if fmt == 1:   # money and numbers
            return _r(self.rng, [f"₹{self.rng.randint(1, 99)},{self.rng.randint(0, 99):02d},{self.rng.randint(0, 999):03d}",
                                 f"${self.rng.uniform(0.1, 99):.1f}M", f"€{self.rng.randint(1, 999)}K",
                                 f"£{self.rng.randint(10, 99)},{self.rng.randint(100, 999)}",
                                 f"{self.rng.randint(1, 99)}.{self.rng.randint(0, 9)} LPA",
                                 f"{self.rng.randint(10, 999):,} users", f"~{self.rng.randint(5, 95)}%",
                                 f"{self.rng.uniform(1, 20):.1f}x", f"{self.rng.randint(1, 30)}°C",
                                 f"v{self.rng.randint(0, 9)}.{self.rng.randint(0, 20)}.{self.rng.randint(0, 9)}"])
        if fmt == 2:   # random alphanumerics
            n = self.rng.randint(3, 24)
            pool = string.ascii_letters + string.digits + "  -_./@#&+:"
            return "".join(self.rng.choice(pool) for _ in range(n)).strip() or "A1"
        if fmt == 3:   # code-ish tokens
            return _r(self.rng, ["pip install -r requirements.txt", "C++17/20", "O(n log n)",
                                 "f(x) = Wx + b", "k-NN, SVM, PCA", "<10 ms", ">= 95%", "[PDF]",
                                 "{\"id\": 42}", "x86_64 / ARM64", "#include <vector>", "~/.config",
                                 "A+ grade", "S&P 500", "Q3 FY24", "24x7", "3:1", "e.g.", "i.e.", "etc."])
        if fmt == 4:   # numbers lists
            return ", ".join(str(self.rng.randint(0, 9999)) for _ in range(self.rng.randint(2, 8)))
        if fmt == 5:
            return "".join(self.rng.choice(string.punctuation + string.ascii_letters) for _ in range(self.rng.randint(2, 12)))
        return f"Page {self.rng.randint(1, 3)} of {self.rng.randint(1, 3)}"

    def from_corpus(self) -> str:
        line = _r(self.rng, self.extra_lines).split()
        if len(line) <= 3:
            return " ".join(line)
        n = self.rng.randint(1, min(20, len(line)))
        s = self.rng.randrange(0, len(line) - n + 1)
        return " ".join(line[s:s + n])

    # -------------------------------------------------------------- public
    def sample(self) -> str:
        gen = self.rng.choices(self._gens, weights=self._weights, k=1)[0]
        text = gen().strip()
        r = self.rng.random()
        if gen not in (self.heading, self.name):
            if r < 0.04:
                text = text.upper()
            elif r < 0.06:
                text = text.lower()
        if self.rng.random() < 0.3:      # Word-style smart quotes
            text = smart_quotes(text)
        if gen is self.bullet and self.rng.random() < 0.65:
            # Word puts a tab after the bullet: render a wider gap (label keeps one space)
            gap = " " if self.rng.random() < 0.6 else " " * self.rng.randint(2, 6)
            text = self.bullet_glyph() + gap + text
        return text

    def bullet_glyph(self) -> str:
        r = self.rng.random()
        if r < 0.55:
            return "•"
        if r < 0.85:
            return _r(self.rng, BULLET_GLYPHS)
        return _r(self.rng, ["-", "–", "*", "o", ">"])


def smart_quotes(text: str) -> str:
    out, double_open = [], True
    for i, ch in enumerate(text):
        prev = text[i - 1] if i else " "
        if ch == "'":
            out.append("‘" if prev in " ([{" else "’")
        elif ch == '"':
            out.append("“" if double_open else "”")
            double_open = not double_open
        else:
            out.append(ch)
    return "".join(out)
