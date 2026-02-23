import sys
import importlib
import pysqlite3
sys.modules["sqlite3"] = importlib.import_module("pysqlite3")

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import anthropic
import pandas as pd

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader  # type: ignore

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
    from langchain.vectorstores import FAISS  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEPT_TITLES: dict[str, str] = {
    "Engineering": "Software Engineer",
    "Sales": "Sales Executive",
    "HR": "HR Specialist",
    "Finance": "Financial Analyst",
    "Operations": "Operations Specialist",
}


@dataclass
class ExtractionResult:
    leave: dict[str, str] = field(default_factory=dict)
    travel: dict[str, str] = field(default_factory=dict)
    wfo: dict[str, str] = field(default_factory=dict)
    citations: dict[str, str] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = """You are a precise HR policy extraction assistant.
Read the provided HR policy document excerpts and extract entitlement values
for the given employee band and department.

Rules:
- Only use values explicitly stated in the document excerpts.
- Never invent, assume, or interpolate values.
- If a value is not clearly present in the documents, set it to null and add
  the field name to missing_fields.
- Respond with valid JSON only — no preamble, no markdown fences.
"""


# build prompt
def build_extraction_prompt(band: str, department: str, chunks: list[dict]) -> str:
    chunks_text = "\n".join(
        f"\n--- CHUNK {i+1} (source: {c['source']}) ---\n{c['content']}"
        for i, c in enumerate(chunks)
    )
    return f"""Extract HR policy entitlements for:
- Band: {band}
- Department: {department}

For each extracted value, add a short supporting citation (max 15 words from the document).
If a value is not found, set it to null and include the field name in missing_fields.

DOCUMENT EXCERPTS:
{chunks_text}

Respond with this JSON structure and nothing else:
{{
  "leave": {{
    "total_days": "<value or null>",
    "earned": "<value or null>",
    "sick": "<value or null>",
    "casual": "<value or null>",
    "wfh": "<value or null>",
    "wfo": "<value or null>"
  }},
  "travel": {{
    "flight": "<value or null>",
    "hotel": "<value or null>",
    "per_diem_domestic": "<value or null>",
    "per_diem_intl": "<value or null>",
    "approval": "<value or null>"
  }},
  "wfo": {{
    "minimum": "<value or null>",
    "suggested": "<value or null>",
    "notes": "<value or null>"
  }},
  "citations": {{
    "<field_name>": "<supporting phrase from document, max 15 words>"
  }},
  "missing_fields": ["<field names not found in documents>"]
}}"""


REQUIRED_FIELDS = {
    "leave": ["total_days", "earned", "sick", "casual", "wfh", "wfo"],
    "travel": ["flight", "hotel", "per_diem_domestic", "per_diem_intl", "approval"],
    "wfo": ["minimum", "suggested", "notes"],
}


# validate extraction
def validate_extraction(result: ExtractionResult) -> ValidationResult:
    issues = []

    for section, fields in REQUIRED_FIELDS.items():
        section_data = getattr(result, section)
        for f in fields:
            if not section_data.get(f):
                issues.append(f"{section}.{f} is missing or null in the source documents")

    total = result.leave.get("total_days", "")
    if total and total not in ("Unlimited (with approval)", "N/A"):
        try:
            days = int(re.sub(r"[^\d]", "", total))
            if not (5 <= days <= 60):
                issues.append(f"total_days value '{total}' looks implausible (expected 5–60)")
        except ValueError:
            pass

    return ValidationResult(passed=len(issues) == 0, issues=issues)


class HROfferLetterPipeline:

    def __init__(self, anthropic_api_key: str) -> None:
        self.employees_df: pd.DataFrame | None = None
        self.vector_store: FAISS | None = None
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self._embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._llm = anthropic.Anthropic(api_key=anthropic_api_key)

    # parse PDF
    @staticmethod
    def _parse_pdf(filepath: str) -> str:
        reader = PdfReader(filepath)
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            raise ValueError(f"{filepath} produced no extractable text — it may be scanned or image-based.")
        return text

    # load data
    def load(
        self,
        csv_path: str = "Employee_List.csv",
        leave_pdf: str = "HR-Leave-Policy.pdf",
        travel_pdf: str = "HR-Travel-Policy.pdf",
    ) -> None:
        self.employees_df = pd.read_csv(csv_path)

        documents: list[Document] = []
        for source_name, filepath in [("HR-Leave-Policy", leave_pdf), ("HR-Travel-Policy", travel_pdf)]:
            text = self._parse_pdf(filepath)
            doc = Document(page_content=text, metadata={"source": source_name})
            documents.extend(self._text_splitter.split_documents([doc]))

        self.vector_store = FAISS.from_documents(documents, self._embeddings)

    # retrieve chunks
    def _retrieve(self, band: str, department: str) -> list[dict]:
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialised. Call load() first.")

        leave_query = f"band {band} department {department} leave entitlement earned sick casual WFH WFO days"
        travel_query = f"band {band} department {department} travel flight hotel per diem approval reimbursement"

        seen: set[str] = set()
        chunks: list[dict] = []

        for query in (leave_query, travel_query):
            for doc in self.vector_store.similarity_search(query, k=4):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    chunks.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                    })

        return chunks

    # LLM extraction
    def _extract(self, band: str, department: str, chunks: list[dict]) -> ExtractionResult:
        prompt = build_extraction_prompt(band, department, chunks)

        response = self._llm.messages.create(
            model="claude-opus-4-6",
            max_tokens=1500,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        clean = re.sub(r"^```(?:json)?\n?", "", raw)
        clean = re.sub(r"\n?```$", "", clean).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Claude returned malformed JSON: {exc}\n\nRaw response:\n{raw}") from exc

        def nulls_to_empty(d: dict) -> dict:
            return {k: (v if v is not None else "") for k, v in d.items()}

        return ExtractionResult(
            leave=nulls_to_empty(data.get("leave", {})),
            travel=nulls_to_empty(data.get("travel", {})),
            wfo=nulls_to_empty(data.get("wfo", {})),
            citations=data.get("citations", {}),
            missing_fields=data.get("missing_fields", []),
        )

    # find employee
    def _find_employee(self, name: str) -> dict:
        if self.employees_df is None:
            raise RuntimeError("Employee data not loaded. Call load() first.")

        matches = self.employees_df[
            self.employees_df["Employee Name"].str.contains(name.strip(), case=False, na=False)
        ]

        if matches.empty:
            raise ValueError(f"No employee found matching '{name}'.")
        if len(matches) > 1:
            names = ", ".join(matches["Employee Name"].tolist())
            raise ValueError(f"Multiple employees match '{name}': {names}. Please be more specific.")

        return matches.iloc[0].to_dict()

    # format salary
    @staticmethod
    def _format_salary(info: dict) -> dict[str, str]:
        ctc = int(info["Total CTC (INR)"])
        return {
            "base_salary":       f"₹{int(info['Base Salary (INR)']):,}",
            "performance_bonus": f"₹{int(info['Performance Bonus (INR)']):,}",
            "retention_bonus":   f"₹{int(info['Retention Bonus (INR)']):,}",
            "total_ctc":         f"₹{ctc:,}",
            "monthly_gross":     f"₹{ctc // 12:,}",
        }

    # generate letter
    def generate(self, employee_name: str) -> str:
        info = self._find_employee(employee_name)
        band = info["Band"]
        department = info["Department"]

        chunks = self._retrieve(band, department)
        extraction = self._extract(band, department, chunks)

        validation = validate_extraction(extraction)
        if not validation.passed:
            issues_str = "\n  - ".join(validation.issues)
            raise ValueError(
                f"Policy extraction incomplete for band {band} / {department}.\n"
                f"The following values could not be found in the HR documents:\n  - {issues_str}\n\n"
                f"Check that your policy PDFs contain explicit entitlements for this band."
            )

        salary = self._format_salary(info)
        title = DEPT_TITLES.get(department, "Team Member")
        leave = extraction.leave
        travel = extraction.travel
        wfo = extraction.wfo

        return f"""
═══════════════════════════════════════════════════════════════
📄 OFFER LETTER – COMPANY ABC
═══════════════════════════════════════════════════════════════

Date: {datetime.now().strftime('%B %d, %Y')}

Dear {info['Employee Name']},

We are pleased to extend this offer of employment for the position of
{title} in the {department} team at Company ABC.

CANDIDATE DETAILS:
• Name          : {info['Employee Name']}
• Position      : {title}
• Band Level    : {band}
• Department    : {department}
• Work Location : {info['Location']}
• Joining Date  : {info['Joining Date']}

═══════════════════════════════════════════════════════════════
1. 💰 COMPENSATION & SALARY BREAKDOWN
═══════════════════════════════════════════════════════════════

Component                            Amount (INR)
─────────────────────────────────────────────────
Base Salary (Fixed)                  {salary['base_salary']}
Performance Bonus                    {salary['performance_bonus']}
Retention Bonus                      {salary['retention_bonus']}
─────────────────────────────────────────────────
TOTAL ANNUAL CTC                     {salary['total_ctc']}
Monthly Gross (Approx.)              {salary['monthly_gross']}

• Performance bonuses are paid quarterly based on individual and company performance.
• Retention bonus is disbursed over the specified period per company policy.
• Salary reviews are conducted annually based on performance and market benchmarks.

═══════════════════════════════════════════════════════════════
2. 🏖️ LEAVE ENTITLEMENTS & WORK ARRANGEMENTS (Band {band})
═══════════════════════════════════════════════════════════════

• Annual Leave Entitlement : {leave.get('total_days', 'N/A')} days
  - Earned Leave           : {leave.get('earned', 'N/A')} days
  - Sick Leave             : {leave.get('sick', 'N/A')} days
  - Casual Leave           : {leave.get('casual', 'N/A')} days
• Work From Home           : {leave.get('wfh', 'N/A')}
• Work From Office         : {leave.get('wfo', 'N/A')}

{department} Team-Specific Requirements:
• Minimum WFO  : {wfo.get('minimum', 'N/A')}
• Suggested    : {wfo.get('suggested', 'N/A')}
• Special Note : {wfo.get('notes', 'N/A')}

Leave Management:
• All leaves must be applied through HRMS with manager approval.
• Leave balances reset annually on January 1st.
• Up to 10 days may be carried forward to the next year.
• Emergency leave can be regularised post-facto.

═══════════════════════════════════════════════════════════════
3. ✈️ TRAVEL POLICY & BENEFITS (Band {band})
═══════════════════════════════════════════════════════════════

• Flight Class            : {travel.get('flight', 'N/A')}
• Hotel Cap               : {travel.get('hotel', 'N/A')}
• Per Diem (Domestic)     : {travel.get('per_diem_domestic', 'N/A')}
• Per Diem (International): {travel.get('per_diem_intl', 'N/A')}
• Approval Required       : {travel.get('approval', 'N/A')}

• All travel must be booked through approved corporate platforms.
• Expenses are reimbursed as per the company travel policy.

Additional Benefits:
• Home office setup support  : ₹5,000 (L3 and above)
• Monthly internet stipend   : ₹1,000/month (hybrid-eligible roles)
• Health insurance and statutory benefits as per company policy.

═══════════════════════════════════════════════════════════════
4. 🔒 EMPLOYMENT TERMS & CONDITIONS
═══════════════════════════════════════════════════════════════

• Employment Type  : Full-time, permanent position
• Probation Period : 3 months from joining date
• Notice Period    : 60 days (15 days during probation)
• Working Hours    : As per company policy and team requirements

Confidentiality & IP:
• All work products and innovations belong to Company ABC.
• Strict confidentiality of proprietary information is required.
• A non-disclosure agreement will be provided separately.

═══════════════════════════════════════════════════════════════
5. 📋 APPLICABLE HR POLICIES
═══════════════════════════════════════════════════════════════

Your employment is governed by:
• Company ABC Employee Handbook
• Leave & Work from Office Policy (Version: July 2025)
• HR Travel Policy (Version: July 2025)
• Code of Conduct and other company policies

All policies are available on the company intranet and HRMS portal.

═══════════════════════════════════════════════════════════════
6. 🎯 NEXT STEPS
═══════════════════════════════════════════════════════════════

To accept this offer:
1. Sign and return this letter within 5 working days.
2. Submit required documents for background verification.
3. Complete pre-joining formalities as communicated by HR.

Your assigned HR Business Partner will contact you with your onboarding
timeline, document requirements, and first-day joining instructions.

We look forward to welcoming you to the Company ABC family!

Warm regards,

Aarti Nair
HR Business Partner
Company ABC

📧 peopleops@companyabc.com
🌐 www.companyabc.com
📞 +91-XXXX-XXXXXX

═══════════════════════════════════════════════════════════════
""".strip()


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Set the ANTHROPIC_API_KEY environment variable before running.")

    pipeline = HROfferLetterPipeline(anthropic_api_key=api_key)
    pipeline.load()

    name = input("\nEnter employee name: ").strip()
    letter = pipeline.generate(name)

    print("\n" + letter)

    save = input("\nSave to file? (y/n): ").strip().lower()
    if save == "y":
        filename = f"{name.replace(' ', '_')}_offer_letter.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(letter)
        print(f"Saved to {filename}")
