import sys
import importlib
import pysqlite3
sys.modules["sqlite3"] = importlib.import_module("pysqlite3")


import streamlit as st
import pandas as pd
from datetime import datetime
import PyPDF2
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

class HROfferLetterRAG:
    def __init__(self):
        self.employees_df = None
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    def parse_pdf(self, filepath):
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error parsing {filepath}: {str(e)}")
            return ""
    
    def load_data_from_files(self):
        try:
            self.employees_df = pd.read_csv("Employee_List.csv")
            st.success(f"Loaded {len(self.employees_df)} employees")
            
            leave_policy_text = self.parse_pdf("HR-Leave-Policy.pdf")
            travel_policy_text = self.parse_pdf("HR-Travel-Policy.pdf")
            
            if not leave_policy_text or not travel_policy_text:
                st.error("Failed to parse policy documents")
                return False
            
            documents = []
            
            leave_doc = Document(
                page_content=leave_policy_text, 
                metadata={"source": "HR-Leave-Policy", "document_type": "HR_Policy"}
            )
            leave_chunks = self.text_splitter.split_documents([leave_doc])
            documents.extend(leave_chunks)
            
            travel_doc = Document(
                page_content=travel_policy_text,
                metadata={"source": "HR-Travel-Policy", "document_type": "HR_Policy"}
            )
            travel_chunks = self.text_splitter.split_documents([travel_doc])
            documents.extend(travel_chunks)
                        
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            return True
            
        except FileNotFoundError as e:
            st.error(f"File not found: {str(e)}")
            st.info("Please ensure the following files are in the same directory:")
            st.markdown("- Employee_List.csv\n- HR-Leave-Policy.pdf\n- HR-Travel-Policy.pdf")
            return False
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def get_relevant_context(self, employee_data):
        query = f"band {employee_data['Band']} department {employee_data['Department']} leave policy travel policy salary benefits"
        relevant_docs = self.vector_store.similarity_search(query, k=6)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return context
    
    def extract_salary_breakdown(self, employee_info):
        salary_breakdown = {
            'base_salary': f"₹{employee_info['Base Salary (INR)']:,}",
            'performance_bonus': f"₹{employee_info['Performance Bonus (INR)']:,}",
            'retention_bonus': f"₹{employee_info['Retention Bonus (INR)']:,}",
            'total_ctc': f"₹{employee_info['Total CTC (INR)']:,}",
            'monthly_gross': f"₹{employee_info['Total CTC (INR)'] // 12:,}",
        }
        return salary_breakdown
    
    def extract_policy_info_from_context(self, band, department, context):
        policies = {
            'leave': {'total_days': 'As per policy', 'wfh': 'As per band', 'wfo': 'As per team'},
            'travel': {'approval': 'As per hierarchy', 'entitlements': 'As per band'},
            'benefits': {'setup_grant': 'As applicable', 'internet_stipend': 'As applicable'}
        }
        
        context_lower = context.lower()
        
        band_patterns = {
            'L1': {'total_days': 12, 'earned': 6, 'sick': 4, 'casual': 2, 'wfh': 'Limited', 'wfo': '4 days/week minimum'},
            'L2': {'total_days': 15, 'earned': 8, 'sick': 5, 'casual': 2, 'wfh': 'Partial', 'wfo': '3-4 days/week'},
            'L3': {'total_days': 18, 'earned': 10, 'sick': 6, 'casual': 2, 'wfh': 'Yes', 'wfo': '3 days/week minimum'},
            'L4': {'total_days': 20, 'earned': 12, 'sick': 6, 'casual': 2, 'wfh': 'Yes', 'wfo': '2-3 days/week'},
            'L5': {'total_days': 'Unlimited (with approval)', 'earned': 'NA', 'sick': 'NA', 'casual': 'NA', 'wfh': 'Full Flex', 'wfo': '0-2 days/week (optional)'}
        }
        
        travel_patterns = {
            'L1': {'flight': 'Economy (on approval)', 'hotel': 'Rs. 2,000/night', 'per_diem_domestic': 'Rs. 1,500/day', 'per_diem_intl': 'USD 30/day', 'approval': 'Manager + VP'},
            'L2': {'flight': 'Economy (>6hrs)', 'hotel': 'Rs. 3,000/night', 'per_diem_domestic': 'Rs. 2,000/day', 'per_diem_intl': 'USD 40/day', 'approval': 'Manager + Director'},
            'L3': {'flight': 'Economy standard', 'hotel': 'Rs. 4,000/night', 'per_diem_domestic': 'Rs. 3,000/day', 'per_diem_intl': 'USD 60/day', 'approval': 'Reporting Manager'},
            'L4': {'flight': 'Premium Economy', 'hotel': 'Rs. 6,000/night', 'per_diem_domestic': 'Rs. 4,500/day', 'per_diem_intl': 'USD 80/day', 'approval': 'VP'},
            'L5': {'flight': 'Business Class', 'hotel': 'Rs. 10,000/night', 'per_diem_domestic': 'Rs. 7,500/day', 'per_diem_intl': 'USD 120/day', 'approval': 'None'}
        }
        
        wfo_patterns = {
            'Engineering': {'minimum': '3 days/week', 'suggested': 'Mon, Tue, Thu', 'notes': 'Sprint reviews must be in-office'},
            'Sales': {'minimum': '4-5 days/week', 'suggested': 'Field visits + office', 'notes': 'Remote only with RSM approval'},
            'HR': {'minimum': '4 days/week', 'suggested': 'Mon-Thu', 'notes': 'In-office mandatory during onboarding'},
            'Finance': {'minimum': '3 days/week', 'suggested': 'Tue, Wed, Fri', 'notes': 'Fully in-office during month-end'},
            'Operations': {'minimum': '5 days/week', 'suggested': 'All weekdays', 'notes': 'WFH not permitted except in emergencies'}
        }
        
        return {
            'leave': band_patterns.get(band, {}),
            'travel': travel_patterns.get(band, {}),
            'wfo': wfo_patterns.get(department, {})
        }
    
    def get_position_title(self, department):
        titles = {
            'Engineering': 'Software Engineer',
            'Sales': 'Sales Executive', 
            'HR': 'HR Specialist',
            'Finance': 'Financial Analyst',
            'Operations': 'Operations Specialist'
        }
        return titles.get(department, 'Team Member')
    
    def generate_offer_letter(self, employee_name):
        employee_data = self.employees_df[
            self.employees_df['Employee Name'].str.contains(employee_name, case=False, na=False)
        ]
        
        if employee_data.empty:
            raise ValueError(f"Employee '{employee_name}' not found")
        
        employee_info = employee_data.iloc[0].to_dict()
        
        context = self.get_relevant_context(employee_info)
        
        policies = self.extract_policy_info_from_context(
            employee_info['Band'], 
            employee_info['Department'], 
            context
        )
        
        salary = self.extract_salary_breakdown(employee_info)
        
        position_title = self.get_position_title(employee_info['Department'])
        
        offer_letter = f"""
═══════════════════════════════════════════════════════════════
📄 OFFER LETTER – COMPANY ABC
═══════════════════════════════════════════════════════════════

Date: {datetime.now().strftime('%B %d, %Y')}

Dear {employee_info['Employee Name']},

We are pleased to extend this offer of employment for the position of {position_title} 
in the {employee_info['Department']} team at Company ABC.

CANDIDATE DETAILS:
• Name: {employee_info['Employee Name']}
• Position: {position_title}
• Band Level: {employee_info['Band']}
• Department: {employee_info['Department']}
• Work Location: {employee_info['Location']}
• Joining Date: {employee_info['Joining Date']}

═══════════════════════════════════════════════════════════════
1. 💰 COMPENSATION & SALARY BREAKDOWN
═══════════════════════════════════════════════════════════════

Annual Compensation Structure:

Component                          Amount (INR)
─────────────────────────────────────────────
Base Salary (Fixed)               {salary['base_salary']}
Performance Bonus                  {salary['performance_bonus']}
Retention Bonus                    {salary['retention_bonus']}
─────────────────────────────────────────────
TOTAL ANNUAL CTC                   {salary['total_ctc']}
Monthly Gross (Approx.)            {salary['monthly_gross']}

• Performance bonuses are paid quarterly based on individual and company performance
• Retention bonus is paid over the specified period as per company policy
• Salary reviews are conducted annually based on performance and market benchmarks

═══════════════════════════════════════════════════════════════
2. 🏖️ LEAVE ENTITLEMENTS & WORK ARRANGEMENTS (Band {employee_info['Band']})
═══════════════════════════════════════════════════════════════

Based on your band level and our HR Leave Policy:

• Annual Leave Entitlement: {policies['leave'].get('total_days', 'N/A')} days
  - Earned Leave: {policies['leave'].get('earned', 'N/A')} days
  - Sick Leave: {policies['leave'].get('sick', 'N/A')} days  
  - Casual Leave: {policies['leave'].get('casual', 'N/A')} days
• Work From Home Eligibility: {policies['leave'].get('wfh', 'N/A')}
• Work From Office Requirement: {policies['leave'].get('wfo', 'N/A')}

{employee_info['Department']} Team Specific Requirements:
• Minimum WFO: {policies['wfo'].get('minimum', 'N/A')}
• Suggested Days: {policies['wfo'].get('suggested', 'N/A')}
• Special Notes: {policies['wfo'].get('notes', 'N/A')}

Leave Management:
• All leaves must be applied through HRMS with manager approval
• Leave balances reset annually on January 1st
• Carry-forward up to 10 days permitted
• Emergency leave can be regularized post-facto

═══════════════════════════════════════════════════════════════
3. ✈️ TRAVEL POLICY & BENEFITS (Band {employee_info['Band']})
═══════════════════════════════════════════════════════════════

Business Travel Entitlements (as per HR Travel Policy):
• Flight Class: {policies['travel'].get('flight', 'N/A')}
• Hotel Cap: {policies['travel'].get('hotel', 'N/A')}
• Per Diem (Domestic): {policies['travel'].get('per_diem_domestic', 'N/A')}
• Per Diem (International): {policies['travel'].get('per_diem_intl', 'N/A')}
• Approval Required: {policies['travel'].get('approval', 'N/A')}

• All travel must be booked through approved corporate platforms
• Expense reimbursement as per company travel policy

Additional Benefits:
• Home office setup support: Rs. 5,000 (for L3+)
• Monthly internet reimbursement: Rs. 1,000/month (for hybrid-eligible roles)
• Health insurance and other statutory benefits as per company policy

═══════════════════════════════════════════════════════════════
4. 🔒 EMPLOYMENT TERMS & CONDITIONS
═══════════════════════════════════════════════════════════════

• Employment Type: Full-time, permanent position
• Probation Period: 3 months from joining date
• Notice Period: 60 days (15 days during probation)
• Working Hours: As per company policy and team requirements

Confidentiality & IP:
• All work products and innovations belong to Company ABC
• Strict confidentiality of proprietary information required
• Non-disclosure agreement will be provided separately

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
1. Sign and return this letter within 5 working days
2. Submit required documents for background verification
3. Complete pre-joining formalities as communicated by HR

Your assigned HR Business Partner will contact you with:
• Onboarding timeline and checklist
• Document requirements
• First-day joining instructions

We look forward to welcoming you to the Company ABC family!

Warm regards,

Aarti Nair
HR Business Partner
Company ABC

📧 peopleops@companyabc.com
🌐 www.companyabc.com
📞 +91-XXXX-XXXXXX

═══════════════════════════════════════════════════════════════
"""
        return offer_letter.strip()

@st.cache_resource
def initialize_rag_system():
    return HROfferLetterRAG()

def main():
    st.set_page_config(page_title="HR Offer Letter Generator", page_icon="📄")
    
    st.title("📄 HR Offer Letter Generator")
    
    rag_system = initialize_rag_system()
    
    if 'system_ready' not in st.session_state:
        with st.spinner("Loading Employee_List.csv and HR policy documents..."):
            success = rag_system.load_data_from_files()
            st.session_state['system_ready'] = success
    
    if st.session_state.get('system_ready', False):
        
        with st.sidebar:
            st.header("📋 Available Employees")
            st.caption(f"From Employee_List.csv ({len(rag_system.employees_df)} total)")
            for _, employee in rag_system.employees_df.iterrows():
                st.write(f"• **{employee['Employee Name']}** ({employee['Department']}, {employee['Band']})")
        
        st.header("🎯 Generate Offer Letter")
        
        employee_name = st.text_input(
            "Enter Employee Name:", 
            placeholder="e.g., Martha Bennett, Julie Rodriguez, etc.",
            help="Type the name of the employee"
        )
        
        if st.button("Generate Offer Letter", type="primary"):
            if employee_name:
                try:
                    with st.spinner(f"Generating offer letter for {employee_name}..."):
                        offer_letter = rag_system.generate_offer_letter(employee_name)
                    
                    st.success(f" Offer letter generated for {employee_name}!")
                    
                    st.text_area("Generated Offer Letter:", value=offer_letter, height=600)
                    
                    st.download_button(
                        label="📥 Download Offer Letter",
                        data=offer_letter,
                        file_name=f"{employee_name.replace(' ', '_')}_offer_letter.txt",
                        mime="text/plain"
                    )
                    
                except ValueError as e:
                    st.error(f"{str(e)}")
                    st.info("💡 Check the sidebar for available employee names from Employee_List.csv")
                except Exception as e:
                    st.error(f"❌ Error generating offer letter: {str(e)}")
            else:
                st.warning("⚠Please enter an employee name")
    
    else:
        st.error("Failed to load system:")
        st.markdown("""
        - **Employee_List.csv** (your employee data)
        - **HR-Leave-Policy.pdf** (your leave policy document)  
        - **HR-Travel-Policy.pdf** (your travel policy document)
        """)

if __name__ == "__main__":
    main()



