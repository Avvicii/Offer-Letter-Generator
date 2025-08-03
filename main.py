import sys
import importlib
import pysqlite3
sys.modules["sqlite3"] = importlib.import_module("pysqlite3")


import streamlit as st
import pandas as pd
from datetime import datetime
import PyPDF2
from io import StringIO
import chromadb
from sentence_transformers import SentenceTransformer

class HROfferLetterRAG:
    def __init__(self):
        self.employees_df = None
        self.collection = None
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        
    def parse_pdf(self, filepath):
        """Parse PDF file and return text"""
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
    
    def chunk_text(self, text, chunk_size=800, chunk_overlap=100):
        """Simple text chunking"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
            if start >= len(text):
                break
        return chunks
    
    def load_data_from_files(self):
        """Load data from local files"""
        try:
            # Load employee data
            self.employees_df = pd.read_csv("Employee_List.csv")
            st.success(f"✅ Loaded {len(self.employees_df)} employees from Employee_List.csv")
            
            # Parse policy documents
            leave_policy_text = self.parse_pdf("HR-Leave-Policy.pdf")
            travel_policy_text = self.parse_pdf("HR-Travel-Policy.pdf")
            
            if not leave_policy_text or not travel_policy_text:
                st.error("Failed to parse policy documents")
                return False
            
            # Create document chunks
            leave_chunks = self.chunk_text(leave_policy_text)
            travel_chunks = self.chunk_text(travel_policy_text)
            
            all_chunks = leave_chunks + travel_chunks
            chunk_ids = [f"chunk_{i}" for i in range(len(all_chunks))]
            metadatas = [{"source": "leave_policy"}] * len(leave_chunks) + [{"source": "travel_policy"}] * len(travel_chunks)
            
            # Create ChromaDB collection
            try:
                self.client.delete_collection("hr_policies")
            except:
                pass
                
            self.collection = self.client.create_collection("hr_policies")
            self.collection.add(
                documents=all_chunks,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            st.success(f"✅ Created {len(all_chunks)} document chunks from HR policies")
            return True
            
        except FileNotFoundError as e:
            st.error(f"File not found: {str(e)}")
            st.info("Please ensure these files are in the same directory:")
            st.markdown("- Employee_List.csv\n- HR-Leave-Policy.pdf\n- HR-Travel-Policy.pdf")
            return False
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def get_relevant_context(self, employee_data):
        """Retrieve relevant policy context using ChromaDB"""
        query = f"band {employee_data['Band']} department {employee_data['Department']} leave policy travel policy salary benefits"
        
        results = self.collection.query(
            query_texts=[query],
            n_results=5
        )
        
        return "\n\n".join(results['documents'][0])
    
    def extract_policy_info_from_context(self, band, department, context):
        """Extract policy information from context"""
        # Band-specific policies
        band_policies = {
            'L1': {'total_days': 12, 'earned': 6, 'sick': 4, 'casual': 2, 'wfh': 'Limited', 'wfo': '4 days/week minimum'},
            'L2': {'total_days': 15, 'earned': 8, 'sick': 5, 'casual': 2, 'wfh': 'Partial', 'wfo': '3-4 days/week'},
            'L3': {'total_days': 18, 'earned': 10, 'sick': 6, 'casual': 2, 'wfh': 'Yes', 'wfo': '3 days/week minimum'},
            'L4': {'total_days': 20, 'earned': 12, 'sick': 6, 'casual': 2, 'wfh': 'Yes', 'wfo': '2-3 days/week'},
            'L5': {'total_days': 'Unlimited (with approval)', 'earned': 'NA', 'sick': 'NA', 'casual': 'NA', 'wfh': 'Full Flex', 'wfo': '0-2 days/week (optional)'}
        }
        
        travel_policies = {
            'L1': {'flight': 'Economy (on approval)', 'hotel': 'Rs. 2,000/night', 'per_diem_domestic': 'Rs. 1,500/day', 'per_diem_intl': 'USD 30/day', 'approval': 'Manager + VP'},
            'L2': {'flight': 'Economy (>6hrs)', 'hotel': 'Rs. 3,000/night', 'per_diem_domestic': 'Rs. 2,000/day', 'per_diem_intl': 'USD 40/day', 'approval': 'Manager + Director'},
            'L3': {'flight': 'Economy standard', 'hotel': 'Rs. 4,000/night', 'per_diem_domestic': 'Rs. 3,000/day', 'per_diem_intl': 'USD 60/day', 'approval': 'Reporting Manager'},
            'L4': {'flight': 'Premium Economy', 'hotel': 'Rs. 6,000/night', 'per_diem_domestic': 'Rs. 4,500/day', 'per_diem_intl': 'USD 80/day', 'approval': 'VP'},
            'L5': {'flight': 'Business Class', 'hotel': 'Rs. 10,000/night', 'per_diem_domestic': 'Rs. 7,500/day', 'per_diem_intl': 'USD 120/day', 'approval': 'None'}
        }
        
        wfo_policies = {
            'Engineering': {'minimum': '3 days/week', 'suggested': 'Mon, Tue, Thu', 'notes': 'Sprint reviews must be in-office'},
            'Sales': {'minimum': '4-5 days/week', 'suggested': 'Field visits + office', 'notes': 'Remote only with RSM approval'},
            'HR': {'minimum': '4 days/week', 'suggested': 'Mon-Thu', 'notes': 'In-office mandatory during onboarding'},
            'Finance': {'minimum': '3 days/week', 'suggested': 'Tue, Wed, Fri', 'notes': 'Fully in-office during month-end'},
            'Operations': {'minimum': '5 days/week', 'suggested': 'All weekdays', 'notes': 'WFH not permitted except in emergencies'}
        }
        
        return {
            'leave': band_policies.get(band, {}),
            'travel': travel_policies.get(band, {}),
            'wfo': wfo_policies.get(department, {})
        }
    
    def get_position_title(self, department):
        """Get position title based on department"""
        titles = {
            'Engineering': 'Software Engineer',
            'Sales': 'Sales Executive', 
            'HR': 'HR Specialist',
            'Finance': 'Financial Analyst',
            'Operations': 'Operations Specialist'
        }
        return titles.get(department, 'Team Member')
    
    def extract_salary_breakdown(self, employee_info):
        """Extract and format salary information"""
        return {
            'base_salary': f"₹{employee_info['Base Salary (INR)']:,}",
            'performance_bonus': f"₹{employee_info['Performance Bonus (INR)']:,}",
            'retention_bonus': f"₹{employee_info['Retention Bonus (INR)']:,}",
            'total_ctc': f"₹{employee_info['Total CTC (INR)']:,}",
            'monthly_gross': f"₹{employee_info['Total CTC (INR)'] // 12:,}",
        }
    
    def generate_offer_letter(self, employee_name):
        """Generate offer letter using parsed documents and RAG"""
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
OFFER LETTER – COMPANY ABC
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
1. COMPENSATION & SALARY BREAKDOWN
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

═══════════════════════════════════════════════════════════════
2. LEAVE ENTITLEMENTS & WORK ARRANGEMENTS (Band {employee_info['Band']})
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

═══════════════════════════════════════════════════════════════
3. TRAVEL POLICY & BENEFITS (Band {employee_info['Band']})
═══════════════════════════════════════════════════════════════

Business Travel Entitlements:
• Flight Class: {policies['travel'].get('flight', 'N/A')}
• Hotel Cap: {policies['travel'].get('hotel', 'N/A')}
• Per Diem (Domestic): {policies['travel'].get('per_diem_domestic', 'N/A')}
• Per Diem (International): {policies['travel'].get('per_diem_intl', 'N/A')}
• Approval Required: {policies['travel'].get('approval', 'N/A')}

═══════════════════════════════════════════════════════════════
4. NEXT STEPS
═══════════════════════════════════════════════════════════════

Please confirm your acceptance by signing and returning this letter within 5 working days.

Warm regards,

Aarti Nair
HR Business Partner
Company ABC

peopleops@companyabc.com
www.companyabc.com

═══════════════════════════════════════════════════════════════
"""
        return offer_letter.strip()

@st.cache_resource
def initialize_rag_system():
    return HROfferLetterRAG()

def main():
    st.set_page_config(page_title="HR Offer Letter Generator", page_icon="📄")
    
    st.title("HR Offer Letter Generator")
    st.markdown("Generate personalized offer letters using your HR documents and employee data")
    
    rag_system = initialize_rag_system()
    
    if 'system_ready' not in st.session_state:
        with st.spinner("Loading Employee_List.csv and processing HR policy documents..."):
            success = rag_system.load_data_from_files()
            st.session_state['system_ready'] = success
    
    if st.session_state.get('system_ready', False):
        with st.sidebar:
            st.header("Available Employees")
            st.caption(f"From Employee_List.csv ({len(rag_system.employees_df)} total)")
            for _, employee in rag_system.employees_df.iterrows():
                st.write(f"• **{employee['Employee Name']}** ({employee['Department']}, {employee['Band']})")
        
        st.header("Generate Offer Letter")
        
        employee_name = st.text_input(
            "Enter Employee Name:", 
            placeholder="e.g., Martha Bennett, Julie Rodriguez, etc.",
            help="Type the full name or part of the name from your Employee_List.csv"
        )
        
        if st.button("Generate Offer Letter", type="primary"):
            if employee_name:
                try:
                    with st.spinner(f"Generating offer letter for {employee_name}..."):
                        offer_letter = rag_system.generate_offer_letter(employee_name)
                    
                    st.success(f"✅ Offer letter generated for {employee_name}!")
                    
                    st.text_area("Generated Offer Letter:", value=offer_letter, height=600)
                    
                    st.download_button(
                        label="Download Offer Letter",
                        data=offer_letter,
                        file_name=f"{employee_name.replace(' ', '_')}_offer_letter.txt",
                        mime="text/plain"
                    )
                    
                except ValueError as e:
                    st.error(f"❌ {str(e)}")
                    st.info("💡 Check the sidebar for available employee names")
                except Exception as e:
                    st.error(f"❌ Error generating offer letter: {str(e)}")
            else:
                st.warning("Please enter an employee name")
    
    else:
        st.error("❌ Failed to load system. Please check if these files exist:")
        st.markdown("- Employee_List.csv\n- HR-Leave-Policy.pdf\n- HR-Travel-Policy.pdf")

if __name__ == "__main__":
    main()

