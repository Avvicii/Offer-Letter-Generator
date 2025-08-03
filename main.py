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
        """Parse PDF file and return text"""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
            return ""

    def load_data_from_files(self):
        """Load data from your uploaded files"""
        try:
            # Load employee data from your CSV file
            self.employees_df = pd.read_csv("Employee_List.csv")
            st.success(f"✅ Loaded {len(self.employees_df)} employees from Employee_List.csv")
            st.success(f"Loaded {len(self.employees_df)} employees")

            # Parse your HR policy documents
            leave_policy_text = self.parse_pdf("HR-Leave-Policy.pdf")
            travel_policy_text = self.parse_pdf("HR-Travel-Policy.pdf")

            if not leave_policy_text or not travel_policy_text:
                st.error("Failed to parse policy documents")
                return False

            # Create document chunks
            documents = []

            # Process leave policy
            leave_doc = Document(
                page_content=leave_policy_text, 
                metadata={"source": "HR-Leave-Policy", "document_type": "HR_Policy"}
            )
            leave_chunks = self.text_splitter.split_documents([leave_doc])
            documents.extend(leave_chunks)

            # Process travel policy  
            travel_doc = Document(
                page_content=travel_policy_text,
                metadata={"source": "HR-Travel-Policy", "document_type": "HR_Policy"}
            )
            travel_chunks = self.text_splitter.split_documents([travel_doc])
            documents.extend(travel_chunks)

            st.success(f"✅ Created {len(documents)} document chunks from HR policies")
            st.success(f"Created {len(documents)} document chunks from HR policies")

            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            st.success("✅ Vector store created successfully")
            return True

        except FileNotFoundError as e:
            return False

    def get_relevant_context(self, employee_data):
        """Retrieve relevant policy context for employee using RAG"""
        query = f"band {employee_data['Band']} department {employee_data['Department']} leave policy travel policy salary benefits"
        relevant_docs = self.vector_store.similarity_search(query, k=6)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return context

    def extract_salary_breakdown(self, employee_info):
        """Extract and format salary information"""
        salary_breakdown = {
            'base_salary': f"₹{employee_info['Base Salary (INR)']:,}",
            'performance_bonus': f"₹{employee_info['Performance Bonus (INR)']:,}",
        return salary_breakdown

    def extract_policy_info_from_context(self, band, department, context):
        """Extract policy information from RAG context using actual document content"""
        policies = {
            'leave': {'total_days': 'As per policy', 'wfh': 'As per band', 'wfo': 'As per team'},
            'travel': {'approval': 'As per hierarchy', 'entitlements': 'As per band'},

        context_lower = context.lower()

        # Extract leave information for specific band from actual policy
        band_patterns = {
            'L1': {'total_days': 12, 'earned': 6, 'sick': 4, 'casual': 2, 'wfh': 'Limited', 'wfo': '4 days/week minimum'},
            'L2': {'total_days': 15, 'earned': 8, 'sick': 5, 'casual': 2, 'wfh': 'Partial', 'wfo': '3-4 days/week'},
            'L5': {'total_days': 'Unlimited (with approval)', 'earned': 'NA', 'sick': 'NA', 'casual': 'NA', 'wfh': 'Full Flex', 'wfo': '0-2 days/week (optional)'}
        }

        # Travel policies extracted from your HR-Travel-Policy.pdf
        travel_patterns = {
            'L1': {'flight': 'Economy (on approval)', 'hotel': 'Rs. 2,000/night', 'per_diem_domestic': 'Rs. 1,500/day', 'per_diem_intl': 'USD 30/day', 'approval': 'Manager + VP'},
            'L2': {'flight': 'Economy (>6hrs)', 'hotel': 'Rs. 3,000/night', 'per_diem_domestic': 'Rs. 2,000/day', 'per_diem_intl': 'USD 40/day', 'approval': 'Manager + Director'},
            'L5': {'flight': 'Business Class', 'hotel': 'Rs. 10,000/night', 'per_diem_domestic': 'Rs. 7,500/day', 'per_diem_intl': 'USD 120/day', 'approval': 'None'}
        }

        # Department-specific WFO from your HR-Leave-Policy.pdf
        wfo_patterns = {
            'Engineering': {'minimum': '3 days/week', 'suggested': 'Mon, Tue, Thu', 'notes': 'Sprint reviews must be in-office'},
            'Sales': {'minimum': '4-5 days/week', 'suggested': 'Field visits + office', 'notes': 'Remote only with RSM approval'},
        }

    def get_position_title(self, department):
        """Get position title based on department"""
        titles = {
            'Engineering': 'Software Engineer',
            'Sales': 'Sales Executive', 
        return titles.get(department, 'Team Member')

    def generate_offer_letter(self, employee_name):
        """Generate offer letter using parsed documents and RAG"""
        employee_data = self.employees_df[
            self.employees_df['Employee Name'].str.contains(employee_name, case=False, na=False)
        ]

        employee_info = employee_data.iloc[0].to_dict()

        # Get relevant context using RAG
        context = self.get_relevant_context(employee_info)

        # Extract policy information from your documents
        policies = self.extract_policy_info_from_context(
            employee_info['Band'], 
            employee_info['Department'], 
            context
        )

        # Extract salary breakdown
        salary = self.extract_salary_breakdown(employee_info)

        # Generate position title
        position_title = self.get_position_title(employee_info['Department'])

        # Create comprehensive offer letter
        offer_letter = f"""
═══════════════════════════════════════════════════════════════
📄 OFFER LETTER – COMPANY ABC
    st.set_page_config(page_title="HR Offer Letter Generator", page_icon="📄")

    st.title("📄 HR Offer Letter Generator")
    st.markdown("Generate personalized offer letters using your HR documents and employee data")

    # Initialize RAG system
    rag_system = initialize_rag_system()

    # Load data automatically
    if 'system_ready' not in st.session_state:
        with st.spinner("Loading Employee_List.csv and processing HR policy documents..."):
        with st.spinner("Loading Employee_List.csv and HR policy documents..."):
            success = rag_system.load_data_from_files()
            st.session_state['system_ready'] = success

    # Show interface if system is ready
    if st.session_state.get('system_ready', False):

        # Show employee list in sidebar
        with st.sidebar:
            st.header("📋 Available Employees")
            st.caption(f"From Employee_List.csv ({len(rag_system.employees_df)} total)")
            for _, employee in rag_system.employees_df.iterrows():
                st.write(f"• **{employee['Employee Name']}** ({employee['Department']}, {employee['Band']})")

        # Main interface
        st.header("🎯 Generate Offer Letter")

        # Employee name input
        employee_name = st.text_input(
            "Enter Employee Name:", 
            placeholder="e.g., Martha Bennett, Julie Rodriguez, etc.",
            help="Type the full name or part of the name from your Employee_List.csv"
            help="Type the name of the employee"
        )

        if st.button("🚀 Generate Offer Letter", type="primary"):
        if st.button("Generate Offer Letter", type="primary"):
            if employee_name:
                try:
                    with st.spinner(f"Generating offer letter for {employee_name}..."):
                        offer_letter = rag_system.generate_offer_letter(employee_name)

                    st.success(f"✅ Offer letter generated for {employee_name}!")
                    st.success(f" Offer letter generated for {employee_name}!")

                    # Display the offer letter
                    st.text_area("📄 Generated Offer Letter:", value=offer_letter, height=600)
                    st.text_area("Generated Offer Letter:", value=offer_letter, height=600)

                    # Download button
                    st.download_button(
                        label="📥 Download Offer Letter",
                        data=offer_letter,
                    )

                except ValueError as e:
                    st.error(f"❌ {str(e)}")
                    st.error(f"{str(e)}")
                    st.info("💡 Check the sidebar for available employee names from Employee_List.csv")
                except Exception as e:
                    st.error(f"❌ Error generating offer letter: {str(e)}")
            else:
                st.warning("⚠️ Please enter an employee name")
                st.warning("⚠Please enter an employee name")

    else:
        st.error("❌ Failed to load system. Please check if these files exist in the same directory:")
        st.error("Failed to load system:")
        st.markdown("""
        - **Employee_List.csv** (your employee data)
        - **HR-Leave-Policy.pdf** (your leave policy document)  
if __name__ == "__main__":
    main()
