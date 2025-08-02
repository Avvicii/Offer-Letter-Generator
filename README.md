An HR automation tool that generates personalised offer letters using Retrieval-Augmented Generation (RAG). The system parses HR policy documents and employee data to create contextually accurate offer letters with appropriate salary breakdowns, leave entitlements, and travel policies based on employee band and department.

## Features

- **RAG-Powered Intelligence**: Uses semantic search to extract relevant policy information from HR documents
- **Document Parsing**: Automatically processes PDF policy documents and CSV employee data
- **Smart Salary Breakdown**: Calculates and displays comprehensive compensation structures
- **Band-Specific Policies**: Applies appropriate leave and travel policies based on employee band (L1-L5)
- **Department-Specific Rules**: Incorporates team-specific WFO and operational requirements
- **Streamlit Web UI**: Clean, intuitive interface for generating offer letters
- **Export Functionality**: Download generated letters as text files
- **Real-time Generation**: Instant offer letter creation with employee name input

## Technology Stack

- **Backend**: Python, LangChain, FAISS Vector Store
- **Frontend**: Streamlit
- **Document Processing**: PyPDF2 for PDF parsing
- **Data Processing**: Pandas for CSV handling
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Search**: FAISS for semantic similarity search


