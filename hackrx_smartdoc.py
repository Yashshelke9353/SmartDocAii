import streamlit as st
import fitz
import nltk
import re
import os
import logging
import json
from dotenv import load_dotenv
import requests  # Replace boto3 with requests
import tempfile
from typing import List, Dict, Optional, Any
import hashlib
from dataclasses import dataclass
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration settings"""
    TESSERACT_CMD: str = os.getenv('TESSERACT_CMD', 'tesseract')
    MAX_CHUNK_SIZE: int = int(os.getenv('MAX_CHUNK_SIZE', '4000'))
    MAX_FILE_SIZE: int = int(os.getenv('MAX_FILE_SIZE', '52428800'))  # 50MB
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')  # Your API key

# Safety-focused prompt prefix
SAFETY_PREFIX = """You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include 
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature. 
If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information."""

class GroqProcessor:
    """Groq API processor for generating responses"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Groq API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama3-8b-8192",  # Free model with good performance
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1024
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"API Error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return f"Error generating response: {str(e)}"

class InsuranceClaimsProcessor:
    """Enhanced document processor for insurance claims"""
    
    def __init__(self):
        self.config = Config()
        self.groq_processor = self._setup_groq_client()
        self._setup_nltk()
        self._setup_tesseract()
    
    def _setup_groq_client(self) -> Optional[GroqProcessor]:
        """Initialize Groq API client"""
        try:
            if not self.config.GROQ_API_KEY:
                st.error("Groq API key not found. Please set the API key.")
                return None
            
            return GroqProcessor(self.config.GROQ_API_KEY)
        except Exception as e:
            logger.error(f"Groq client setup failed: {e}")
            st.error(f"Failed to initialize Groq client: {e}")
            return None
    
    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    
    def _setup_tesseract(self):
    """Tesseract setup disabled for cloud deployment"""
    pass

    
    def safe_prompt(self, user_prompt: str) -> str:
        """Add safety prefix to all prompts"""
        return f"{SAFETY_PREFIX}\n\n{user_prompt}"
    
    def validate_file(self, uploaded_file) -> bool:
        """Validate uploaded PDF file"""
        if uploaded_file.size > self.config.MAX_FILE_SIZE:
            st.error("File too large. Please upload a file smaller than 50MB.")
            return False
        
        if not uploaded_file.name.lower().endswith('.pdf'):
            st.error("Invalid file type. Please upload a PDF file.")
            return False
        
        return True
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            st.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
    """OCR disabled for cloud deployment"""
    st.info("📝 OCR functionality disabled for cloud deployment. Using text extraction only.")
    return ""
    
    def preprocess_text(self, text: str) -> List[str]:
        """Clean and preprocess text"""
        try:
            # Clean text
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize into sentences
            sentences = nltk.sent_tokenize(text)
            return sentences
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return []
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Groq API"""
        if not self.groq_processor:
            return "Groq API client not available."
        
        try:
            safe_prompt_text = self.safe_prompt(prompt)
            return self.groq_processor.generate_response(safe_prompt_text)
            
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            st.error(f"Error generating response: {e}")
            return "Unable to generate response due to an error."
    
    # 🧠 NEW: Structured Query Parsing
    def extract_structured_info(self, user_query: str) -> Dict[str, Any]:
        """Extract structured information from user query"""
        prompt = f"""
        Extract structured data from the following user input for insurance claim processing.
        
        User Input: "{user_query}"
        
        Extract the following fields if mentioned:
        - age (number)
        - gender (male/female/other)
        - location (city/state/country)
        - procedure (medical procedure/treatment)
        - condition (medical condition)
        - hospital (hospital name)
        - doctor (doctor name)
        - date (date of treatment/procedure)
        - amount_claimed (monetary amount if mentioned)
        
        Respond ONLY in valid JSON format with the extracted fields. 
        If a field is not mentioned, omit it from the JSON.
        
        Example format:
        {{
            "age": 47,
            "gender": "female",
            "location": "Bangalore",
            "procedure": "gallbladder surgery"
        }}
        """
        
        try:
            response = self.generate_response(prompt)
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in structured info extraction response")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            st.error("Failed to parse structured information. Please try rephrasing your query.")
            return {}
        except Exception as e:
            logger.error(f"Structured info extraction failed: {e}")
            return {}
    
    # ⚖️ NEW: Decision-Making Engine
    def make_decision(self, user_data: Dict[str, Any], document_text: str) -> Dict[str, Any]:
        """Make automated approval/rejection decision"""
        prompt = f"""
        As an insurance claims processor, analyze the following user information against the insurance policy document to make a claim decision.
        
        User Claim Information:
        {json.dumps(user_data, indent=2)}
        
        Insurance Policy Document:
        {document_text[:4000]}
        
        Analysis Requirements:
        1. Check if the procedure/condition is covered by the policy
        2. Verify if the user meets eligibility criteria (age limits, location restrictions, etc.)
        3. Check for any exclusions that might apply
        4. Determine the coverage amount or percentage
        5. Look for any waiting periods or pre-conditions
        
        Respond ONLY in valid JSON format with these exact fields:
        {{
            "decision": "APPROVED" or "REJECTED",
            "amount": <number representing payout amount, 0 if rejected>,
            "coverage_percentage": <percentage of coverage, 0-100>,
            "justification": "<detailed explanation with specific policy clause references>",
            "policy_clauses": ["<list of relevant policy clauses>"],
            "exclusions_applied": ["<list of exclusions if any>"],
            "waiting_period": "<any waiting period information>",
            "next_steps": "<what the claimant should do next>"
        }}
        
        Be thorough in your analysis and provide specific references to policy terms.
        """
        
        try:
            response = self.generate_response(prompt)
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                decision_data = json.loads(json_str)
                return decision_data
            else:
                logger.warning("No JSON found in decision response")
                return {
                    "decision": "MANUAL_REVIEW",
                    "amount": 0,
                    "justification": "Unable to process automatically. Requires manual review."
                }
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed in decision making: {e}")
            return {
                "decision": "ERROR",
                "amount": 0,
                "justification": f"Processing error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            return {
                "decision": "ERROR",
                "amount": 0,
                "justification": f"System error: {str(e)}"
            }
    
    def get_file_hash(self, file_content: bytes) -> str:
        """Generate hash for file caching"""
        return hashlib.md5(file_content).hexdigest()

def main():
    """Main Streamlit application for HackRx 6.0"""
    st.set_page_config(
        page_title="SmartDoc - Insurance Claims Processor",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 SmartDoc: AI-Powered Insurance Claims Processor")
    st.markdown("**HackRx 6.0 Solution** - Upload your insurance policy document and submit your claim details for automated processing.")
    
    # Initialize processor
    processor = InsuranceClaimsProcessor()
    
    # Initialize session state
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    if 'claim_decision' not in st.session_state:
        st.session_state.claim_decision = None
    if 'user_structured_data' not in st.session_state:
        st.session_state.user_structured_data = None
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        1. **Upload** your insurance policy document (PDF)
        2. **Enter** your claim details in natural language
        3. **Get** automated claim decision with justification
        4. **Review** structured analysis and recommendations
        """)
        
        st.header("📝 Example Claim Input")
        st.markdown("""
        *"I'm a 35-year-old male from Mumbai who underwent knee replacement surgery at Apollo Hospital. The total cost was ₹2,50,000. My doctor is Dr. Sharma."*
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Insurance Policy Document")
        uploaded_file = st.file_uploader(
            "Upload your insurance policy document", 
            type="pdf",
            help="Upload the PDF containing your insurance policy terms and conditions"
        )
        
        if uploaded_file is not None:
            if not processor.validate_file(uploaded_file):
                return
            
            # Check if file already processed
            file_hash = processor.get_file_hash(uploaded_file.getvalue())
            
            if file_hash not in st.session_state.processed_files:
                with st.spinner('📖 Processing policy document...'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name
                    
                    # Extract text
                    pdf_text = processor.extract_text_from_pdf(temp_file_path)
                    ocr_text = processor.extract_text_with_ocr(temp_file_path)
                    full_text = pdf_text + "\n" + ocr_text
                    
                    # Preprocess
                    preprocessed_text = processor.preprocess_text(full_text)
                    
                    # Store in session
                    st.session_state.processed_files[file_hash] = {
                        'filename': uploaded_file.name,
                        'full_text': full_text,
                        'preprocessed_text': preprocessed_text
                    }
                    
                    # Cleanup
                    os.unlink(temp_file_path)
                    
                st.success(f"✅ Policy document '{uploaded_file.name}' processed successfully!")
            else:
                st.success(f"✅ Using cached policy document: {st.session_state.processed_files[file_hash]['filename']}")
    
    with col2:
        st.header("🗣️ Claim Details")
        
        if uploaded_file is not None and processor.groq_processor:
            # Get processed document
            file_hash = processor.get_file_hash(uploaded_file.getvalue())
            file_data = st.session_state.processed_files[file_hash]
            full_text = file_data['full_text']
            
            # User input for claim
            user_claim_input = st.text_area(
                "Describe your claim in natural language:",
                placeholder="e.g., I'm a 35-year-old male from Mumbai who underwent knee replacement surgery...",
                height=100
            )
            
            if st.button("🔍 Process Claim", type="primary"):
                if user_claim_input.strip():
                    with st.spinner('🧠 Analyzing claim...'):
                        # Step 1: Extract structured information
                        structured_data = processor.extract_structured_info(user_claim_input)
                        st.session_state.user_structured_data = structured_data
                        
                        # Step 2: Make decision
                        decision_data = processor.make_decision(structured_data, full_text)
                        st.session_state.claim_decision = decision_data
                    
                    st.success("✅ Claim analysis completed!")
                else:
                    st.error("Please enter your claim details.")
    
    # Results Section
    if st.session_state.claim_decision is not None and st.session_state.user_structured_data is not None:
        st.header("📊 Claim Analysis Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Decision", "📋 Structured Data", "📄 JSON Output", "💬 Chat"])
        
        with tab1:
            # Decision Display
            decision = st.session_state.claim_decision
            decision_type = decision.get('decision', 'UNKNOWN')
            
            if decision_type == 'APPROVED':
                st.success(f"✅ **CLAIM APPROVED**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💰 Payout Amount", f"₹{decision.get('amount', 0):,.2f}")
                with col2:
                    st.metric("📈 Coverage", f"{decision.get('coverage_percentage', 0)}%")
                with col3:
                    st.metric("⏱️ Status", "Approved")
            elif decision_type == 'REJECTED':
                st.error(f"❌ **CLAIM REJECTED**")
                st.metric("💰 Payout Amount", "₹0")
            else:
                st.warning(f"⚠️ **{decision_type}**")
            
            # Justification
            st.subheader("📝 Justification")
            st.write(decision.get('justification', 'No justification provided'))
            
            # Additional details
            if decision.get('policy_clauses'):
                st.subheader("📋 Relevant Policy Clauses")
                for clause in decision.get('policy_clauses', []):
                    st.write(f"• {clause}")
            
            if decision.get('exclusions_applied'):
                st.subheader("🚫 Exclusions Applied")
                for exclusion in decision.get('exclusions_applied', []):
                    st.write(f"• {exclusion}")
            
            if decision.get('next_steps'):
                st.subheader("➡️ Next Steps")
                st.info(decision.get('next_steps'))
        
        with tab2:
            # Structured Data Display
            st.subheader("🧠 Extracted Structured Information")
            structured_data = st.session_state.user_structured_data
            
            if structured_data:
                # Display in a nice format
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'age' in structured_data:
                        st.metric("👤 Age", structured_data['age'])
                    if 'gender' in structured_data:
                        st.metric("⚥ Gender", structured_data['gender'].title())
                    if 'location' in structured_data:
                        st.metric("📍 Location", structured_data['location'])
                
                with col2:
                    if 'procedure' in structured_data:
                        st.metric("🏥 Procedure", structured_data['procedure'])
                    if 'hospital' in structured_data:
                        st.metric("🏥 Hospital", structured_data['hospital'])
                    if 'amount_claimed' in structured_data:
                        st.metric("💰 Amount Claimed", f"₹{structured_data['amount_claimed']:,}")
                
                # Show all extracted data
                st.subheader("📊 Complete Extracted Data")
                st.json(structured_data)
            else:
                st.warning("No structured data was extracted from your input.")
        
        with tab3:
            # 🧾 JSON Output
            st.subheader("🧾 Complete JSON Output")
            
            complete_output = {
                "claim_input": user_claim_input,
                "structured_data": st.session_state.user_structured_data,
                "decision_output": st.session_state.claim_decision,
                "processing_timestamp": str(pd.Timestamp.now())
            }
            
            st.json(complete_output)
            
            # Download option
            if st.button("📥 Download Results as JSON"):
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(complete_output, indent=2),
                    file_name=f"claim_analysis_{hash(user_claim_input)}.json",
                    mime="application/json"
                )
        
        with tab4:
            # Chat functionality
            st.subheader("💬 Ask Questions About Your Claim")
            chat_input = st.text_input("Ask a question about your claim or policy:")
            
            if chat_input and st.button("Send"):
                with st.spinner('Generating response...'):
                    chat_prompt = f"""
                    Based on the insurance policy document and the claim analysis, answer the user's question.
                    
                    Policy Document: {full_text[:3000]}
                    
                    User's Claim Data: {json.dumps(st.session_state.user_structured_data, indent=2)}
                    
                    Claim Decision: {json.dumps(st.session_state.claim_decision, indent=2)}
                    
                    User's Question: {chat_input}
                    
                    Provide a helpful and accurate answer based on the policy and claim information.
                    """
                    
                    response = processor.generate_response(chat_prompt)
                    st.write("**Response:**")
                    st.write(response)
    
    # Footer
    st.markdown("---")
    st.markdown("**HackRx 6.0 - Team SmartDoc** | AI-Powered Insurance Claims Processing | Built with Streamlit & Groq API")

if __name__ == "__main__":
    main()