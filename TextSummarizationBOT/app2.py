import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize Streamlit config with improved metadata
st.set_page_config(
    page_title="Entity Extractor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved prompt template with clearer instructions and examples
PROMPT_TEMPLATE = """You are an expert in entity extraction and natural language processing. 
Extract named entities from the given text, focusing on people, organizations, locations, and associated geographic information.

Input text: {text}

Provide a structured analysis in the following markdown table format:
| Name | Entity_Type | City | Country | Country_Code |

Guidelines:
- Name: The extracted entity name
- Entity_Type: One of [PERSON, ORGANIZATION, LOCATION]
- City: Associated city (if applicable)
- Country: Full country name (if applicable)
- Country_Code: ISO 2-letter country code (if applicable)

Example:
For text: "Tim Cook from Apple in Cupertino, USA announced..."
| Tim Cook | PERSON | Cupertino | United States | US |
| Apple | ORGANIZATION | Cupertino | United States | US |

Only include relevant named entities. Leave fields blank with '-' if not applicable.
Ensure all country codes are valid ISO 2-letter codes."""

class EntityExtractor:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the Streamlit UI components"""
        st.title("🔍 Advanced Entity Extractor")
        st.markdown("""
        This tool extracts named entities from text and identifies associated geographic information.
        Enter your text below to analyze people, organizations, and locations.
        """)
        
    def initialize_llm(self):
        """Initialize the ChatGroq LLM with error handling"""
        try:
            return ChatGroq(
                api_key=self.api_key,
                model="mixtral-8x7b-32768",
                temperature=0,
                max_tokens=1000,
                timeout=15,
                max_retries=3
            )
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            return None
            
    def process_text(self, input_text):
        """Process the input text and extract entities"""
        if not self.api_key:
            st.error("⚠️ GROQ_API_KEY not found. Please check your environment variables.")
            return
            
        llm = self.initialize_llm()
        if not llm:
            return
            
        try:
            # Create prompt
            prompt = PromptTemplate(
                input_variables=['text'],
                template=PROMPT_TEMPLATE
            )
            formatted_prompt = prompt.format(text=input_text)
            
            # Process with LLM
            with st.spinner("🔄 Analyzing text..."):
                start_time = time.time()
                messages = [HumanMessage(content=formatted_prompt)]
                response = llm.invoke(messages)
                processing_time = time.time() - start_time
                
                # Extract and display results
                result = response.content if hasattr(response, 'content') else str(response)
                
                # Display results in a nice format
                st.subheader("📊 Extracted Entities")
                st.markdown(result)
                st.info(f"⚡ Processing completed in {processing_time:.2f} seconds")
                
                # Add download functionality
                st.download_button(
                    label="📥 Download Results",
                    data=result,
                    file_name="extracted_entities.md",
                    mime="text/markdown"
                )
                
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.info("Please try again or check your input text.")

def main():
    extractor = EntityExtractor()
    
    # Input area with improved UX
    input_text = st.text_area(
        "Enter text to analyze:",
        placeholder="Example: Tim Cook from Apple in Cupertino, USA announced...",
        height=150,
        key="input_area"
    )
    
    # Add a clear button
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.session_state.input_area = ""
        st.experimental_rerun()
        
    if analyze_button and input_text.strip():
        extractor.process_text(input_text)
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze.")

if __name__ == "__main__":
    main()