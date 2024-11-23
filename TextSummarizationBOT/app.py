import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Streamlit config
st.set_page_config(page_title="Medical Entity Extractor", layout="wide")

# Check if the API key is loaded
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please check your environment variables.")
    st.stop()

# Initialize ChatGroq instance with proper error handling
try:
    llm = ChatGroq(
        api_key=groq_api_key,
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=500,
        timeout=10,
        max_retries=2
    )
except Exception as e:
    st.error(f"Error initializing ChatGroq: {e}")
    st.stop()

# Define the prompt template with more structured output format
template = """You are a medical expert and good at English. Extract all the medical entities from the given tweet and assign the appropriate medical entity labels.

Input text: {text}

Please format your response as a markdown table with the following columns:
| Entity | Label | Context |

Only include medical terms, conditions, symptoms, medications, or healthcare-related entities.
"""

prompt = PromptTemplate(
    input_variables=['text'],
    template=template
)

# Streamlit UI
st.title("Medical Entity Extractor")
st.write("This tool extracts medical entities from text using the Groq LLM API.")

# Input text from the user
input_text = st.text_area(
    "Enter the text to analyze:",
    placeholder="Type your tweet or text here...",
    height=150
)

if st.button("Analyze"):
    if input_text.strip():
        # Show loading spinner
        with st.spinner("Analyzing text..."):
            try:
                # Format the prompt
                formatted_prompt = prompt.format(text=input_text)
                
                # Create a message object
                messages = [HumanMessage(content=formatted_prompt)]
                
                # Send the prompt to the model and get response
                response = llm.invoke(messages)
                
                # Extract content from the response
                if hasattr(response, 'content'):
                    result = response.content
                else:
                    result = str(response)
                
                # Display results
                st.subheader("Extracted Medical Entities")
                st.markdown(result)  # Using markdown to properly render the table
                
                # Add download button for results
                st.download_button(
                    label="Download Results",
                    data=result,
                    file_name="medical_entities.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error processing text: {str(e)}")
                st.info("Please try again or check your API configuration.")
    else:
        st.warning("Please enter some text to analyze.")