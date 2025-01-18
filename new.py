from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM  # Updated import
import streamlit as st

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert in data science and helpful assistant. Your task is to provide clear, concise, and easy-to-understand responses, 
            which are also understandable by a five-year-old."""
        ),
        ('user', 'Question: {question}'),
    ]
)

# Streamlit Application
st.title("LLM First Project")
input_text = st.text_input("Search the topic you want")

# OllamaLLM Model Initialization
llm = OllamaLLM(model="llama3")  # Updated initialization
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Handle User Input
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
