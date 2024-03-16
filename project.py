from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re
from langdetect import detect
from pyarabic.araby import normalize_hamza, strip_tatweel, strip_tashkeel 

def preparethePDF():
  text = ""
  pdf_reader = PdfReader("HR.pdf")
  for page in pdf_reader.pages:
    text+=page.extract_text()
   
  text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=3000,
      chunk_overlap=400,
      length_function=len
    )
  chunks = text_splitter.split_text(text=text)

  preprocessed_chunks = [
    chunk if detect(chunk) != 'ar' else re.sub(r'[,\t\n\r\x0b\x0c]', ' ', strip_tatweel(strip_tashkeel(chunk))).strip()
    for chunk in chunks
  ]
  embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                    model_kwargs={'device': 'cpu'})
  
  vectorstore = FAISS.from_texts(
        texts=preprocessed_chunks,
        embedding=embeddings)  
   
  return vectorstore  

def process_question(user_question, vectorstore):
  with st.spinner('Please wait for response😊...'):     
    llm = ChatGoogleGenerativeAI(model='gemini-pro',
                   google_api_key=os.getenv("GOOGLE_API_KEY"),
                  temperature=0.5,
                  convert_system_message_to_human=True)
    prompt_template = """

            **انت خبير استشاري في الموارد البشرية**
            **السياق: **
            {context}
            **التعليمات: **
            سوف يقوم المستخدم بإستشارتك أو سؤالك عن شيءً متعلق في الموارد البشرية وانت بدورك يجب عليك مساعدته عن طريق إعطاء جواب صحيح وموجز .
            '.إذا لم تكن تعرف الإجابة فقل له 'إنك لا تعلم الإجابة، حاول أن تصيغ السؤال بطريقةٍ أخرى.

            إذا قام المستخدم بسؤالك عن شيء غير متعلق بالموارد البشرية فقل له إنك
            'يجب أن تقوم بسؤالي عن شيء متعلق بالموارد البشرية'.
            {question}
            **الإجابة: **

          """.strip()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    retriever = vectorstore.as_retriever()
    chain = (
          {"context": retriever, "question": RunnablePassthrough()}
          | prompt
          | llm
          | StrOutputParser()
        )
    response = chain.invoke(user_question)

    return response
   
def main():
  load_dotenv()
  st.set_page_config(page_title="Ask your AI assistant-HR consultant  🦜️🔗")
  st.markdown("<h1 class='title'>AI assistant-HR consultant</h1>", unsafe_allow_html=True)
  
  user_question = st.text_input("Ask your question")
  
  embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                    model_kwargs={'device': 'cpu'})
  if user_question:
    # vectorstore = preparethePDF()
    # Save the vector store to a file
     
    # vectorstore.save_local("vectorstore_V2.faiss")
    # Load the vector store from a file

    vectorstore = FAISS.load_local("vectorstore_V2.faiss",embeddings)
    response = process_question(user_question, vectorstore)
    st.write(response)


if __name__ == '__main__':
  main() 