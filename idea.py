# import streamlit as st
# from streamlit_chat import message
# import tempfile
# from langchain.document_loaders.csv_loader import CSVLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import CTransformers
# from langchain.chains import ConversationalRetrievalChain
# from langchain_google_genai import ChatGoogleGenerativeAI
# import os
# from dotenv import load_dotenv
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.prompts import ChatPromptTemplate

# from langchain.prompts import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     MessagesPlaceholder,
#     SystemMessagePromptTemplate
# )
  
# def main():
#      load_dotenv()
#      st.title("Ask your AI assistant-HR consultant  🦜️🔗")
     
#      embeddings = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1",
#                     model_kwargs={'device': 'cpu'})
#      db = FAISS.load_local("vectorstore.faiss",embeddings)
     
#      llm = ChatGoogleGenerativeAI(model='gemini-pro',
#                    google_api_key=os.getenv("GOOGLE_API_KEY"),
#                   temperature=0.6,
#                   convert_system_message_to_human=True)
     
#      general_system_template =  """

#                                     **انت خبير استشاري في الموارد البشرية**
#                                     **السياق: **
#                                     {context}
#                                     **التعليمات: **
#                                     سوف يقوم المستخدم بإستشارتك أو سؤالك عن شيءً متعلق في الموارد البشرية وانت بدورك يجب عليك مساعدته عن طريق إعطاء جواب صحيح وشامل ودقيق .
#                                     '.إذا لم تكن تعرف الإجابة فقل له 'إنك لا تعلم الإجابة، حاول أن تصيغ السؤال بطريقةٍ أخرى.

#                                     إذا قام المستخدم بسؤالك عن شيء غير متعلق بالموارد البشرية فقل له إنك
#                                     'يجب أن تقوم بسؤالي عن شيء متعلق بالموارد البشرية'.

#                                     """
#      general_user_template = "Question:```{question}```"
     
#      messages = [
#                 SystemMessagePromptTemplate.from_template(general_system_template),
#                 HumanMessagePromptTemplate.from_template(general_user_template)
#     ]
#      qa_prompt = ChatPromptTemplate.from_messages( messages )
     
#      chain= ConversationalRetrievalChain.from_llm(
#                 llm,
#                 retriever=db.as_retriever(),
#                 chain_type="stuff",
#                 verbose=True,
#                 combine_docs_chain_kwargs={'prompt': qa_prompt}
#             ) 
#      def conversational_chat(query):
#             result = chain({"question": query, "chat_history": st.session_state['history']})
#             st.session_state['history'].append((query, result["answer"]))
#             return result["answer"]

#         # Initialize chat history
#      if 'history' not in st.session_state:
#             st.session_state['history'] = []

#         # Initialize messages
#      if 'generated' not in st.session_state:
#             st.session_state['generated'] = ["Hello ! Ask me about " + "HR" + " 🤗"]

#      if 'past' not in st.session_state:
#             st.session_state['past'] = ["Hey ! 👋"]
            
          

#         # Create containers for chat history and user input
#      response_container = st.container()
#      container = st.container()

#         # User input form
#      with container:
        
#         with st.form(key='my_form', clear_on_submit=True):
#             user_input = st.text_input("Query:", placeholder="Talk with HR AI 👉 (:", key='input')
#             submit_button = st.form_submit_button(label='Send')

#         if submit_button and user_input:
#             with st.spinner('Wait for response...'):
#                 output = conversational_chat(user_input)
#                 st.session_state['past'].append(user_input)
#                 st.session_state['generated'].append(output)

#         # Display chat history
#         if st.session_state['generated']:
#             with response_container:
#                 for i in range(len(st.session_state['generated'])):
#                     message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
#                     message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

# if __name__ == '__main__':
#   main() 

import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)
  
def main():
     load_dotenv()
     st.title("Ask your AI assistant-HR consultant  🦜️🔗")
     
     embeddings = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1",
                    model_kwargs={'device': 'cpu'})
     db = FAISS.load_local("vectorstore.faiss",embeddings)
     
     llm = ChatGoogleGenerativeAI(model='gemini-pro',
                   google_api_key=os.getenv("GOOGLE_API_KEY"),
                  temperature=0.6,
                  convert_system_message_to_human=True)
     
     general_system_template =  """

                                **انت خبير استشاري في الموارد البشرية**
                                **السياق: **
                                
                                    {context}
                                
                                **التعليمات: **

                                سيقوم المستخدم بطرح سؤال أو طلب استشارة حول موضوع متعلق بالموارد البشرية.
                                دورك هو مساعدته من خلال تقديم إجابة صحيحة وشاملة ودقيقة.
                                إذا لم تكن متأكدًا من الإجابة، أخبر المستخدم بذلك وحاول إعادة صياغة السؤال.
                                
                                **قواعد الرد : **
                                إذا كان السؤال باللغة العربية، يجب أن تكون الإجابة باللغة العربية.
                                إذا كان السؤال لا يتعلق بالموارد البشرية، أخبر المستخدم أن عليه طرح سؤال متعلق بهذا المجال.

                                    """
     general_user_template = "Question:```{question}```"
     
     messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
     qa_prompt = ChatPromptTemplate.from_messages( messages )
     
     chain= ConversationalRetrievalChain.from_llm(
                llm,
                retriever=db.as_retriever(),
                chain_type="stuff",
                verbose=True,
                combine_docs_chain_kwargs={'prompt': qa_prompt}
            ) 
     def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        # Initialize chat history
     if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Create containers for chat history and user input
     response_container = st.container()
     container = st.container()

        # User input form
     with container:
        
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk with HR AI 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Wait for response...'):
                output = conversational_chat(user_input)

        # Display chat history
        if st.session_state['history']:
            with response_container:
                for i in range(len(st.session_state['history'])):
                    message(st.session_state['history'][i][0], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state['history'][i][1], key=str(i), avatar_style="thumbs")

if __name__ == '__main__':
  main() 
