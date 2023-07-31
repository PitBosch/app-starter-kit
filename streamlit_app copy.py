import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat
st.set_page_config(page_title="HugChat - An LLM-powered Streamlit app")

st.title('🎈 App Name')

st.write('Hello world!')

import pdfplumber
from hugchat import hugchat
from hugchat.login import Login
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import random
import string
import shutil


# DOCUMENTS PLUGIN
        if st.session_state['plugin'] == "📝 Talk with your DOCUMENTS" and 'documents' not in st.session_state:
            with st.expander("📝 Talk with your DOCUMENT", expanded=True):  
                upload_pdf = st.file_uploader("Upload your DOCUMENT", type=['txt', 'pdf', 'docx'], accept_multiple_files=True)
                if upload_pdf is not None and st.button('📝✅ Load Documents'):
                    documents = []
                    with st.spinner('🔨 Reading documents...'):
                        for upload_pdf in upload_pdf:
                            print(upload_pdf.type)
                            if upload_pdf.type == 'text/plain':
                                documents += [upload_pdf.read().decode()]
                            elif upload_pdf.type == 'application/pdf':
                                with pdfplumber.open(upload_pdf) as pdf:
                                    documents += [page.extract_text() for page in pdf.pages]
                            elif upload_pdf.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                                text = docx2txt.process(upload_pdf)
                                documents += [text]
                    st.session_state['documents'] = documents
                    # Split documents into chunks
                    with st.spinner('🔨 Creating vectorstore...'):
                        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                        texts = text_splitter.create_documents(documents)
                        # Select embeddings
                        embeddings = st.session_state['hf']
                        # Create a vectorstore from documents
                        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                        db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db_" + random_str)

                    with st.spinner('🔨 Saving vectorstore...'):
                        # save vectorstore
                        db.persist()
                        #create .zip file of directory to download
                        shutil.make_archive("./chroma_db_" + random_str, 'zip', "./chroma_db_" + random_str)
                        # save in session state and download
                        st.session_state['db'] = "./chroma_db_" + random_str + ".zip" 
                    
                    with st.spinner('🔨 Creating QA chain...'):
                        # Create retriever interface
                        retriever = db.as_retriever()
                        # Create QA chain
                        qa = RetrievalQA.from_chain_type(llm=st.session_state['LLM'], chain_type='stuff', retriever=retriever,  return_source_documents=True)
                        st.session_state['pdf'] = qa

                    st.experimental_rerun()

        if st.session_state['plugin'] == "📝 Talk with your DOCUMENTS":
            if 'db' in st.session_state:
                # leave ./ from name for download
                file_name = st.session_state['db'][2:]
                st.download_button(
                    label="📩 Download vectorstore",
                    data=open(file_name, 'rb').read(),
                    file_name=file_name,
                    mime='application/zip'
                )
            if st.button('🛑📝 Remove PDF from context'):
                if 'pdf' in st.session_state:
                    del st.session_state['db']
                    del st.session_state['pdf']
                    del st.session_state['documents']
                del st.session_state['plugin']
                    
                st.experimental_rerun()

                        
                


def pre_load_pdf():                    
    upload_pdf = "BandoChat.pdf"
    with pdfplumber.open(upload_pdf) as pdf:
        documents += [page.extract_text() for page in pdf.pages]
        with st.spinner('🔨 Creating vectorstore...'):
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.create_documents(documents)
            # Select embeddings
            embeddings = st.session_state['hf']
            # Create a vectorstore from documents
            random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db_" + random_str)
            with st.spinner('🔨 Saving vectorstore...'):
                # save vectorstore
                db.persist()
                #create .zip file of directory to download
                shutil.make_archive("./chroma_db_" + random_str, 'zip', "./chroma_db_" + random_str)
                # save in session state and download
                st.session_state['db'] = "./chroma_db_" + random_str + ".zip" 
            with st.spinner('🔨 Creating QA chain...'):
                # Create retriever interface
                retriever = db.as_retriever()
                # Create QA chain
                qa = RetrievalQA.from_chain_type(llm=st.session_state['LLM'], chain_type='stuff', retriever=retriever,  return_source_documents=True)
                st.session_state['pdf'] = qa
                st.experimental_rerun()

input_container = st.container()
response_container = st.container()
data_view_container = st.container()
loading_container = st.container()
with data_view_container:
        with st.expander("🤖 View your **DOCUMENTs**"):
            st.write(st.session_state['documents'])

# Response output
## Function for taking user prompt as input followed

def prompt4Context(prompt, context, solution):
    final_prompt = f"""GENERAL INFORMATION : You is built by Alessandro Ciciarelli  the owener of intelligenzaartificialeitalia.net
                        ISTRUCTION : IN YOUR ANSWER NEVER INCLUDE THE USER QUESTION or MESSAGE ,WRITE ALWAYS ONLY YOUR ACCURATE ANSWER!
                        PREVIUS MESSAGE : ({context})
                        NOW THE USER ASK : {prompt}
                        THIS IS THE CORRECT ANSWER : ({solution}) 
                        WITHOUT CHANGING ANYTHING OF CORRECT ANSWER , MAKE THE ANSWER MORE DETALIED:"""
    return final_prompt

def generate_response(prompt):
    final_prompt =  ""
    make_better = True
    source = ""

    with loading_container:

        # FOR DEVELOPMENT PLUGIN
        # if st.session_state['plugin'] == "🔌 PLUGIN NAME" and 'PLUGIN DB' in st.session_state:
        #     with st.spinner('🚀 Using PLUGIN NAME...'):
        #         solution = st.session_state['PLUGIN DB']({"query": prompt})
        #         final_prompt = YourCustomPrompt(prompt, context)
        if st.session_state['plugin'] == "📝 Talk with your DOCUMENTS" and 'pdf' in st.session_state:
        #get only last message
            context = f"User: {st.session_state['past'][-1]}\nBot: {st.session_state['generated'][-1]}\n"
            with st.spinner('🚀 Using tool to get information...'):
                result = st.session_state['pdf']({"query": prompt})
                solution = result["result"]
                if len(solution.split()) > 110:
                    make_better = False
                    final_prompt = solution
                    if 'source_documents' in result and len(result["source_documents"]) > 0:
                        final_prompt += "\n\n✅Source:\n" 
                        for d in result["source_documents"]:
                            final_prompt += "- " + str(d) + "\n"
                else:
                    final_prompt = prompt4Context(prompt, context, solution)
                    if 'source_documents' in result and len(result["source_documents"]) > 0:
                        source += "\n\n✅Source:\n"
                        for d in result["source_documents"]:
                            source += "- " + str(d) + "\n"



## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if input_text and 'hf_email' in st.session_state and 'hf_pass' in st.session_state:
        response = generate_response(input_text)
        st.session_state.past.append(input_text)
        st.session_state.generated.append(response)
    

    #print message in normal order, frist user then bot
    if 'generated' in st.session_state:
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                with st.chat_message(name="user"):
                    st.markdown(st.session_state['past'][i])
                
                with st.chat_message(name="assistant"):
                    if len(st.session_state['generated'][i].split("✅Source:")) > 1:
                        source = st.session_state['generated'][i].split("✅Source:")[1]
                        mess = st.session_state['generated'][i].split("✅Source:")[0]

                        st.markdown(mess)
                        with st.expander("📚 Source of message number " + str(i+1)):
                            st.markdown(source)

                    else:
                        st.markdown(st.session_state['generated'][i])

            st.markdown('', unsafe_allow_html=True)
            
            
    else:
        st.info("👋 Hey , we are very happy to see you here 🤗")
        st.info("👉 Please Login to continue, click on top left corner to login 🚀")
        st.error("👉 If you are not registered on Hugging Face, please register first and then login 🤗")