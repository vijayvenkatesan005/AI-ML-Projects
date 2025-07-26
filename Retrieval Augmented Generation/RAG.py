import time
from yt_dlp import YoutubeDL
import streamlit as st
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from secret_key import openai_key

embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

relevant_metadata_set = {
    "title", 
    "uploader", 
    "upload_date", 
    "duration", 
    "view_count", 
    "like_count", 
    "description", 
    "tags"
}

def extract_metadata(video_url):
    with YoutubeDL({}) as ydl:
        info = ydl.extract_info(video_url, download=False)
    return info

def filter_metadata(url_metadata):
    return {k: url_metadata[k] for k in relevant_metadata_set if k in url_metadata}

def create_formatted_string(metadata_map):
    parts = []
    if "title" in metadata_map:
        parts.append(f"Title: {metadata_map['title']}")
    if "uploader" in metadata_map:
        parts.append(f"Uploader: {metadata_map['uploader']}")
    if "upload_date" in metadata_map:
        parts.append(f"Upload date: {metadata_map['upload_date']}")
    if "duration" in metadata_map:
        parts.append(f"Duration: {metadata_map['duration']} seconds")
    if "view_count" in metadata_map:
        parts.append(f"Views: {metadata_map['view_count']}")
    if "like_count" in metadata_map:
        parts.append(f"Likes: {metadata_map['like_count']}")
    if "description" in metadata_map:
        parts.append(f"Description: {metadata_map['description']}")
    if "tags" in metadata_map:
        tags_string = ", ".join(metadata_map["tags"])
        parts.append(f"Tags: {tags_string}")
    
    return "\n".join(parts)

st.title("Video Metadata Q&A 📽️")
st.sidebar.title("Video URL Links")

main_placeholder = st.empty()

urls = []
n = 3

for i in range(n):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

video_metadata_list = []
filtered_video_metadata_list = []
formatted_strings_list = []

if process_url_clicked:
    for url in urls:
        metadata = extract_metadata(url)
        video_metadata_list.append(metadata)
    
    for url_metadata in video_metadata_list:
        filtered_metadata = filter_metadata(url_metadata)
        filtered_video_metadata_list.append(filtered_metadata)
    
    for url_filtered_metadata in filtered_video_metadata_list:
        formatted_string = create_formatted_string(url_filtered_metadata)
        formatted_strings_list.append(formatted_string)
    
    main_placeholder.text("Data Loading...Started...✅✅✅")
    docs = [Document(page_content=formatted_string, metadata={"url": urls[index]}) for index, formatted_string in enumerate(formatted_strings_list)]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")

    chunks = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    main_placeholder.text("Embeddings Finished Building...✅✅✅")
    time.sleep(2)
    vectorstore.save_local("my_faiss_index")

query = main_placeholder.text_input("Question: ")

if query:
    vectorstore = FAISS.load_local(
        "my_faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        openai_api_key=openai_key,
        model="gpt-4o"
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

    result = rag_chain({"query": query})

    st.header("Answer")
    st.subheader(result["result"])

    sources = result.get("source_documents", [])

    if sources:
        st.subheader("Sources")
        seen_urls = set()
        for doc in sources:
            url = doc.metadata.get("url", "Unknwon URL")
            if url not in seen_urls:
                st.write(url)
                seen_urls.add(url)
    








    







