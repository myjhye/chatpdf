import sys
import io
import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st

# 표준 출력의 인코딩을 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# 웹 앱의 제목을 설정
st.title("ChatPDF")
st.write("---")

# 사용자로부터 OpenAI API 키 입력받기 (비밀번호 형식)
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

# PDF 파일 업로더 설정
uploaded_file = st.file_uploader("Drop your PDF here!", type=['pdf'])
st.write("---")

# 업로드된 PDF 파일을 처리하여 페이지별로 나누는 함수
def pdf_to_document(uploaded_file):
    # 임시 경로
    temp_dir = tempfile.TemporaryDirectory()
    # 업로드된 파일을 임시 경로에 저장
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    # 업로드된 파일을 바이너리 쓰기 모드로 열고, 업로드된 파일 데이터 저장
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    # 임시 경로의 파일 읽기
    loader = PyPDFLoader(temp_filepath)
    # 파일을 페이지 별로 나누기 (1, 2, 3페이지...)
    pages = loader.load_and_split()
    return pages

# 업로드된 파일이 있다면 다음 코드를 실행
if uploaded_file is not None:

    # 파일을 페이지 별로 나누기
    pages = pdf_to_document(uploaded_file)

    # 페이지를 더 세분화(각 페이지당 300 텍스트로)
    text_splitter = RecursiveCharacterTextSplitter(
        # 각 청크의 크기 설정
        chunk_size=300,
        # 청크 간 겹치는 부분 설정  
        chunk_overlap=20,
        # 길이 측정 함수  
        length_function=len,  
        is_separator_regex=False
    )

     # 더 세분화한 최종 텍스트
    texts = text_splitter.split_documents(pages)

    # 임베딩 초기화
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # 텍스트를 임베딩하여 벡터로 변환하고 Chroma 데이터베이스에 저장
    db = Chroma.from_documents(texts, embeddings_model)

    # 질문 입력 필드와 버튼 설정
    st.header("Ask your PDF a question!")
    question = st.text_input('What do you want to know?')

    if st.button('Ask Away!'):
        with st.spinner('loading...'):
            # 챗봇 변수 초기화
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)
            # 질문 처리 및 결과 반환
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])
