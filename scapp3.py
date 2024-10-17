import streamlit as st
import xml.etree.ElementTree as ET
import requests
from langchain.schema import Document  # Importing Document class
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap

# Your API key and URL (updated to request XML)
key = '6448425a457377653630797a536a4c'
url = 'http://openapi.seoul.go.kr:8088/6448425a457377653630797a536a4c/xml/parkingKickboard/1/1000/'

# Fetch data from the API
response = requests.get(url)
content = response.content.decode('utf-8')

# Parse the XML data
tree = ET.ElementTree(ET.fromstring(content))
root = tree.getroot()

# Function to filter and get scooter parking data
def scoot_parklot_data(root):
    items = root.findall(".//row")
    주차구역 = []
    for item in items:
        info = {
                '거치대순번': item.find("SN").text if item.find("SN") is not None else '',
                '거치대지역':item.find("SGG_NM").text if item.find("SGG_NM") is not None else '',
                '거치대주소': item.find("PSTN").text if item.find("PSTN") is not None else '', ''
                '거치대상세위치': item.find("DTL_PSTN").text if item.find("DTL_PSTN") is not None else '',
                '거치대 유무': item.find("STAND_YN").text if item.find("STAND_YN") is not None else '',
                '거치대 크기': item.find("STAND_SIZE").text if item.find("STAND_SIZE") is not None else '',
            }
        주차구역.append(info)
    return 주차구역


def main():
    st.markdown(
        """
        <h1 style='text-align: center;'>서울시 구별 전동킥보드 주차구역</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h4 style='text-align: center;'>🛴희망하는 주차구역을 입력해주세요: </h4>", unsafe_allow_html=True)
    query = st.text_input("")
    
    if st.button("검색"):
        주차구역 = scoot_parklot_data(root)
        
        if 주차구역:
            documents = [
                Document(page_content=", ".join([f"{key}: {str(info[key])}" for key in ['거치대순번', '거치대지역', '거치대주소', '거치대상세위치', '거치대 유무', '거치대 크기']]))
                for info in 주차구역
            ]
            
            embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
            db = FAISS.from_documents(documents, embedding_function)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 30, 'fetch_k': 1000})
            
            llm = ChatOllama(model="gemma2:9b", temperature=0.3, base_url="http://127.0.0.1:11434/")
            template = """
    너는 거치대 안내로봇이야. 사용자가 요청한 지역에 있는 거치대의 유무와 크기, 그리고 가장 가까운 지하철역을 알려줘야 해. 예를 들어 용산 지역에 있는 거치대를 알려달라고 하면 거치대지역이 용산구인 거치대를 답변해줘
    다음의 형식에 맞춰서 정보를 제공해줘:

    예시: 
    A동 거치대주소, 거치대 크기 상세내용, 가까운 지하철역 상세 내용
    B동 거치대주소, 거치대 크기 상세내용, 가까운 지하철역 상세 내용

    최대 10개의 정보를 사용자에게 제공할 수 있어. 반드시 한국어로 답해줘.

    제공된 정보에서 최대한 정확하고 상세하게 응답하도록 해.

    Answer the question based only on the following context:
    {context}

    Question: {question}
        """

            
            prompt = ChatPromptTemplate.from_template(template)

            chain = RunnableMap({
                "context": lambda x: retriever.get_relevant_documents(x['question']),
                "question": lambda x: x['question']
            }) | prompt | llm
            content = chain.invoke({'question': query}).content
            
            st.markdown(f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">해당 구에 대한 주차 정보가 없습니다.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
