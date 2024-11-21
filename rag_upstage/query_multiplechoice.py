'''
This code is for invoking answers based on multiple choice question format queries.
'''

import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from get_embedding_function import get_embedding_function
from dotenv import load_dotenv
from langchain.schema import Document
import os
import pandas as pd
from fetch_wikipedia import fetch_wikipedia_snippet

load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']

CHROMA_PATH = "D:/UpstageNLP_Team8/rag_upstage/chroma"
TEST_PATH = "D:/UpstageNLP_Team8/rag_upstage/test_data/test_samples.csv"

PROMPT_TEMPLATE = """
Answer the question based only on the following context
If the answer is not present in the context, please write "The information is not present in the context.":

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    prompts, answers = read_test_data(TEST_PATH)
    responses = []

    # Generate responses for each prompt
    for prompt in prompts:
        response = query_rag(prompt)
        responses.append(response)  # Append the raw response content
    
    # Calculate and print accuracy
    acc = accuracy(answers, responses)
    print(f"Final Accuracy: {acc}%")
    
    
def read_test_data(data_path):
    data = pd.read_csv(data_path)
    prompts = data['prompts']
    answers = data['answers']
    return prompts, answers

def query_rag(query_text: str):
    """
    Use RAG system to search context, and fetch data from DuckDuckGo only if context is missing.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # RAG 데이터베이스에서 질의에 대한 문맥 검색
    results = db.similarity_search_with_score(query_text, k=10)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # 초기 프롬프트 생성
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=query_text)
    model = ChatUpstage(api_key=upstage_api_key)
    response = model.invoke(prompt)
    response_content = response.content

    # 문맥이 없을 경우 추가 데이터를 가져옴
    if detect_missing_context(response_content):
        print("🔍 Missing context detected. Fetching data from wikipedia...")

        # wikipedia로 데이터 가져오기
        search_results = fetch_wikipedia_snippet(query_text)
        if search_results:
            print(f"👉 Adding wikipedia content for '{query_text}' to the database.")
            new_docs = []
            for result in search_results:
                snippet = result['snippet']
                link = result['link']
                try:
                    # 텍스트 요약
                    summary_prompt = f"""
                    Summarize the following content briefly and clearly:
                    {snippet}
                    """
                    summary = model.invoke(summary_prompt).content.strip()
                    new_docs.append(Document(page_content=summary, metadata={"source": link}))
                except Exception as e:
                    print(f"❌ Error summarizing content from {link}: {e}")

            # 새로운 문서가 있을 경우 DB에 추가
            if new_docs:
                db.add_documents(new_docs)
                db.persist()
                print("✅ New Wikipedia content added to the database.")

            # 업데이트된 DB에서 다시 문맥 검색
            results = db.similarity_search_with_score(query_text, k=10)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=query_text)
            response = model.invoke(prompt)

        else: print("❌ Error in fetching data from Wikipedia") 

    return response.content

def detect_missing_context(response_content: str) -> bool:
    """
    Uses the model to determine if the response indicates missing context.
    """
    # Prompt to evaluate whether the response indicates missing context
    evaluation_prompt = f"""
    Is the response below indicating missing context?
    Response: {response_content}

    Answer with "Yes" or "No".
    """

    model = ChatUpstage(api_key=upstage_api_key)
    response = model.invoke(evaluation_prompt)

    # Process the model's response
    evaluation_result = response.content.strip().lower()
    if evaluation_result in ["yes", "y"]:
        return True
    return False



import re

def extract_answer(response):
    """
    extracts the answer from the response using a regular expression.
    expected format: "[ANSWER]: (A) convolutional networks"

    if there are any answers formatted like the format, it returns None.
    """
    pattern = r"\[ANSWER\]:\s*\((A|B|C|D|E)\)"  # Regular expression to capture the answer letter and text
    match = re.search(pattern, response)

    if match:
        return match.group(1) # Extract the letter inside parentheses (e.g., A)
    else:
        return extract_again(response)

def extract_again(response):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, response)
    if match:
        return match.group(0)
    else:
        return None
    
def accuracy(answers, responses):
    """
    Calculates the accuracy of the generated answers.
    
    Parameters:
        answers (list): The list of correct answers.
        responses (list): The list of generated responses.

    Returns:
        float: The accuracy percentage.
    """
    cnt = 0

    for answer, response in zip(answers, responses):
        print("-" * 10)
        generated_answer = extract_answer(response)
        print(response)
        
        # check
        if generated_answer:
            print(f"generated answer: {generated_answer}, answer: {answer}")
        else:
            print("extraction fail")

        if generated_answer is None:
            continue
        if generated_answer in answer:
            cnt += 1

    acc = (cnt / len(answers)) * 100
    
    return acc


if __name__ == "__main__":
    main()
