from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
from typing import List, Dict, Any, Mapping, Optional
import os
from dotenv import load_dotenv
import re
import google.generativeai as genai # Added for Gemini

# Langchain imports
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
# from langchain.llms.base import LLM # No longer needed for direct LLM wrapper
# from langchain.callbacks.manager import CallbackManagerForLLMRun # No longer needed
from langchain_google_genai import ChatGoogleGenerativeAI # Added for Gemini LLM
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError # Added for specific error handling

# Load environment variables
load_dotenv()

# FastAPI 애플리케이션 설정
app = FastAPI(
    title='회의 인사이트 추출 API (Gemini & Langchain)',
    description='Langchain Map-Reduce와 Google Gemini를 사용하여 회의록을 분석하고 핵심 인사이트를 JSON 형식으로 반환하는 API',
    version='1.2' # Version updated
)

# Load Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# 전역 LLM 인스턴스 (Gemini 사용)
try:
    # Initialize the Gemini LLM using the Langchain wrapper
    # Using "gemini-1.5-flash" as it's generally available and efficient
    langchain_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=0.3,
        # convert_system_message_to_human=True # May be needed depending on prompt structure
        # Add safety settings if needed, e.g.,
        # safety_settings={
        #     genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        #     # Add other categories as needed
        # }
    )
    # Perform a simple test call to ensure the API key is valid and the model is accessible
    langchain_llm.invoke("Test prompt")
    print("Gemini LLM 초기화 성공.")
except Exception as e:
    # Handle potential errors during LLM initialization (e.g., invalid API key, network issues)
    print(f"Gemini LLM 초기화 실패: {e}")
    langchain_llm = None # Set to None to prevent app from running with a broken LLM

# Prompt 템플릿 (Langchain 형식)
# Note: Ensure the variable name in the template matches the input variable for the chain ('text' or 'input_documents')
# load_summarize_chain expects input variable named 'text' for map_prompt and combine_prompt
MAP_PROMPT_TEMPLATE = """
당신은 ‘회의록 분석 전문가’입니다.
주어진 회의 내용 청크에서 가능한 후보 '핵심 인사이트'를 모두 뽑아내세요. 각 인사이트는 상세한 인사이트를 담은 문장이어야 하며, 반드시 JSON 배열 형태로만 출력하세요.

회의 내용 청크:
```
{text}
```

반드시 다음 형식으로만 응답하세요:
[
  {{ "insight": "인사이트 문장 A" }},
  {{ "insight": "인사이트 문장 B" }}
]
"""
map_prompt = PromptTemplate(template=MAP_PROMPT_TEMPLATE, input_variables=["text"])

REDUCE_PROMPT_TEMPLATE = """
당신은 ‘최종 인사이트 필터링 전문가’입니다.
Map 단계에서 추출된 여러 후보 인사이트 목록을 검토하고, 중복되는 내용을 합치고, 이를 정리하여 최종 3개의 핵심 인사이트로 요약하세요.
각 인사이트는 상세한 인사이트를 담은 문장이어야 하며, 반드시 JSON 객체 형태로만 출력하세요.
score 정확도 점수는 0.0에서 1.0 사이의 값으로, 인사이트의 신뢰도를 나타냅니다. 유사도를 평가하세요.

반드시 다음 형식으로만 응답하세요:
{{
  "1": {{"insight": "첫 번째 핵심 인사이트", "score": 정확도 점수}},
  "2": {{"insight": "두 번째 핵심 인사이트", "score": 정확도 점수}},
  "3": {{"insight": "세 번째 핵심 인사이트", "score": 정확도 점수}}
}}

후보 인사이트 목록 (입력):
```
{text}
```

중요:
- 키는 항상 "1", "2", "3" 이어야 합니다.
- 값은 핵심을 담은 인사이트 문장이어야 합니다.
"""
reduce_prompt = PromptTemplate(template=REDUCE_PROMPT_TEMPLATE, input_variables=["text"])


# Pydantic 모델 정의 (기존 코드 유지)
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description='회의 내용')

class InsightItem(BaseModel):
    insight: str
    score: float

class AnalyzeResponse(BaseModel):
    insight_1: InsightItem = Field(..., alias='1', description='첫 번째 인사이트')
    insight_2: InsightItem = Field(..., alias='2', description='두 번째 인사이트')
    insight_3: InsightItem = Field(..., alias='3', description='세 번째 인사이트')

    class Config:
        validate_by_name = True # Updated from allow_population_by_field_name
        # Example added for clarity in OpenAPI docs
        json_schema_extra = { # Updated from schema_extra
            "example": {
                "1": {"insight": "The main risk for Project A is the possibility of budget overrun.", "score": 0.9}, # Example updated
                "2": {"insight": "마케팅 팀은 다음 분기 캠페인 전략을 재검토해야 합니다.", "score": 0.85},
                "3": {"insight": "신규 기능 X에 대한 사용자 피드백은 긍정적이지만, 성능 개선이 필요합니다.", "score": 0.8}
            }
        }


@app.post("/analyze", response_model=AnalyzeResponse, summary="회의록 분석 및 인사이트 추출 (Gemini & Langchain)")
async def analyze_meeting_gemini(request: AnalyzeRequest): # Renamed function
    if not langchain_llm:
         raise HTTPException(status_code=503, detail="Gemini LLM 서비스가 초기화되지 않았습니다.")

    try:
        # 1) Text Splitting
        # Adjust chunk_size to be well below the model's token limit (e.g., 1024 tokens).
        # Using a smaller character count as a proxy.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(request.text)]

        if not docs:
             raise HTTPException(status_code=400, detail="입력 텍스트가 비어 있거나 처리할 수 없습니다.")

        # 2) Load Summarization Chain
        # Note: Langchain might have different ways to pass prompts depending on version.
        # Ensure map_prompt and combine_prompt are correctly passed.
        chain = load_summarize_chain(
            llm=langchain_llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=reduce_prompt,
            # verbose=True # Uncomment for debugging
        )

        # 3) Run Chain
        # The chain expects a list of Document objects.
        # The output key depends on the chain type; for summarize chains, it's often 'output_text'.
        # Use invoke for newer Langchain versions, run for older ones. Let's use invoke.
        result_dict = await chain.ainvoke({"input_documents": docs}) # Use await for async call
        summary_json_str = result_dict.get("output_text", "")

        if not summary_json_str:
             raise HTTPException(status_code=500, detail="요약 체인에서 결과를 생성하지 못했습니다.")

        # 4) Parse Final JSON Output
        try:
            summary_json_str=re.sub(r"^```json\n|\n```$", "", summary_json_str.strip())
            print(f"Reduce 단계 최종 출력: {summary_json_str}") # Debugging output
            final_insights = json.loads(summary_json_str)
            # Validate the structure
            if not isinstance(final_insights, dict) or not all(k in final_insights for k in ["1", "2", "3"]):
                 raise ValueError("Reduce 단계에서 예상된 JSON 객체 형식(키 '1', '2', '3')을 반환하지 않았습니다.")
            # Validate value types (should be strings)
            if not all(isinstance(v, dict) and isinstance(v.get("insight"), str) for v in final_insights.values()):
                raise ValueError("Reduce 단계 JSON 객체의 'insight' 값이 문자열이 아닙니다.")

        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail=f"Reduce 단계의 최종 출력을 JSON으로 파싱하는 데 실패했습니다. 출력: {summary_json_str}")
        except ValueError as e:
             raise HTTPException(status_code=500, detail=f"Reduce 단계의 최종 출력 유효성 검사 실패: {e}")


        # 5) Return validated result
        # Pydantic will automatically handle the alias mapping for the response
        return final_insights

    # Catch specific HTTPExceptions raised during validation or processing
    except HTTPException as he:
        raise he
    # Catch specific errors from the Google Generative AI API
    except ChatGoogleGenerativeAIError as ge:
        print(f"Gemini API 오류 발생: {ge}")
        raise HTTPException(status_code=503, detail=f"Gemini API 통신 오류: {ge}")
    # Catch other potential errors from the Langchain wrapper or general execution
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Gemini 분석 중 예기치 않은 오류: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중 예기치 않은 오류 발생: {e}")

# Add a root endpoint for basic check
@app.get("/")
async def root():
    return {"message": "회의 인사이트 추출 API (Gemini & Langchain) 실행 중"}

# If running directly (for testing)
if __name__ == "__main__":
    import uvicorn
    # Ensure .env is loaded if running directly
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        print("경고: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. API가 제대로 작동하지 않을 수 있습니다.")
    # Initialize LLM here as well for standalone testing if needed, or rely on global init
    if langchain_llm is None and API_KEY: # Attempt re-init if global failed but key exists
        try:
            genai.configure(api_key=API_KEY)
            langchain_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
            langchain_llm.invoke("Test prompt") # Test again
            print("Gemini LLM 초기화 성공 (standalone).")
        except Exception as e:
            print(f"Gemini LLM 초기화 실패 (standalone): {e}")
            langchain_llm = None # Ensure it remains None if init fails

    uvicorn.run(app, host="0.0.0.0", port=8000)
