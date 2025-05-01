from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
import json
from typing import List, Dict, Any, Mapping, Optional
import os
from dotenv import load_dotenv
import re

# Langchain imports
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Load environment variables
load_dotenv()

# FastAPI 애플리케이션 설정
app = FastAPI(
    title='회의 인사이트 추출 API (Langchain)',
    description='Langchain Map-Reduce를 사용하여 회의록을 분석하고 핵심 인사이트를 JSON 형식으로 반환하는 API',
    version='1.1'
)

LLAMA_API_URL = os.getenv("LLAMA_API_URL")

if not LLAMA_API_URL:
    raise ValueError("LLAMA_API_URL 환경 변수가 설정되지 않았습니다.")

# Llama API 호출 래퍼 (기존 코드 유지)
class LlamaAPI:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def __call__(self, prompt: str) -> str:
        payload = {
            "messages": [
                {
                "user": prompt
                }
            ],
            "max_tokens": 32768,
            "temperature": 0.7,
            "top_p": 0.9
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=120) # Increased timeout
            response.raise_for_status()
            result = response.json()
            # Assuming the response structure is like {'response': '...'}
            if "text" not in result:
                 raise ValueError("LLM 응답에 'text' 키가 없습니다.")
            # Clean the response string
            cleaned_result = result["text"]
            cleaned_result = re.sub(r'<think>.*?</think>\s*\n*', '', cleaned_result, flags=re.DOTALL)
            cleaned_result = cleaned_result.replace('```json\n', "").replace('```', "").strip()
            print(f"LLM API 응답: {result}")  # Debugging line to check the response
            return cleaned_result
        except requests.exceptions.Timeout:
             raise HTTPException(status_code=504, detail="LLM API 요청 시간 초과")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"LLM API 요청 실패: {e}")
        except (ValueError, KeyError) as e:
            # Catch potential issues with response structure or cleaning
            raise HTTPException(status_code=500, detail=f"LLM 응답 처리 오류: {e}")
        except Exception as e:
            # Catch any other unexpected errors during the API call
             raise HTTPException(status_code=500, detail=f"LLM API 호출 중 예기치 않은 오류: {e}")

# Custom Langchain LLM Wrapper
class CustomLlamaLangchain(LLM):
    llama_api: LlamaAPI

    @property
    def _llm_type(self) -> str:
        return "custom_llama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Note: The 'stop' parameter is not explicitly handled by LlamaAPI but is part of the LLM interface.
        # LlamaAPI handles exceptions internally and raises HTTPException if needed.
        try:
            return self.llama_api(prompt)
        except HTTPException as http_exc:
            # Re-raise HTTPException to be caught by FastAPI error handling
            raise http_exc
        except Exception as e:
            # Wrap other exceptions potentially missed by LlamaAPI's handler
            raise RuntimeError(f"CustomLlamaLangchain 내부 오류: {e}") from e


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"api_url": self.llama_api.api_url}

# 전역 LLM 인스턴스 (Langchain Wrapper 사용)
try:
    llama_api_instance = LlamaAPI(LLAMA_API_URL)
    langchain_llm = CustomLlamaLangchain(llama_api=llama_api_instance)
except Exception as e:
    # Handle potential errors during LLM initialization (e.g., invalid URL)
    print(f"LLM 초기화 실패: {e}")
    # Depending on the desired behavior, you might exit or disable the endpoint
    langchain_llm = None # Or raise an error to prevent app startup

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

반드시 다음 형식으로만 응답하세요:
{{
  "1": "첫 번째 핵심 인사이트",
  "2": "두 번째 핵심 인사이트",
  "3": "세 번째 핵심 인사이트"
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

class AnalyzeResponse(BaseModel):
    insight_1: str = Field(..., alias='1', description='첫 번째 인사이트')
    insight_2: str = Field(..., alias='2', description='두 번째 인사이트')
    insight_3: str = Field(..., alias='3', description='세 번째 인사이트')

    class Config:
        allow_population_by_field_name = True
        # Example added for clarity in OpenAPI docs
        schema_extra = {
            "example": {
                "1": "프로젝트 A의 주요 위험 요소는 예산 초과 가능성입니다.",
                "2": "마케팅 팀은 다음 분기 캠페인 전략을 재검토해야 합니다.",
                "3": "신규 기능 X에 대한 사용자 피드백은 긍정적이지만, 성능 개선이 필요합니다."
            }
        }


@app.post("/analyze", response_model=AnalyzeResponse, summary="회의록 분석 및 인사이트 추출 (Langchain)")
async def analyze_meeting_langchain(request: AnalyzeRequest):
    if not langchain_llm:
         raise HTTPException(status_code=503, detail="LLM 서비스가 초기화되지 않았습니다.")

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
            final_insights = json.loads(summary_json_str)
            # Validate the structure
            if not isinstance(final_insights, dict) or not all(k in final_insights for k in ["1", "2", "3"]):
                 raise ValueError("Reduce 단계에서 예상된 JSON 객체 형식(키 '1', '2', '3')을 반환하지 않았습니다.")
            # Validate value types (should be strings)
            if not all(isinstance(v, str) for v in final_insights.values()):
                 raise ValueError("Reduce 단계 JSON 객체의 값이 문자열이 아닙니다.")

        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail=f"Reduce 단계의 최종 출력을 JSON으로 파싱하는 데 실패했습니다. 출력: {summary_json_str}")
        except ValueError as e:
             raise HTTPException(status_code=500, detail=f"Reduce 단계의 최종 출력 유효성 검사 실패: {e}")


        # 5) Return validated result
        # Pydantic will automatically handle the alias mapping for the response
        return final_insights

    # Catch specific HTTPExceptions raised by LlamaAPI or validation
    except HTTPException as he:
        raise he
    # Catch potential runtime errors from the Langchain LLM wrapper
    except RuntimeError as rte:
         raise HTTPException(status_code=500, detail=f"Langchain LLM 실행 중 오류: {rte}")
    # Catch any other unexpected errors during the process
    except Exception as e:
        # Log the error for debugging purposes if possible
        print(f"예기치 않은 분석 오류: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중 예기치 않은 오류 발생: {e}")

# Add a root endpoint for basic check
@app.get("/")
async def root():
    return {"message": "회의 인사이트 추출 API (Langchain) 실행 중"}

# If running directly (for testing)
if __name__ == "__main__":
    import uvicorn
    # Ensure .env is loaded if running directly
    load_dotenv()
    API_URL = os.getenv("LLAMA_API_URL")
    if not API_URL:
        print("경고: LLAMA_API_URL 환경 변수가 설정되지 않았습니다. API가 제대로 작동하지 않을 수 있습니다.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
