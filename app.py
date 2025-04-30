from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
import json
from typing import List, Dict
import os

# FastAPI 애플리케이션 설정
app = FastAPI(
    title='회의 인사이트 추출 API',
    description='회의록을 분석하여 핵심 인사이트를 JSON 형식으로 반환하는 API',
    version='1.0'
)

LLAMA_API_URL = os.getenv("LLAMA_API_URL")

# Llama API 호출 래퍼
class LlamaAPI:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def __call__(self, prompt: str) -> str:
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            print("result", result) # json
            result = result["response"].replace("<|im_end|>", "").replace("<|endofturn|>", "").replace('```json\n',"").replace('```',"").strip()
            # 'insight' 검증
            # if not isinstance(insights, list) or not all("insight" in item for item in insights):
            #     raise ValueError("모든 요소에 'insight' 키가 없습니다")
            return result
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"LLM API 요청 실패: {e}")
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"LLM 응답 형식 오류: {e}")

# 전역 LLM 인스턴스
llm = LlamaAPI(LLAMA_API_URL)

# Prompt 템플릿
MAP_PROMPT = """
당신은 ‘회의록 분석 전문가’입니다.
주어진 회의 내용에서 가능한 후보 '핵심 인사이트'를 모두 뽑아내세요. 각 인사이트는 상세한 인사이트를 담은 문장이어야 하며, 반드시 JSON 배열 형태로만 출력하세요.

회의 내용:
```
{text}
```

반드시 다음 형식으로만 응답하세요:
[
  {{ "insight": "인사이트 문장 A" }},
  {{ "insight": "인사이트 문장 B" }}
]
```"""

REDUCE_PROMPT = """
당신은 ‘최종 인사이트 필터링 전문가’입니다.
Map 단계에서 추출된 여러 후보 인사이트 중, 중복되는 내용을 합치고, 이를 정리하여 3개의 핵심 인사이트로 요약하세요. 
각 인사이트는 상세한 인사이트를 담은 문장이어야 하며, 반드시 JSON 객체 형태로만 출력하세요.

반드시 다음 형식으로만 응답하세요:
{{
  "1": "첫 번째 핵심 인사이트",
  "2": "두 번째 핵심 인사이트",
  "3": "세 번째 핵심 인사이트"
}}
```

후보 인사이트 목록(입력):
```
{text}
```

중요:
- 키는 항상 "1", "2", "3"
- 값은 핵심을 담은 인사이트여야합니다.
"""

# Pydantic 모델 정의
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description='회의 내용')

class AnalyzeResponse(BaseModel):
    insight_1: str = Field(..., alias='1', description='첫 번째 인사이트')
    insight_2: str = Field(..., alias='2', description='두 번째 인사이트')
    insight_3: str = Field(..., alias='3', description='세 번째 인사이트')

    class Config:
        allow_population_by_field_name = True

@app.post("/analyze", response_model=AnalyzeResponse, summary="회의록 분석 및 인사이트 추출")
async def analyze_meeting(request: AnalyzeRequest):
    try:
        # 1) Map 단계
        map_input = MAP_PROMPT.format(text=request.text)
        map_output = llm(map_input)
        print("map_output", type(map_output)) # json
        candidates = json.loads(map_output)
        print("candidates", candidates) # json
        # Map 단계에서 JSON 배열인지 확인
        if not isinstance(candidates, list):
            raise HTTPException(status_code=500, detail="Map 단계에서 JSON 배열을 반환하지 않았습니다.")
        print(type(candidates)) 
        # 각 후보의 'insight' 키 존재 확인
        insights_list = []
        for idx, item in enumerate(candidates, start=1):
            print("item", item, type(item)) # json
            if not isinstance(item, dict) or "insight" not in item:
                raise HTTPException(status_code=500, detail=f"Map 단계 후보 형식 오류: 항목 {idx}")
            insights_list.append(item["insight"])
        print("insights_list", insights_list) # json
        # 2) Reduce 단계
        combined_text = "\n".join(insights_list)
        reduce_input = REDUCE_PROMPT.format(text=combined_text)
        reduce_output = llm(reduce_input)
        result = json.loads(reduce_output)
        if not isinstance(result, dict) or not all(k in result for k in ["1", "2", "3"]):
            raise HTTPException(status_code=500, detail="Reduce 단계에서 예상 JSON 객체를 반환하지 않았습니다.")

        return result

    # HTTPException은 그대로 전달
    except HTTPException as he:
        raise he
    except Exception as e:
        # 기타 예기치 못한 오류
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {e}")
