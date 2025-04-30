# 회의 인사이트 추출 API

이 프로젝트는 FastAPI를 사용하여 회의록을 분석하고 핵심 인사이트를 추출하는 API를 제공합니다. MapReduce 방식을 활용하여 후보 인사이트를 생성하고 최종적으로 3개의 핵심 인사이트로 요약합니다.

## 기능

- 회의 내용 분석 및 핵심 인사이트 추출
- MapReduce 기반의 인사이트 생성 및 요약
- JSON 형식의 API 응답

## 설정 및 설치

1.  **저장소 클론:**

    ```bash
    git clone [저장소 URL]
    cd [프로젝트 디렉토리]
    ```

2.  **가상 환경 생성 및 활성화 (권장):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    # .venv\Scripts\activate  # Windows
    ```

3.  **의존성 설치:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **.env 파일 설정:**

    프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가합니다.

    ```env
    LLAMA_API_URL="[Llama API 엔드포인트 URL]"
    ```

    `[Llama API 엔드포인트 URL]` 부분에는 실제로 사용할 Llama API의 엔드포인트를 입력합니다.

5.  **애플리케이션 실행:**

    ```bash
    uvicorn app:app --reload
    ```

    애플리케이션은 기본적으로 `http://127.0.0.1:8000`에서 실행됩니다.

## API 엔드포인트 사용

### `POST /analyze`

회의 내용을 분석하고 핵심 인사이트를 추출합니다.

**요청 (Request Body):**

```json
{
  "text": "여기에 분석할 회의 내용을 입력하세요."
}
```

**응답 (Response Body):**

```json
{
  "1": "첫 번째 핵심 인사이트",
  "2": "두 번째 핵심 인사이트",
  "3": "세 번째 핵심 인사이트"
}
```

**예시:**

```bash
curl -X POST \
  http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "오늘 회의에서는 프로젝트 일정에 대해 논의했습니다. 특히 다음 스프린트의 목표와 각 팀원의 담당 업무를 확정했습니다. 디자인 팀은 UI/UX 개선 작업을 시작하고, 개발 팀은 백엔드 API 개발에 집중하기로 했습니다. 마케팅 팀은 다음 주부터 홍보 자료 준비에 착수할 예정입니다."
  }'
```

## 프로젝트 구조

```
.
├── .env              # 환경 변수 설정 파일
├── .gitignore        # Git 추적에서 제외할 파일 목록
├── app.py            # FastAPI 애플리케이션 코드
└── requirements.txt  # 프로젝트 의존성 목록
```

## 환경 변수

- `LLAMA_API_URL`: Llama API의 엔드포인트 URL. `.env` 파일에 설정해야 합니다.
