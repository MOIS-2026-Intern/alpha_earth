# 1. 저장소 클론
`git clone <repo-url>`
<br>
`cd alpha_earth`

# 2. venv 생성 & 활성화
**macOS**

`python3 -m venv venv`
<br>
`source venv/bin/activate`

**Windows**

`venv\Scripts\activate`

# 3. 의존성 설치
`pip install -r requirements.txt`

# 4. .env 파일 생성 (.env.example 참고)
EE_PROJECT_ID 값 채우기

`EE_PROJECT_ID=your-earth-engine-project-id`


# 5. Earth Engine 인증 (최초 1회)
earthengine authenticate

# 6. 실행
`python test.py`


## `run_analysis.py` 라인 범위별 역할

아래 라인 번호는 현재 주석이 추가된 `seoul_flood/run_analysis.py` 기준이다.

- `1-36`: 라이브러리 import, 스크립트 경로 고정, `.env` 로드
- `39-115`: 입력/출력 경로 해석, JSON/CSV 저장 유틸, 숫자 파싱
- `118-200`: 서울 자치구명 매핑, CSV에서 구 이름 추출, 구별 통계 정규화
- `203-287`: 배수펌프장 CSV를 자치구별 펌프장 수/용량/유역/유수지 feature로 요약
- `290-355`: 하수관로 수위 센서 자료를 자치구별 센서 수 feature로 요약
- `358-390`: 분석 설정값, 입력/출력 파일 경로, Earth Engine 초기화
- `393-431`: 공간 block 기반 5-fold 배정 로직과 서울 행정경계 로드
- `434-476`: 침수 기준점 GeoJSON을 `label=1` 양성 target으로 읽고, 서울 전체 fold raster 생성
- `479-534`: 구 단위 배수 인프라 통계를 ADM2 polygon raster feature로 변환
- `537-615`: 지형/수문/토지피복/배수 인프라 정적 feature 이미지 생성
- `618-634`: 2024년 서울 AlphaEarth annual embedding 로드
- `637-651`: 물 영역과 침수점 주변 300m를 제외한 음성 후보 mask 생성
- `654-694`: AlphaEarth 64차원 embedding을 침수 기준점 평균과의 유사도 `alpha_score`로 축약
- `697-734`: fold별 train/validation 양성점 분리, 양성 60m buffer, 최종 입력 feature 목록 정의
- `737-764`: 30m 분석 스케일로 양성/음성 sample을 균형 추출
- `767-787`: Random Forest 학습과 validation 혼동행렬 생성
- `790-900`: accuracy, recall, F1, ROC-AUC, PR-AUC 등 성능 지표와 feature importance 계산
- `903-952`: fold 하나에 대해 sample 추출, 학습, 검증, importance 계산
- `955-1040`: 5-fold 결과 평균/표준편차 요약, feature importance 요약, 저장용 결과 정리
- `1043-1101`: 전체 실행 진입점, 5-fold 반복 실행, `metrics.json`/`cv_results.csv`/`feature_importance.csv` 저장

