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

- `1-48`: 라이브러리 import, 스크립트 경로 고정, `.env` 로드
- `51-135`: 입력/출력 경로 해석, 정수 목록 파싱, JSON/CSV 저장 유틸, 숫자 파싱
- `138-220`: 서울 자치구명 매핑, CSV에서 구 이름 추출, 구별 통계 정규화
- `223-375`: 배수펌프장 CSV와 하수관로 수위 센서 자료를 자치구별 배수 인프라 feature로 요약
- `378-456`: 분석 설정값, 입력/출력 파일 경로, 외부 침수구역 검증 설정, Earth Engine 초기화
- `459-542`: 공간 block 기반 fold 배정 로직, 서울 행정경계 로드, 침수 기준점 로드, 서울 전체 fold raster 생성
- `545-681`: 구 단위 배수 인프라 통계를 ADM2 polygon raster feature로 변환하고 지형/수문/토지피복 정적 feature와 결합
- `684-760`: 2024년 서울 AlphaEarth annual embedding 로드, 음성 후보 mask 생성, `alpha_score` feature 생성
- `763-922`: fold별 train/validation 입력 구성, 최종 입력 feature 목록 정의, Gradient Tree Boosting soft voting 모델과 Optuna 튜닝 설정
- `924-1270`: 30m 분석 스케일 sample 추출, GTB 학습/soft voting 분류, validation metric, hotspot metric, feature importance 계산
- `1273-1541`: fold sample 캐시, Optuna 하이퍼파라미터 튜닝 실행, 최적 파라미터를 최종 모델 설정에 반영
- `1544-1662`: fold 하나에 대한 학습/검증 실행과 전체 침수 기준점을 사용한 최종 모델 학습
- `1665-1798`: 5-fold 결과 평균/표준편차, feature importance, top-k hotspot metric 요약, 저장용 결과 정리
- `1801-2355`: 공식 침수구역 shp/zip 자료 로드, 단순화, 빈도별 외부 검증 overlap metric 계산
- `2357-2366`: `create_final_flood_risk_map.py`를 import하고 `create_map_outputs(analysis_result, run_analysis_module)`를 호출해 최종 지도/구별 hotspot 순위 파일을 생성
- `2369-2523`: 전체 실행 진입점. 선택적 Optuna 튜닝, 5-fold 검증, 최종 모델 학습, 외부 검증을 수행하고, `GENERATE_FINAL_MAP=1`이면 `run_final_map_outputs()`를 통해 `create_final_flood_risk_map.py`까지 함께 실행한 뒤 `metrics.json`/`cv_results.csv`/`feature_importance.csv`/`topk_summary.csv`/`external_validation.csv`/`hyperparameter_tuning.csv`를 저장


# 모델 실행 및 결과물
### 실행 방법
터미널에서 아래 명령어 실행
```bash
python seoul_flood/run_analysis.py
```

기본값은 `GENERATE_FINAL_MAP=1`이므로 `run_analysis.py` 안에서 `create_final_flood_risk_map.py`도 함께 실행된다.
최종 지도 생성을 건너뛰려면 `GENERATE_FINAL_MAP=0 python seoul_flood/run_analysis.py`로 실행한다.

### 결과물
- 지도 : `seoul_flood/seoul_flood_risk_final.html`
- 구 단위 위험도 상위 5% 포함 개수 : `seoul_flood/outputs/analysis/top5_red_points_by_gu.csv`

