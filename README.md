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
