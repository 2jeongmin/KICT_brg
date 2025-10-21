# 빠른 시작 가이드

## 5분 안에 시작하기

### 1단계: 설치 (1분)
```bash
# 프로젝트 디렉토리로 이동
cd KICT_brg

# 패키지 설치
pip install -r requirements.txt
```

### 2단계: 설정 (2분)

`config/my_config.yaml` 생성:
```yaml
data:
  root_dir: "YOUR_DATA_PATH"  # 실제 경로로 변경
  mode: "stream"

features:
  acceleration:
    extract: ["ptp", "noise", "fft"]
    sampling_rate: 100

database:
  host: "localhost"
  database: "bridge_monitoring"
  user: "postgres"
  password: "YOUR_PASSWORD"  # 실제 비밀번호

processing:
  max_files: 10  # 테스트용
  n_workers: 2
```

### 3단계: DB 초기화 (1분)
```bash
python scripts/init_database.py --config config/my_config.yaml
```

### 4단계: 실행 (1분)
```bash
python main.py --config config/my_config.yaml
```

**완료!** 🎉

### 결과 확인
```bash
# 처리 결과
ls output/

# DB 확인
python scripts/analyze_results.py -d
```

---

## 다음 단계

✅ 테스트 완료  
→ `max_files: null`로 변경 후 전체 실행

✅ 전체 실행  
→ [USER_GUIDE.md](USER_GUIDE.md) 참고하여 고급 기능 활용

✅ 문제 발생  
→ [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 참고

---

## 주요 명령어
```bash
# 테스트 실행
python main.py --config config/test.yaml --max-files 10

# 전체 실행
python main.py --config config/default.yaml

# 특정 센서
python main.py --config config/default.yaml --sensors DNA21001,DNA21002

# 재처리
python main.py --config config/default.yaml --reprocess

# 진행 모니터링
python scripts/monitor_progress.py

# 결과 분석
python scripts/analyze_results.py -d -o output/report.xlsx
```
