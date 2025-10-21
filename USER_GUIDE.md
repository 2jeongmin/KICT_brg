# 교량 센서 데이터 특성값 추출 시스템
## 사용자 가이드

버전: 1.0  
최종 업데이트: 2025-10-28

---

## 📑 목차

1. [시스템 개요](#1-시스템-개요)
2. [설치 및 환경 설정](#2-설치-및-환경-설정)
3. [빠른 시작](#3-빠른-시작)
4. [설정 파일 작성](#4-설정-파일-작성)
5. [실행 방법](#5-실행-방법)
6. [데이터 조회 및 분석](#6-데이터-조회-및-분석)
7. [고급 기능](#7-고급-기능)
8. [문제 해결](#8-문제-해결)
9. [FAQ](#9-faq)

---

## 1. 시스템 개요

### 1.1 목적

한국건설기술연구원의 교량 모니터링 프로젝트에서 수집된 대용량 센서 데이터(10TB, 100개 교량, 3년)로부터 시간 단위 특성값을 추출하고 TimescaleDB에 저장하는 시스템입니다.

### 1.2 주요 기능

- **Zip 스트리밍 처리**: 디스크 용량 증가 없이 압축 파일에서 직접 데이터 처리
- **특성값 자동 추출**: PTP, 노이즈 레벨, 고유진동수(FFT) 등
- **시계열 DB 저장**: PostgreSQL + TimescaleDB
- **체크포인트 & 재시작**: 중단 시 자동 재개
- **실시간 모니터링**: 진행 상황 추적

### 1.3 시스템 요구사항

#### 하드웨어
- **CPU**: 4코어 이상 권장
- **메모리**: 16GB 이상 권장 (32GB 최적)
- **저장공간**: 
  - 소스 데이터: 10TB (Dropbox 등 클라우드)
  - DB 저장공간: 100GB 이상
  - 임시 공간: 10GB

#### 소프트웨어
- **운영체제**: Windows 10/11, Linux (Ubuntu 18.04+)
- **Python**: 3.8 이상
- **PostgreSQL**: 12 이상
- **TimescaleDB**: 2.0 이상

---

## 2. 설치 및 환경 설정

### 2.1 Python 환경 설정

#### Step 1: 프로젝트 다운로드
```bash
cd /path/to/your/workspace
git clone 
cd KICT_brg
```

#### Step 2: 가상 환경 생성 (권장)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: 의존성 설치
```bash
pip install -r requirements.txt
```

**requirements.txt 내용:**
```txt
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
sqlalchemy>=1.4.0
psycopg2-binary>=2.9.0
tqdm>=4.62.0
pyyaml>=6.0
```

### 2.2 PostgreSQL + TimescaleDB 설치

#### Windows

1. **PostgreSQL 설치**
   - [PostgreSQL 공식 사이트](https://www.postgresql.org/download/windows/) 다운로드
   - 설치 시 포트: `5432`, 비밀번호 설정 필수

2. **TimescaleDB 설치**
```powershell
   # PostgreSQL 설치 후
   # TimescaleDB installer 다운로드 및 실행
   # https://docs.timescale.com/install/latest/self-hosted/installation-windows/
```

3. **데이터베이스 생성**
```sql
   -- pgAdmin 또는 psql에서 실행
   CREATE DATABASE bridge_monitoring;
   \c bridge_monitoring
   CREATE EXTENSION IF NOT EXISTS timescaledb;
```

#### Linux (Ubuntu)
```bash
# PostgreSQL 설치
sudo apt update
sudo apt install postgresql postgresql-contrib

# TimescaleDB 저장소 추가
sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -

# TimescaleDB 설치
sudo apt update
sudo apt install timescaledb-2-postgresql-14

# TimescaleDB 설정
sudo timescaledb-tune

# PostgreSQL 재시작
sudo systemctl restart postgresql

# 데이터베이스 생성
sudo -u postgres psql
CREATE DATABASE bridge_monitoring;
\c bridge_monitoring
CREATE EXTENSION IF NOT EXISTS timescaledb;
\q
```

### 2.3 데이터베이스 초기화
```bash
# 테이블 생성
python scripts/init_database.py --config config/default.yaml

# 또는 수동으로
psql -U postgres -d bridge_monitoring -f schema.sql
```

**확인:**
```bash
python scripts/init_database.py --config config/default.yaml
```

출력 예시:
```
======================================
Database Initialization
======================================
Host: localhost
Port: 5432
Database: bridge_monitoring
...
Tables created successfully
Hypertable created: acc_hourly_features
Indexes created successfully
Database initialization completed!
```

---

## 3. 빠른 시작

### 3.1 5분 안에 시작하기

#### Step 1: 설정 파일 복사
```bash
cp config/test.yaml config/my_config.yaml
```

#### Step 2: 설정 파일 편집
`config/my_config.yaml` 파일을 열어 다음 항목 수정:
```yaml
data:
  root_dir: "Z:/05. Data/01. Under_Process/027. KICT_BMAPS/upload"  # 실제 경로로 변경

database:
  host: "localhost"
  port: 5432
  database: "bridge_monitoring"
  user: "postgres"
  password: "your_password"  # 실제 비밀번호 입력
```

#### Step 3: 테스트 실행 (10개 파일)
```bash
python main.py --config config/my_config.yaml --max-files 10
```

#### Step 4: 결과 확인
```bash
# 처리된 센서 목록 확인
python scripts/analyze_results.py -d

# 결과 파일 확인
ls output/
```

### 3.2 전체 데이터 처리

설정 파일에서 `max_files: null`로 변경 후 실행:
```bash
python main.py --config config/my_config.yaml
```

**예상 소요 시간:** 
- 50,000개 파일 기준: 약 4-6시간 (4 workers)
- 파일당 평균 처리 시간: ~0.3초

---

## 4. 설정 파일 작성

### 4.1 설정 파일 구조

`config/default.yaml` 전체 구조:
```yaml
# 데이터 소스
data:
  root_dir: "Z:/05. Data/01. Under_Process/027. KICT_BMAPS/upload"
  mode: "stream"              # stream | extract_delete | temp_dir
  temp_dir: null              # temp_dir 모드일 때 사용
  file_pattern: "*.zip"

# 추출할 특성값
features:
  acceleration:
    extract: ["ptp", "mean", "std", "rms", "noise", "fft", "kurtosis", "skewness"]
    sampling_rate: 100        # Hz
    
    fft:
      freq_range: [0.5, 20]   # Hz
      n_peaks: 3
      prominence_ratio: 0.1
    
    noise:
      method: "static_std"    # static_std | full_std | snr
      static_threshold: 0.01
  
  static:
    extract: ["mean", "ptp", "noise"]
    sensor_types: ["temperature", "humidity", "displacement", "crack"]

# 데이터베이스
database:
  host: "localhost"
  port: 5432
  database: "bridge_monitoring"
  user: "postgres"
  password: "your_password"
  schema: "public"
  on_conflict: "skip"         # skip | update
  batch_size: 1000

# 처리 설정
processing:
  batch_size: 100             # 한번에 처리할 zip 파일 개수
  n_workers: 4                # 병렬 처리 워커 수
  use_multiprocessing: true
  max_files: null             # null = 전체, 숫자 = 제한
  
  # 필터
  sensors: null               # ["DNA21001", "DNA21002"] 또는 null
  date_range:
    start: null               # "2024-01-01" 또는 null
    end: null
  
  reprocess: false            # 기존 데이터 덮어쓰기
  skip_existing: true         # 이미 처리된 파일 건너뛰기

# 검증 설정
validation:
  enable: true
  acceleration:
    min_records: 320000
    max_records: 400000
    value_range: [-100, 100]
    time_delta_range: [9, 11]

# 로깅
logging:
  level: "INFO"               # DEBUG | INFO | WARNING | ERROR
  file:
    enable: true
    dir: "logs"
  console:
    enable: true

# 모니터링
monitoring:
  enable: true
  report_interval: 100        # N개 파일마다 진행 상황 출력
```

### 4.2 주요 설정 항목 설명

#### 4.2.1 데이터 처리 모드 (`data.mode`)

| 모드 | 설명 | 디스크 사용 | 속도 | 권장 |
|------|------|------------|------|------|
| **stream** | 메모리에서 직접 처리 | 0 증가 | 중간 | ✅ 기본값 |
| extract_delete | 압축 해제 후 원본 삭제 | 일시적 최소 | 빠름 | ⚠️ 주의 |
| temp_dir | 로컬 임시 디렉토리 사용 | 로컬만 | 빠름 | 특수 상황 |

**권장:** Dropbox 환경에서는 `stream` 모드 사용

#### 4.2.2 특성값 선택 (`features.acceleration.extract`)

사용 가능한 특성값:
- `ptp`: Peak-to-Peak (필수)
- `mean`, `std`, `median`: 기본 통계
- `rms`: Root Mean Square
- `noise`: 노이즈 레벨 (필수)
- `fft`: 고유진동수 (필수)
- `kurtosis`, `skewness`: 분포 형태

**권장 최소 세트:** `["ptp", "noise", "fft"]`

#### 4.2.3 FFT 설정
```yaml
fft:
  freq_range: [0.5, 20]      # 분석 주파수 범위 [Hz]
  n_peaks: 3                 # 추출할 피크 개수 (1차, 2차, 3차 모드)
  prominence_ratio: 0.1      # 피크 검출 임계값 (0.05-0.2 권장)
```

**교량별 권장 설정:**

| 교량 종류 | freq_range | 이유 |
|----------|------------|------|
| 소형 교량 | [1.0, 30] | 고유진동수 높음 |
| 중형 교량 | [0.5, 20] | 기본값 |
| 대형 교량 | [0.2, 10] | 고유진동수 낮음 |

#### 4.2.4 중복 처리 (`database.on_conflict`)

- `skip`: 중복 시 건너뛰기 (기본값, 안전)
- `update`: 중복 시 덮어쓰기 (재처리 시)

**사용 예:**
```bash
# 재처리 시
python main.py --config config/default.yaml --reprocess
```

### 4.3 시나리오별 설정 예시

#### 시나리오 1: 빠른 테스트 (10개 파일)
```yaml
# config/quick_test.yaml
processing:
  max_files: 10
  n_workers: 2
  batch_size: 5

features:
  acceleration:
    extract: ["ptp", "noise", "fft"]  # 필수만

logging:
  level: "DEBUG"
```

#### 시나리오 2: 특정 센서만 처리
```yaml
# config/specific_sensors.yaml
processing:
  sensors: ["DNA21001", "DNA21002", "DNA21003"]
  max_files: null  # 해당 센서의 모든 파일
```
```bash
# 또는 커맨드라인에서
python main.py --config config/default.yaml --sensors DNA21001,DNA21002
```

#### 시나리오 3: 날짜 범위 제한
```yaml
# config/date_range.yaml
processing:
  date_range:
    start: "2023-01-01"
    end: "2023-12-31"
```

#### 시나리오 4: 고성능 서버
```yaml
# config/high_performance.yaml
processing:
  batch_size: 200
  n_workers: 8
  
database:
  batch_size: 2000
```

#### 시나리오 5: 재처리 (데이터 업데이트)
```yaml
# config/reprocess.yaml
processing:
  reprocess: true
  skip_existing: false

database:
  on_conflict: "update"
```

---

## 5. 실행 방법

### 5.1 기본 실행

#### Windows
```batch
REM 기본 실행
python main.py --config config/default.yaml

REM 배치 파일로 실행
scripts\run_full_pipeline.bat config\default.yaml
```

#### Linux/macOS
```bash
# 기본 실행
python main.py --config config/default.yaml

# 셸 스크립트로 실행
bash scripts/run_full_pipeline.sh config/default.yaml
```

### 5.2 커맨드라인 옵션
```bash
python main.py --help
```

**주요 옵션:**

| 옵션 | 단축 | 설명 | 예시 |
|------|------|------|------|
| `--config` | `-c` | 설정 파일 경로 (필수) | `--config config/default.yaml` |
| `--max-files` | | 처리할 최대 파일 수 | `--max-files 100` |
| `--sensors` | | 처리할 센서 ID (쉼표 구분) | `--sensors DNA21001,DNA21002` |
| `--features` | | 추출할 특성값 | `--features ptp,noise,fft` |
| `--workers` | | 병렬 처리 워커 수 | `--workers 8` |
| `--reprocess` | | 재처리 모드 | `--reprocess` |
| `--start-date` | | 시작 날짜 | `--start-date 2024-01-01` |
| `--end-date` | | 종료 날짜 | `--end-date 2024-12-31` |
| `--dry-run` | | 실행하지 않고 설정만 확인 | `--dry-run` |

**실행 예시:**
```bash
# 1. 테스트 실행 (10개 파일, 디버그 모드)
python main.py --config config/test.yaml --max-files 10

# 2. 특정 센서만 처리
python main.py --config config/default.yaml --sensors DNA21001,DNA21002

# 3. 2024년 데이터만 처리
python main.py --config config/default.yaml \
  --start-date 2024-01-01 --end-date 2024-12-31

# 4. 고성능 처리 (8 workers)
python main.py --config config/default.yaml --workers 8

# 5. 재처리
python main.py --config config/default.yaml --reprocess

# 6. 설정 검증만 (실행 안함)
python main.py --config config/default.yaml --dry-run
```

### 5.3 진행 상황 모니터링

#### 방법 1: 실시간 모니터링 스크립트

별도 터미널에서:
```bash
python scripts/monitor_progress.py
```

출력 예시:
```
========================================
Pipeline Progress Monitor
========================================
████████████████████░░░░░░░░░░░░░░░░░░ 45.2% | 2260/5000 files | Speed: 3.21 f/s | ETA: 0:23:45 | Failed: 12
```

#### 방법 2: 로그 파일 확인
```bash
# 최신 로그 실시간 확인
tail -f logs/processing_*.log

# 에러만 확인
grep ERROR logs/processing_*.log
```

#### 방법 3: 체크포인트 확인
```bash
python scripts/manage_checkpoint.py show
```

출력:
```
======================================
Checkpoint Information
======================================
Start time: 2024-10-28T14:30:22.123456
Last update: 2024-10-28T15:45:10.987654
Progress: 2260/5000 files
Failed: 12 files
Records saved: 814400
Last processed: smartcs_2_DNAGW2109_202405231900.zip
======================================
```

### 5.4 중단 및 재시작

#### 안전한 중단
```
Ctrl+C  (한 번만 누르기)
```

시스템이 자동으로:
1. 현재 배치 처리 완료
2. 체크포인트 저장
3. 안전하게 종료

#### 재시작
```bash
# 동일한 명령어로 실행하면 자동으로 이어서 처리
python main.py --config config/default.yaml
```

출력:
```
Resuming from checkpoint...
Resumed at: 2260 files processed
```

#### 처음부터 다시 시작
```bash
# 체크포인트 삭제
python scripts/manage_checkpoint.py clear

# 다시 실행
python main.py --config config/default.yaml
```

---

## 6. 데이터 조회 및 분석

### 6.1 결과 파일 확인

처리 완료 후 `output/` 디렉토리에 생성되는 파일:
```
output/
├── pipeline_results_20241028_143022.json   # 처리 결과 요약
├── features_temp_batch10.csv               # 중간 결과 (옵션)
└── report_20241028.xlsx                    # 종합 리포트 (생성 시)
```

#### 결과 JSON 파일 예시
```json
{
  "success": true,
  "start_time": "2024-10-28T14:30:22.123456",
  "end_time": "2024-10-28_18:45:33.654321",
  "duration_seconds": 15311.53,
  "duration_formatted": "4:15:11",
  "files": {
    "total": 5000,
    "processed": 4988,
    "failed": 12,
    "success_rate": 99.76
  },
  "records": {
    "total": 1796400,
    "saved": 1796200,
    "failed": 200
  },
  "performance": {
    "files_per_second": 3.26,
    "records_per_second": 117.3
  },
  "errors": {
    "empty_zip": 5,
    "processing_error": 7
  }
}
```

### 6.2 데이터베이스 조회

#### Python 스크립트로 조회
```python
from src.database import TimeseriesDB

# DB 연결
db = TimeseriesDB(
    host='localhost',
    database='bridge_monitoring',
    user='postgres',
    password='your_password'
)

# 센서 목록
sensors = db.get_sensor_list()
print(f"Total sensors: {len(sensors)}")

# 특정 센서 데이터 조회
df = db.query_features(
    sensor_id='DNA21001',
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31)
)

print(df[['time', 'ptp', 'noise_level', 'freq_1', 'freq_2', 'freq_3']])

# CSV로 내보내기
db.export_to_csv(
    output_path='output/DNA21001_2024.csv',
    sensor_id='DNA21001',
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31)
)

db.close()
```

#### SQL로 직접 조회
```sql
-- psql 또는 pgAdmin에서 실행

-- 1. 전체 센서 목록
SELECT DISTINCT sensor_id 
FROM acc_hourly_features 
ORDER BY sensor_id;

-- 2. 특정 센서의 최근 데이터
SELECT time, ptp, noise_level, freq_1, freq_2, freq_3
FROM acc_hourly_features
WHERE sensor_id = 'DNA21001'
  AND time > NOW() - INTERVAL '7 days'
ORDER BY time DESC
LIMIT 100;

-- 3. 일별 집계
SELECT 
    sensor_id,
    time_bucket('1 day', time) AS day,
    COUNT(*) AS n_records,
    AVG(ptp) AS avg_ptp,
    AVG(noise_level) AS avg_noise,
    AVG(freq_1) AS avg_freq1
FROM acc_hourly_features
WHERE sensor_id = 'DNA21001'
GROUP BY sensor_id, day
ORDER BY day DESC;

-- 4. 고유진동수 추이
SELECT time, freq_1, freq_2, freq_3
FROM acc_hourly_features
WHERE sensor_id = 'DNA21001'
  AND time BETWEEN '2024-01-01' AND '2024-12-31'
ORDER BY time;

-- 5. 이상치 검출 (PTP가 평균의 3배 이상)
WITH stats AS (
    SELECT sensor_id, AVG(ptp) AS avg_ptp, STDDEV(ptp) AS std_ptp
    FROM acc_hourly_features
    GROUP BY sensor_id
)
SELECT a.sensor_id, a.time, a.ptp, s.avg_ptp
FROM acc_hourly_features a
JOIN stats s ON a.sensor_id = s.sensor_id
WHERE a.ptp > s.avg_ptp + 3 * s.std_ptp
ORDER BY a.time DESC;

-- 6. 센서별 통계 요약
SELECT 
    sensor_id,
    COUNT(*) AS n_records,
    MIN(time) AS first_record,
    MAX(time) AS last_record,
    AVG(ptp) AS avg_ptp,
    AVG(noise_level) AS avg_noise,
    AVG(freq_1) AS avg_freq1,
    SUM(CASE WHEN is_valid THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS valid_rate
FROM acc_hourly_features
GROUP BY sensor_id
ORDER BY sensor_id;
```

### 6.3 종합 리포트 생성
```bash
# Excel 리포트 생성
python scripts/analyze_results.py -d -o output/comprehensive_report.xlsx
```

생성되는 Excel 파일 구성:

| 시트 | 내용 |
|------|------|
| Summary | 전체 요약 통계 |
| Sensor Stats | 센서별 상세 통계 |
| Daily Counts | 일별 데이터 개수 |
| Sample Data | 샘플 데이터 (최근 1000개) |

### 6.4 시각화 예시

#### Python (matplotlib)
```python
import pandas as pd
import matplotlib.pyplot as plt
from src.database import TimeseriesDB

db = TimeseriesDB(host='localhost', database='bridge_monitoring', 
                  user='postgres', password='your_password')

# 데이터 조회
df = db.query_features(
    sensor_id='DNA21001',
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31)
)

# 시간에 따른 PTP 변화
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['ptp'])
plt.xlabel('Time')
plt.ylabel('Peak-to-Peak [m/s²]')
plt.title('PTP Trend - DNA21001')
plt.grid(True)
plt.savefig('output/ptp_trend.png', dpi=150)
plt.show()

# 고유진동수 분포
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    axes[i].hist(df[f'freq_{i+1}'].dropna(), bins=50)
    axes[i].set_xlabel('Frequency [Hz]')
    axes[i].set_title(f'{i+1}차 고유진동수')
plt.tight_layout()
plt.savefig('output/freq_distribution.png', dpi=150)
plt.show()

db.close()
```

---

## 7. 고급 기능

### 7.1 병렬 처리 최적화

#### CPU 코어 수에 따른 권장 설정

| CPU 코어 | workers | batch_size | 예상 처리 속도 |
|----------|---------|------------|---------------|
| 4 코어 | 2-3 | 50-100 | 2.5 files/s |
| 8 코어 | 4-6 | 100-150 | 4.0 files/s |
| 16 코어 | 8-12 | 150-200 | 6.5 files/s |
```yaml
# config/optimized.yaml
processing:
  n_workers: 8
  batch_size: 150
```

### 7.2 선택적 특성값 추출

특정 센서에만 FFT 적용:
```python
# custom_pipeline.py
from src.pipeline import ProcessingPipeline

class CustomPipeline(ProcessingPipeline):
    def _process_files(self, zip_files):
        for zip_file in zip_files:
            # 파일명에서 센서 ID 추출
            if 'DNAGW' in zip_file.name:
                # GW 센서는 FFT 제외
                self.features_config['acceleration']['extract'] = ['ptp', 'noise']
            else:
                # 일반 센서는 모든 특성값
                self.features_config['acceleration']['extract'] = [
                    'ptp', 'noise', 'fft', 'rms'
                ]
            
            # 처리 계속
            super()._process_single_file(zip_file)
```

### 7.3 커스텀 특성값 추가
```python
# src/feature_extraction.py에 추가

class FeatureExtractor:
    # ... 기존 코드
    
    def extract_custom_feature(self, values: np.ndarray) -> float:
        """
        사용자 정의 특성값
        예: 진동의 에너지 레벨
        """
        energy = np.sum(values**2) / len(values)
        return float(energy)
    
    def extract_all(self, df: DataFrame, sensor_type: str = 'acceleration') -> Dict:
        features = self._extract_acceleration_features(df)
        
        # 커스텀 특성값 추가
        features['energy_level'] = self.extract_custom_feature(df['value'].values)
        
        return features
```

### 7.5 데이터 품질 모니터링
```python
# scripts/quality_check.py
"""
데이터 품질 모니터링
"""
from src.database import TimeseriesDB
import pandas as pd

def check_data_quality():
    """데이터 품질 검사"""
    db = TimeseriesDB(host='localhost', database='bridge_monitoring',
                      user='postgres', password='postgres')
    
    print("="*60)
    print("Data Quality Check")
    print("="*60)
    
    # 1. 유효성 비율
    df = db.query_features(table='acceleration')
    valid_rate = df['is_valid'].sum() / len(df) * 100
    print(f"\n1. Valid records: {valid_rate:.2f}%")
    
    if valid_rate < 95:
        print("   ⚠️  Warning: Low valid rate!")
    
    # 2. 센서별 데이터 개수
    sensor_counts = df.groupby('sensor_id').size()
    print(f"\n2. Records per sensor:")
    print(f"   Min: {sensor_counts.min()}")
    print(f"   Max: {sensor_counts.max()}")
    print(f"   Mean: {sensor_counts.mean():.0f}")
    
    # 불균형 센서 찾기
    threshold = sensor_counts.mean() * 0.5
    low_sensors = sensor_counts[sensor_counts < threshold]
    if len(low_sensors) > 0:
        print(f"\n   ⚠️  Sensors with low data count:")
        for sensor, count in low_sensors.items():
            print(f"      {sensor}: {count}")
    
    # 3. 데이터 연속성 검사
    print(f"\n3. Data continuity:")
    for sensor_id in df['sensor_id'].unique()[:5]:  # 상위 5개만
        sensor_df = df[df['sensor_id'] == sensor_id].sort_values('time')
        time_diffs = sensor_df['time'].diff().dt.total_seconds() / 3600
        
        gaps = time_diffs[time_diffs > 2].count()  # 2시간 이상 간격
        print(f"   {sensor_id}: {gaps} gaps")
    
    # 4. 특성값 이상치
    print(f"\n4. Outliers:")
    
    # PTP 이상치 (평균 + 3*std 이상)
    ptp_mean = df['ptp'].mean()
    ptp_std = df['ptp'].std()
    ptp_outliers = df[df['ptp'] > ptp_mean + 3*ptp_std]
    print(f"   PTP outliers: {len(ptp_outliers)} ({len(ptp_outliers)/len(df)*100:.2f}%)")
    
    # 노이즈 이상치
    noise_mean = df['noise_level'].mean()
    noise_std = df['noise_level'].std()
    noise_outliers = df[df['noise_level'] > noise_mean + 3*noise_std]
    print(f"   Noise outliers: {len(noise_outliers)} ({len(noise_outliers)/len(df)*100:.2f}%)")
    
    # 5. 주파수 범위 확인
    print(f"\n5. Frequency range:")
    for i in range(1, 4):
        freq_col = f'freq_{i}'
        if freq_col in df.columns:
            freqs = df[freq_col].dropna()
            if len(freqs) > 0:
                print(f"   {i}차: {freqs.min():.2f} - {freqs.max():.2f} Hz")
                
                # 비정상적인 주파수
                abnormal = freqs[(freqs < 0.1) | (freqs > 30)]
                if len(abnormal) > 0:
                    print(f"      ⚠️  {len(abnormal)} abnormal frequencies")
    
    print("\n" + "="*60)
    db.close()

if __name__ == "__main__":
    check_data_quality()
```

실행:
```bash
python scripts/quality_check.py
```

---

## 8. 문제 해결

### 8.1 일반적인 문제

#### 문제 1: DB 연결 실패
```
Error: could not connect to server: Connection refused
```

**해결 방법:**
```bash
# PostgreSQL 실행 여부 확인
# Windows
services.msc  # PostgreSQL 서비스 확인

# Linux
sudo systemctl status postgresql

# 재시작
sudo systemctl restart postgresql

# 포트 확인
netstat -an | grep 5432
```

#### 문제 2: 메모리 부족
```
MemoryError: Unable to allocate array
```

**해결 방법:**

1. 배치 크기 감소:
```yaml
processing:
  batch_size: 50  # 100 → 50
```

2. 워커 수 감소:
```yaml
processing:
  n_workers: 2  # 4 → 2
```

3. 스왑 메모리 추가 (Linux):
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 문제 3: 파일을 찾을 수 없음
```
FileNotFoundError: Directory not found: Z:/...
```

**해결 방법:**

1. 경로 확인:
```bash
# Windows
dir "Z:\05. Data\01. Under_Process"

# Linux/macOS
ls -la /path/to/data
```

2. 네트워크 드라이브 연결 확인

3. 상대 경로로 변경:
```yaml
data:
  root_dir: "./data"  # 절대 경로 대신 상대 경로
```

#### 문제 4: TimescaleDB 확장 오류
```
ERROR: could not open extension control file
```

**해결 방법:**
```sql
-- PostgreSQL에 접속
psql -U postgres

-- TimescaleDB 확장 생성
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 확인
\dx
```

#### 문제 5: 디스크 공간 부족
```
OSError: [Errno 28] No space left on device
```

**해결 방법:**

1. 디스크 공간 확인:
```bash
df -h
```

2. 불필요한 파일 삭제:
```bash
# 로그 파일 정리
rm logs/*.log

# 임시 파일 정리
rm -rf checkpoints/*.json
rm -rf output/*.csv
```

3. DB 압축:
```sql
VACUUM FULL acc_hourly_features;
```

#### 문제 6: 처리 속도 너무 느림
```
Speed: 0.5 files/s  (목표: 3+ files/s)
```

**해결 방법:**

1. 병렬 처리 증가:
```yaml
processing:
  n_workers: 8  # CPU 코어 수에 맞게
```

2. SSD 사용 권장

3. 네트워크 드라이브 대신 로컬 복사:
```bash
# 데이터를 로컬로 복사
rsync -av --progress Z:/data/ /local/data/
```

4. DB 인덱스 확인:
```sql
-- 인덱스 확인
\d+ acc_hourly_features

-- 누락된 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_acc_sensor_time 
    ON acc_hourly_features (sensor_id, time DESC);
```

### 8.2 에러 메시지별 해결

| 에러 메시지 | 원인 | 해결 방법 |
|------------|------|----------|
| `BadZipFile` | 손상된 zip 파일 | 해당 파일 건너뛰기 (자동) |
| `PermissionError` | 파일 접근 권한 없음 | 관리자 권한으로 실행 |
| `sqlalchemy.exc.OperationalError` | DB 연결 끊김 | PostgreSQL 재시작 |
| `KeyboardInterrupt` | 사용자 중단 | 정상, 재시작 가능 |
| `Empty DataFrame` | bin 파일 데이터 없음 | 파일 확인 또는 건너뛰기 |

### 8.3 로그 분석

#### 에러 로그 확인
```bash
# 모든 에러 확인
grep -i error logs/processing_*.log

# 특정 센서 에러
grep "DNA21001" logs/processing_*.log | grep -i error

# 최근 10개 에러
grep -i error logs/processing_*.log | tail -10
```

#### 처리 진행률 확인
```bash
# 처리된 파일 개수
grep "Processed batch" logs/processing_*.log | wc -l

# 실패한 파일
grep "Failed to process" logs/processing_*.log
```

### 8.4 데이터 무결성 검증
```python
# scripts/verify_data.py
"""
데이터 무결성 검증
"""
from src.database import TimeseriesDB
from datetime import datetime, timedelta

def verify_data_integrity():
    """데이터 무결성 검사"""
    db = TimeseriesDB(host='localhost', database='bridge_monitoring',
                      user='postgres', password='postgres')
    
    print("="*60)
    print("Data Integrity Verification")
    print("="*60)
    
    sensors = db.get_sensor_list()
    
    for sensor_id in sensors[:10]:  # 샘플 10개
        df = db.query_features(sensor_id=sensor_id)
        
        if len(df) == 0:
            print(f"\n{sensor_id}: No data")
            continue
        
        # 시간 연속성 검사
        df = df.sort_values('time')
        time_diffs = df['time'].diff().dt.total_seconds() / 3600
        
        # 정상: 1시간 간격
        normal = (time_diffs >= 0.9) & (time_diffs <= 1.1)
        normal_rate = normal.sum() / len(time_diffs) * 100
        
        # 결과
        status = "✓" if normal_rate > 95 else "✗"
        print(f"\n{status} {sensor_id}")
        print(f"   Records: {len(df)}")
        print(f"   Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"   Continuity: {normal_rate:.1f}%")
        
        # 특성값 범위 검사
        if 'ptp' in df.columns:
            ptp_range = (df['ptp'].min(), df['ptp'].max())
            if ptp_range[0] < 0 or ptp_range[1] > 100:
                print(f"   ⚠️  PTP out of range: {ptp_range}")
        
        if 'freq_1' in df.columns:
            freq_range = (df['freq_1'].min(), df['freq_1'].max())
            if freq_range[0] < 0.1 or freq_range[1] > 30:
                print(f"   ⚠️  Frequency out of range: {freq_range}")
    
    print("\n" + "="*60)
    db.close()

if __name__ == "__main__":
    verify_data_integrity()
```

---

## 9. FAQ

### Q1: 처리 중 중단하면 데이터가 손실되나요?
**A:** 아니요. Ctrl+C로 중단 시:
1. 현재 배치 처리 완료
2. 체크포인트 자동 저장
3. 이미 저장된 데이터는 DB에 보존
4. 재시작 시 중단 지점부터 계속

### Q2: 같은 파일을 다시 처리하면 어떻게 되나요?
**A:** `on_conflict: skip` 설정 시:
- 중복 데이터는 자동으로 건너뜀
- `on_conflict: update` 설정 시:
- 기존 데이터를 새 데이터로 덮어씀

### Q3: 전체 처리에 얼마나 걸리나요?
**A:** 
- 50,000개 파일 기준: 4-6시간
- 파일당 평균: ~0.3초
- 워커 4개, 배치 100 기준

실제 소요 시간은 하드웨어에 따라 다름:
- SSD: 빠름
- HDD: 느림 (2배 이상)
- 네트워크 드라이브: 매우 느림 (5배 이상)

### Q4: DB 용량은 얼마나 필요한가요?
**A:**
- 가속도 데이터: 시간당 1개 × 500채널 × 3년 = 약 13,140,000 레코드
- 레코드당 크기: ~200 bytes
- 예상 총 용량: 약 2.5GB (압축 전)
- TimescaleDB 압축 후: 약 500MB
- 여유분 포함 권장: 10GB 이상

### Q5: 특정 센서만 재처리하고 싶습니다
**A:**
```bash
# 방법 1: 센서 필터
python main.py --config config/default.yaml --sensors DNA21001

# 방법 2: DB에서 해당 센서 데이터 삭제 후 재처리
psql -U postgres -d bridge_monitoring
DELETE FROM acc_hourly_features WHERE sensor_id = 'DNA21001';
\q

python main.py --config config/default.yaml --sensors DNA21001
```

### Q6: 압축 파일을 풀어야 하나요?
**A:** 아니요! 
- 기본 `stream` 모드는 압축 해제 없이 메모리에서 직접 처리
- 디스크 용량 증가 0
- Dropbox 동기화 문제 없음

### Q7: 에러가 발생한 파일만 다시 처리하려면?
**A:**
```bash
# 1. 에러 파일 목록 확인
grep "Failed to process" logs/processing_*.log > failed_files.txt

# 2. 해당 파일들만 재처리 (수동)
# 또는 validation이 false인 데이터 재처리
python main.py --config config/default.yaml --reprocess

# DB에서 is_valid = false 데이터 삭제 후 재처리
psql -U postgres -d bridge_monitoring
DELETE FROM acc_hourly_features WHERE is_valid = false;
```

### Q8: 결과 데이터를 Excel로 보고 싶습니다
**A:**
```bash
# 방법 1: 종합 리포트 생성
python scripts/analyze_results.py -d -o output/report.xlsx

# 방법 2: 특정 센서만
python -c "
from src.database import TimeseriesDB
db = TimeseriesDB(host='localhost', database='bridge_monitoring', 
                  user='postgres', password='postgres')
df = db.query_features(sensor_id='DNA21001')
df.to_excel('output/DNA21001.xlsx', index=False)
db.close()
"
```

### Q9: 다른 PC로 DB를 옮기려면?
**A:**
```bash
# 백업
pg_dump -U postgres -d bridge_monitoring > backup.sql

# 복원
psql -U postgres -d bridge_monitoring_new < backup.sql

# 또는 pg_dump로 압축 백업
pg_dump -U postgres -Fc -d bridge_monitoring -f backup.dump

# 복원
pg_restore -U postgres -d bridge_monitoring_new backup.dump
```

### Q10: 실시간으로 데이터를 확인하려면?
**A:**
```bash
# 터미널 1: 파이프라인 실행
python main.py --config config/default.yaml

# 터미널 2: 실시간 모니터링
python scripts/monitor_progress.py

# 터미널 3: DB 조회 (주기적)
watch -n 5 'psql -U postgres -d bridge_monitoring -c "SELECT sensor_id, COUNT(*) FROM acc_hourly_features GROUP BY sensor_id ORDER BY sensor_id LIMIT 10;"'
```

---

## 10. 부록

### 10.1 디렉토리 구조 전체
```
KICT_brg/
├── main.py                          # 메인 실행 파일
├── README.md                        # 프로젝트 개요
├── USER_GUIDE.md                    # 이 문서
├── requirements.txt                 # Python 패키지
├── setup.py                         # 설치 스크립트
├── schema.sql                       # DB 스키마 (참고용)
│
├── config/                          # 설정 파일
│   ├── default.yaml                 # 기본 설정
│   ├── test.yaml                    # 테스트 설정
│   └── production.yaml              # 운영 설정
│
├── src/                             # 소스 코드
│   ├── __init__.py
│   ├── io_utils.py                  # Binary 파일 I/O
│   ├── zip_handler.py               # Zip 스트리밍
│   ├── feature_extraction.py        # 특성값 추출
│   ├── database.py                  # DB 연동
│   └── pipeline.py                  # 전체 파이프라인
│
├── scripts/                         # 유틸리티 스크립트
│   ├── init_database.py             # DB 초기화
│   ├── manage_checkpoint.py         # 체크포인트 관리
│   ├── monitor_progress.py          # 진행 모니터링
│   ├── analyze_results.py           # 결과 분석
│   ├── quality_check.py             # 품질 검사
│   ├── verify_data.py               # 무결성 검증
│   ├── run_full_pipeline.sh         # 실행 (Linux)
│   └── run_full_pipeline.bat        # 실행 (Windows)
│
├── tests/                           # 테스트 코드
│   ├── test_io_utils.py
│   ├── test_zip_handler.py
│   ├── test_feature_extraction.py
│   ├── test_database.py
│   └── test_pipeline.py
│
├── checkpoints/                     # 체크포인트 (자동 생성)
│   └── pipeline_checkpoint.json
│
├── logs/                            # 로그 파일 (자동 생성)
│   └── processing_YYYYMMDD_HHMMSS.log
│
└── output/                          # 결과 파일 (자동 생성)
    ├── pipeline_results_*.json
    ├── report_*.xlsx
    └── *.csv
```

### 10.2 주요 명령어 요약
```bash
# 설치
pip install -r requirements.txt

# DB 초기화
python scripts/init_database.py --config config/default.yaml

# 테스트 실행
python main.py --config config/test.yaml --max-files 10

# 전체 실행
python main.py --config config/default.yaml

# 모니터링
python scripts/monitor_progress.py

# 결과 분석
python scripts/analyze_results.py -d -o output/report.xlsx

# 품질 검사
python scripts/quality_check.py

# 체크포인트 관리
python scripts/manage_checkpoint.py show
python scripts/manage_checkpoint.py clear
```

### 10.3 설정 옵션 전체 목록

| 섹션 | 옵션 | 기본값 | 설명 |
|------|------|--------|------|
| **data** | root_dir | (필수) | 데이터 디렉토리 |
| | mode | stream | 처리 모드 |
| | file_pattern | *.zip | 파일 패턴 |
| **features.acceleration** | sampling_rate | 100 | 샘플링 주파수 [Hz] |
| | extract | [...] | 추출할 특성값 목록 |
| **features.acceleration.fft** | freq_range | [0.5, 20] | 주파수 범위 [Hz] |
| | n_peaks | 3 | 추출할 피크 개수 |
| | prominence_ratio | 0.1 | 피크 검출 임계값 |
| **database** | host | localhost | DB 호스트 |
| | port | 5432 | DB 포트 |
| | database | bridge_monitoring | DB 이름 |
| | on_conflict | skip | 중복 처리 방식 |
| | batch_size | 1000 | DB insert 배치 크기 |
| **processing** | batch_size | 100 | 처리 배치 크기 |
| | n_workers | 4 | 병렬 워커 수 |
| | max_files | null | 최대 파일 수 |
| | reprocess | false | 재처리 여부 |
| **logging** | level | INFO | 로그 레벨 |
| **monitoring** | report_interval | 100 | 리포트 간격 |
