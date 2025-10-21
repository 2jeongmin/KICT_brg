"""
실시간 파이프라인 진행 상황 모니터링
체크포인트 파일을 읽어 진행률을 시각화
"""

import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys


def format_time(seconds: float) -> str:
    """초를 HH:MM:SS 형식으로 변환"""
    if seconds < 0 or seconds > 86400:  # 24시간 이상
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def monitor_progress(checkpoint_path: Path = Path("checkpoints/pipeline_checkpoint.json"),
                     refresh_rate: int = 1):
    """
    체크포인트 파일 모니터링
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        refresh_rate: 갱신 주기 (초)
    """
    print("=" * 80)
    print("Pipeline Progress Monitor")
    print("=" * 80)
    print(f"Monitoring: {checkpoint_path}")
    print("Press Ctrl+C to exit\n")
    
    last_processed = 0
    last_time = time.time()
    
    try:
        while True:
            if not checkpoint_path.exists():
                print("\r⏳ Waiting for checkpoint file...", end='', flush=True)
                time.sleep(refresh_rate)
                continue
            
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                time.sleep(refresh_rate)
                continue
            
            # 데이터 추출
            total = data.get('total_files', 0)
            processed = data.get('processed_files', 0)
            failed = data.get('failed_files', 0)
            saved_records = data.get('records_saved', 0)
            
            # 진행률 계산
            progress = (processed / total * 100) if total > 0 else 0
            
            # 속도 계산 (최근 속도)
            current_time = time.time()
            time_diff = current_time - last_time
            if time_diff >= 1.0:  # 1초 이상 경과 시 속도 계산
                speed = (processed - last_processed) / time_diff
                last_processed = processed
                last_time = current_time
            else:
                speed = 0
            
            # ETA 계산
            remaining = total - processed
            eta_seconds = remaining / speed if speed > 0 else 0
            eta_str = format_time(eta_seconds)
            
            # 진행률 바
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # 출력
            status = (
                f"\r{bar} {progress:.1f}% | "
                f"{processed:,}/{total:,} files | "
                f"Speed: {speed:.2f} f/s | "
                f"ETA: {eta_str} | "
                f"Failed: {failed} | "
                f"Records: {saved_records:,}"
            )
            
            print(status, end='', flush=True)
            
            # 완료 체크
            if processed >= total and total > 0:
                print("\n\n✅ Processing completed!")
                break
            
            time.sleep(refresh_rate)
            
    except KeyboardInterrupt:
        print("\n\n⏹ Monitoring stopped by user")
        sys.exit(0)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor pipeline progress in real-time"
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path("checkpoints/pipeline_checkpoint.json"),
        help="Checkpoint file path"
    )
    parser.add_argument(
        '--refresh',
        type=int,
        default=1,
        help="Refresh rate in seconds (default: 1)"
    )
    
    args = parser.parse_args()
    
    monitor_progress(
        checkpoint_path=args.checkpoint,
        refresh_rate=args.refresh
    )


if __name__ == "__main__":
    main()