#!/usr/bin/env python3
# scripts/manage_checkpoint.py
"""
체크포인트 관리 스크립트
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def show_checkpoint():
    """체크포인트 정보 표시"""
    checkpoint_file = Path('checkpoints/pipeline_checkpoint.json')
    
    if not checkpoint_file.exists():
        print("No checkpoint found")
        return
    
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    
    print("="*60)
    print("Checkpoint Information")
    print("="*60)
    print(f"Start time: {data['start_time']}")
    print(f"Last update: {data['last_update']}")
    print(f"Progress: {data['processed_files']}/{data['total_files']} files")
    print(f"Failed: {data['failed_files']} files")
    print(f"Records saved: {data['saved_records']}")
    print(f"Last processed: {data.get('last_processed_file', 'N/A')}")
    print("="*60)


def clear_checkpoint():
    """체크포인트 삭제"""
    checkpoint_file = Path('checkpoints/pipeline_checkpoint.json')
    
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Checkpoint cleared")
    else:
        print("No checkpoint to clear")


def main():
    parser = argparse.ArgumentParser(description='Manage pipeline checkpoint')
    parser.add_argument('action', choices=['show', 'clear'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'show':
        show_checkpoint()
    elif args.action == 'clear':
        clear_checkpoint()


if __name__ == "__main__":
    main()