@echo off
REM folder with standards_enriched.csv (full 23,813-standard catalogue)
REM keyword search scored best on our test (38/50)
set BIS_EMBED_MODEL=none
set BIS_DATA_DIR=D:\BIS_Assistant_Data\2026-10-01_sih
start "" http://127.0.0.1:5000
python app_local.py
pause
