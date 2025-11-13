@echo off
REM === Infinite loop to call download_d03.py every hour ===

:loop
echo [%date% %time%] Running Python script...
python310 download_d03.py

echo [%date% %time%] Sleeping for 1 hour...
timeout /t 3600 /nobreak >nul

goto loop