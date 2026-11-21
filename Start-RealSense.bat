@echo off
setlocal
title RealSense RGB-D Publisher
if not exist "%~dp0runtime\python.exe" (
    echo Extract the entire ZIP, then run this batch file from the extracted folder.
    pause
    exit /b 1
)
"%~dp0runtime\python.exe" -B -u "%~dp0realsense_publisher.py"
set "publisher_exit=%errorlevel%"
if not "%publisher_exit%"=="0" (
    echo.
    echo Publisher exited with code %publisher_exit%.
    pause
)
exit /b %publisher_exit%
