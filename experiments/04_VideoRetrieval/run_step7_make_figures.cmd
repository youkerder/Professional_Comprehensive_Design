@echo off
chcp 65001 >nul
rem 实验 04 · 视频检索 —— 生成报告插图（reports\figures\04_*.png）
cd /d "%~dp0"
if not exist "%~dp0..\..\datasets\retrieval\images" (
    echo [错误] 先运行 run_step1_make_dataset.cmd 生成数据集
    exit /b 1
)
python code\make_figures.py
echo.
echo 完成：生成报告插图
pause
