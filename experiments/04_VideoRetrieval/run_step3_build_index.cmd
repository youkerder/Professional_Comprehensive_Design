@echo off
chcp 65001 >nul
rem 实验 04 · 视频检索 —— 建立四种检索索引并自检
cd /d "%~dp0"
if not exist "%~dp0..\..\datasets\retrieval\images" (
    echo [错误] 先运行 run_step1_make_dataset.cmd 生成数据集
    exit /b 1
)
python code\build_index.py
echo.
echo 完成：建立四种检索索引并自检
pause
