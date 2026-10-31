@echo off
chcp 65001 >nul
rem 实验 04 · 视频检索 —— 提取 ResNet50 深度特征与 HSV 直方图基线特征
cd /d "%~dp0"
if not exist "%~dp0..\..\datasets\retrieval\images" (
    echo [错误] 先运行 run_step1_make_dataset.cmd 生成数据集
    exit /b 1
)
python code\extract_features.py
echo.
echo 完成：提取 ResNet50 深度特征与 HSV 直方图基线特征
pause
