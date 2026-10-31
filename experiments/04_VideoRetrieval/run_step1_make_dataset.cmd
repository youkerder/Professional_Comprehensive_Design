@echo off
chcp 65001 >nul
rem 实验 04 · 视频检索 —— 合成本地小型检索测试库（12 视频 + 120 图 + 24 查询图）
cd /d "%~dp0"
if not exist "%~dp0..\..\datasets\retrieval\images" (
    echo [错误] 先运行 run_step1_make_dataset.cmd 生成数据集
    exit /b 1
)
python code\make_dataset.py
echo.
echo 完成：合成本地小型检索测试库
pause
