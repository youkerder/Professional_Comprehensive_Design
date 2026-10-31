@echo off
chcp 65001 >nul
rem 实验 04 · 视频检索 —— 四个查询图的检索演示（outputs\results\<query>\）
cd /d "%~dp0"
if not exist "%~dp0..\..\datasets\retrieval\images" (
    echo [错误] 先运行 run_step1_make_dataset.cmd 生成数据集
    exit /b 1
)
python code\search.py --query sunset_sea_q0
python code\search.py --query city_night_q1
python code\search.py --query fireworks_q0
python code\search.py --query aquarium_q1
echo.
echo 完成：检索演示
pause
