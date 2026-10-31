@echo off
REM ============================================================
REM  实验 FCSN 视频摘要 —— 环境变量（每个脚本都会先 call 本文件）
REM  Python 解释器：默认用 PATH 上的 python；
REM  需要指定时先 set PYTHON_EXE=D:\path\to\python.exe 再运行脚本。
REM  路径全部由本文件位置推导（env.cmd 在实验根目录下）。
REM ============================================================
if not defined PYTHON_EXE set "PYTHON_EXE=python"
set "PY=%PYTHON_EXE%"
set "EXP=%~dp0"
set "EXP=%EXP:~0,-1%"
for %%I in ("%~dp0..\..") do set "WS=%%~fI"
set "DATA=%WS%\datasets\tvsum50"
set "PYTHONPATH=%EXP%\env_pkgs"
set "TORCH_HOME=%EXP%\pretrained"
set "PYTHONIOENCODING=utf-8"
set "KMP_DUPLICATE_LIB_OK=TRUE"
