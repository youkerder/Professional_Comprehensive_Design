@echo off
chcp 65001 >nul
title LabelImg - 实验一 猫脸标注

rem ============================================================
rem  一键启动 LabelImg，并预先配好图片目录 / 预设类别 / 保存目录
rem  路径相对本脚本所在目录（tools/），整个仓库可整体搬迁。
rem  labelImg 路径：优先用环境变量 LABELIMG_EXE，默认取 PATH 上的 labelImg。
rem  双击本文件即可。
rem ============================================================

if not defined LABELIMG_EXE set "LABELIMG_EXE=labelImg"
set "LABELIMG=%LABELIMG_EXE%"
if not exist "%LABELIMG%" (
    for /f "delims=" %%P in ('where %LABELIMG_EXE% 2^>nul') do set "LABELIMG=%%P"
)
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "IMGDIR=%ROOT%\datasets\cat_face\JPEGImages"
set "CLSFILE=%ROOT%\datasets\cat_face\predefined_classes.txt"
set "SAVEDIR=%ROOT%\datasets\cat_face\Annotations"

echo ============================================================
echo   LabelImg - 实验一 猫脸标注
echo ============================================================
echo.
echo   图片目录 : %IMGDIR%
echo   保存目录 : %SAVEDIR%
echo   预设类别 : cat
echo.
echo   【重要】左下角按钮确认显示的是 PascalVOC
echo            如果显示 YOLO，点一下切换过来，否则导出的格式
echo            训练脚本读不了。
echo.
echo   每张图已经有一个预标注的红框（整只猫）。
echo   你要做的是把它收紧到「猫脸」：拖四个角，标签保持 cat。
echo.
echo   快捷键： w 画框   d 下一张   a 上一张
echo            Ctrl+S 保存   Delete 删框
echo            Ctrl+滚轮 缩放（已修复崩溃）
echo.

if not exist "%LABELIMG%" (
    echo   [错误] 找不到 labelImg：%LABELIMG_EXE%
    echo.
    echo   请先安装（任意 Python 3.9~3.11 环境）：
    echo     python -m pip install --only-binary=:all: labelImg lxml
    echo   或先设置环境变量 LABELIMG_EXE 指向 labelImg.exe。
    echo.
    pause
    exit /b 1
)

if not exist "%IMGDIR%" (
    echo   [错误] 找不到图片目录：%IMGDIR%
    echo.
    pause
    exit /b 1
)

if not exist "%CLSFILE%" (
    echo   [提示] 没有预设类别文件，会自动用默认的，可忽略。
)

rem 用 start 让 labelImg 独立运行，关掉本窗口不会影响它
start "" "%LABELIMG%" "%IMGDIR%" "%CLSFILE%" "%SAVEDIR%"

exit /b 0
