@echo off
chcp 65001 >nul
title LabelImg - 剩余待标注

rem ============================================================
rem  只打开「还没改过的」那些图，省得在 205 个文件里翻找。
rem  保存目录仍然是正式的 Annotations，存下去就是正式标注。
rem  路径相对本脚本所在目录（tools/）。
rem ============================================================

if not defined LABELIMG_EXE set "LABELIMG_EXE=labelImg"
set "LABELIMG=%LABELIMG_EXE%"
if not exist "%LABELIMG%" (
    for /f "delims=" %%P in ('where %LABELIMG_EXE% 2^>nul') do set "LABELIMG=%%P"
)
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "IMGDIR=%ROOT%\experiments\01_ObjectDetection_YOLOv3\outputs\remaining_to_label"
set "CLSFILE=%ROOT%\datasets\cat_face\predefined_classes.txt"
set "SAVEDIR=%ROOT%\datasets\cat_face\Annotations"

echo ============================================================
echo   剩余待标注（只列还没改过的图）
echo ============================================================
echo.
echo   图片目录 : %IMGDIR%
echo   保存目录 : %SAVEDIR%   （正式的，存下去就是正式标注）
echo.
echo   这些图当前还是「整只猫」的预标注框，请收紧到「猫脸」。
echo   其中 204.jpg 没有任何框，需要按 w 从头画。
echo.
echo   快捷键： w 画框   d 下一张   a 上一张
echo            Ctrl+S 保存   Delete 删框   Ctrl+滚轮 缩放
echo.

if not exist "%LABELIMG%" (
    echo   [错误] 找不到 labelImg：%LABELIMG_EXE%
    echo   请先 pip install labelImg lxml，或设置 LABELIMG_EXE 指向 labelImg.exe
    pause
    exit /b 1
)
if not exist "%IMGDIR%" (
    echo   [错误] 找不到图片目录：%IMGDIR%
    echo          先运行：
    echo            python experiments\01_ObjectDetection_YOLOv3\code\list_remaining.py
    pause
    exit /b 1
)

start "" "%LABELIMG%" "%IMGDIR%" "%CLSFILE%" "%SAVEDIR%"
exit /b 0
