@echo off
chcp 65001 >nul
call "%~dp0env.cmd"
cd /d "%EXP%\code"

echo ================================================================
echo  步骤 5/6  生成 10 段测试视频的摘要视频 / 关键帧 / 得分柱状图
echo  输入: %EXP%\outputs\scores\final_test10.json + %DATA%\video
echo  输出: %EXP%\outputs\summary\<video_id>\summary.avi
echo        %EXP%\outputs\summary\<video_id>\keyframes\*.jpg
echo        %EXP%\outputs\summary\<video_id>\score_bar.png
echo        %EXP%\outputs\summary\score_bars_all.png
echo ================================================================
"%PY%" gen_summary.py ^
  --h5_path  "%DATA%\fcsn_tvsum.h5" ^
  -j         "%EXP%\outputs\scores\final_test10.json" ^
  -r         "%DATA%\video" ^
  -s         "%EXP%\outputs\summary" ^
  --ids      "%EXP%\logs\test_ids.txt" ^
  --n        10 ^
  --fps      24 ^
  -b
echo.
echo 完成，按任意键退出...
pause >nul
