@echo off
chcp 65001 >nul
call "%~dp0env.cmd"
cd /d "%EXP%\code"

echo ================================================================
echo  步骤 4/6  对 10 段测试视频计算 Precision / Recall / F-score
echo  权重: %EXP%\models\fcsn_final.pkl
echo  明细: %EXP%\outputs\metrics\prf_test10.csv
echo  预测: %EXP%\outputs\scores\final_test10.json
echo ================================================================
"%PY%" evaluate_summary.py ^
  --ckpt     "%EXP%\models\fcsn_final.pkl" ^
  --ids      "%EXP%\logs\test_ids.txt" ^
  --n        10 ^
  --json_out "%EXP%\outputs\scores\final_test10.json" ^
  --csv_out  "%EXP%\outputs\metrics\prf_test10.csv" ^
  --gpu
echo.
echo 完成，按任意键退出...
pause >nul
