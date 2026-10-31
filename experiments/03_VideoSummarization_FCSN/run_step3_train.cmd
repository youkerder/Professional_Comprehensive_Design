@echo off
chcp 65001 >nul
call "%~dp0env.cmd"
cd /d "%EXP%\code"

echo ================================================================
echo  步骤 3/6  训练 FCSN
echo  数据: %DATA%\fcsn_tvsum.h5
echo  权重: %EXP%\models\epoch-*.pkl / fcsn_final.pkl
echo  日志: %EXP%\logs\tensorboard
echo  预测: %EXP%\outputs\scores\epoch-*.json
echo ================================================================
"%PY%" train.py
echo.
echo 完成，按任意键退出...
pause >nul
