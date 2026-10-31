@echo off
chcp 65001 >nul
call "%~dp0env.cmd"
set "PYTHONPATH=%PYTHONPATH%;%EXP%\env_tensorboard"

echo ================================================================
echo  步骤 6/6  TensorBoard（查看训练 loss 与每 5 epoch 的 P/R/F）
echo  日志目录: %EXP%\logs\tensorboard
echo  启动后浏览器打开: http://127.0.0.1:6006
echo  停止: 在本窗口按 Ctrl+C
echo ================================================================
"%PY%" -m tensorboard.main --logdir "%EXP%\logs\tensorboard" --port 6006 --bind_all
pause >nul
