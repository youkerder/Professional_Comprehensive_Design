@echo off
chcp 65001 >nul
call "%~dp0env.cmd"
cd /d "%EXP%\code"

echo ================================================================
echo  步骤 2/6  make_dataset.py  提取 320 帧 GoogLeNet pool5 特征
echo  输入: %DATA%\video\*.mp4 + eccv16 等价 h5
echo  输出: %DATA%\fcsn_tvsum.h5
echo ================================================================
"%PY%" make_dataset.py ^
  --video_dir  "%DATA%\video" ^
  --h5_path    "%DATA%\fcsn_tvsum.h5" ^
  --vsumm_data "%DATA%\eccv16_dataset_tvsum_google_pool5_local.h5" ^
  --gpu
echo.
echo 完成，按任意键退出...
pause >nul
