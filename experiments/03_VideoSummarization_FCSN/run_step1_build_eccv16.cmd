@echo off
chcp 65001 >nul
call "%~dp0env.cmd"
cd /d "%EXP%\code"

echo ================================================================
echo  步骤 1/6  离线重建 eccv16 等价数据集（官方 h5 在 Google Drive，本机无法下载）
echo  输出: %DATA%\eccv16_dataset_tvsum_google_pool5_local.h5
echo ================================================================
"%PY%" build_eccv16_tvsum.py ^
  --video_dir "%DATA%\video" ^
  --anno_path "%DATA%\data\ydata-tvsum50-anno.tsv" ^
  --h5_path   "%DATA%\eccv16_dataset_tvsum_google_pool5_local.h5" ^
  --gpu
echo.
echo 完成，按任意键退出...
pause >nul
