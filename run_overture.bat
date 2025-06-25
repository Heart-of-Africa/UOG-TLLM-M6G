@echo off
setlocal ENABLEEXTENSIONS

:: 显示许可协议简要摘要
echo.
echo ============================================================
echo            overture 许可协议 - 非商用、禁止再发布
echo ------------------------------------------------------------
echo 本软件仅授权用于个人用途，禁止商业使用与再发布。
echo 若您不同意此协议，请关闭本窗口。
echo ============================================================
echo.

:: 用户确认
set /p agree=是否同意并继续运行？(y/n): 
if /I NOT "%agree%"=="y" (
    echo 用户未同意协议，程序退出。
    exit /b
)

:: 输入训练层数
set /p layer=请输入要训练的层编号（例如 6）: 

:: 输入数据集路径
set /p dataset=请输入训练数据集的文件路径（例如 train.txt）: 

:: 执行训练命令
echo 正在启动训练...
python train.py --layer %layer% --dataset "%dataset%"

pause
