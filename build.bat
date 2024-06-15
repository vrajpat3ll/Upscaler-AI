if not exist builds\ mkdir builds\
set CFLAGS=-W
set CLIBS=-L./include/ -I./include/
set COMPILER=g++
@REM cls
@REM %COMPILER% %CFLAGS% "./examples/gui.cpp" %CLIBS% -o "./builds/gui.exe"
@REM "./builds/gui.exe"

@REM %COMPILER% %CFLAGS% "./examples/xor_trained_using_own_framework.cpp" %CLIBS% -o "./builds/xor_trained_using_own_framework.exe"
@REM "./builds/xor_trained_using_own_framework.exe"

%COMPILER% %CFLAGS% "./img2matrix.cpp" %CLIBS% -o "./builds/img2matrix"

"./builds/img2matrix.exe" "./mnist/train/100.png"