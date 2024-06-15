if not exist builds\ mkdir builds\
set CFLAGS=-W
set CLIBS=-L./include/ -I./include/
set COMPILER=g++
@REM cls
@REM %COMPILER% %CFLAGS% "./examples/gui.cpp" %CLIBS% -o "./builds/gui.exe"
@REM "./builds/gui.exe"

%COMPILER% %CFLAGS% "./examples/xor_trained_using_own_framework.cpp" %CLIBS% -o "./builds/xor_trained_using_own_framework.exe"
"./builds/xor_trained_using_own_framework.exe"