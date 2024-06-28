if not exist builds\ mkdir builds\
set CFLAGS=-W
set CLIBS=-L./include/ -I./include/
set COMPILER=g++
set raylibInclude=-IC:/addtional-libs/raylib-5.0_win64_mingw-w64/include/ -LC:/addtional-libs/raylib-5.0_win64_mingw-w64/lib/ -lraylib -lgdi32 -lwinmm 
@REM cls
@REM %COMPILER% %CFLAGS% "./examples/gui.cpp" %CLIBS% -o "./builds/gui.exe"
@REM "./builds/gui.exe"

@REM %COMPILER% %CFLAGS% "./examples/xor_trained_using_own_framework.cpp" %CLIBS% -o "./builds/xor_trained_using_own_framework.exe"
@REM "./builds/xor_trained_using_own_framework.exe"

@REM %COMPILER% %CFLAGS% "./img2matrix.cpp" %CLIBS% -o "./builds/img2matrix"

@REM "./builds/img2matrix.exe" "./mnist/train/100.png"
@REM %COMPILER% %CFLAGS% "./mnist/main.cpp" %CLIBS% -o "./builds/mnist-main"
@REM "./builds/mnist-main"


@REM  for GUI...  raylib
@REM %COMPILER% %CFLAGS% "./examples/GUI.cpp" %CLIBS% -o "./builds/gui" %raylibInclude%
@REM "./builds/gui"
@REM %COMPILER% %CFLAGS% "./gym.cpp" %CLIBS% -o "./builds/gym" %raylibInclude%
@REM "./builds/gym"
%COMPILER% %CFLAGS% "./examples/xor_gym.cpp" %CLIBS% -o "./builds/xor_gym" %raylibInclude%
"./builds/xor_gym"