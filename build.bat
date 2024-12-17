if not exist builds\ mkdir builds\
if not exist debug\ mkdir debug\
set CFLAGS=-W
set CLIBS=-L./include/ -I./include/
set COMPILER=g++
set raylibInclude=-IC:/addtional-libs/raylib-5.0_win64_mingw-w64/include/ -LC:/addtional-libs/raylib-5.0_win64_mingw-w64/lib/ -lraylib -lgdi32 -lwinmm 

@REM %COMPILER% %CFLAGS% "./img2matrix.cpp" %CLIBS% -o "./builds/img2matrix"
@REM "./builds/img2matrix.exe"
@REM  "./mnist/train/100.png"
@REM %COMPILER% %CFLAGS% "./mnist/main.cpp" %CLIBS% -o "./builds/mnist-main"
@REM "./builds/mnist-main"

@REM %COMPILER% %CFLAGS% "./gym.cpp" %CLIBS% -o "./test150" %raylibInclude%
@REM "./test150" "./network.arch" "./layers.functions" "examples/testing/save-method/t.mat"

@REM %COMPILER% %CFLAGS% "./examples/xor_gym.cpp" %CLIBS% -o "./builds/xor_gym_new" %raylibInclude%
@REM "./builds/xor_gym_new"

@REM %COMPILER% %CFLAGS% "./img2matrix.cpp" %CLIBS% -o "./debug/img2mat"
@REM "./debug/img2mat" "mnist\train\100.png"

%COMPILER% %CFLAGS% "./main.cpp" %CLIBS% -o "./debug/imgTest1-sativkChainDemo" %raylibInclude%
"./debug/imgTest1-sativkChainDemo" "mnist/train/100.png" "./network.arch" "./layers.functions" "img.mat"