# compiling matrix.hpp to a kind of object file
g++ "./include/matrix.hpp"
mkdir lib
mv  "./include/matrix.hpp.gch" "./lib/"
#
file_to_compile=$1
g++ `file_to_compile` 