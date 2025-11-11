apt update
apt install llvm clang
rm -rf build && mkdir build && cd build
cp ../cmake/config.cmake .
cmake .. && cmake --build . --parallel $(nproc)
