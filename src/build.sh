#!/usr/bin/env sh
# Build the Path Finder runtime. Needs only a C/C++17 compiler (gcc/g++ >= 9 or clang).
#   sh build.sh            -> ./pathfinder (or pathfinder.exe on Windows/MSYS2)
set -eu
cd "$(dirname "$0")"
CC=${CC:-gcc}
CXX=${CXX:-g++}
OPT="-O2 -DNDEBUG -march=x86-64-v2"
mkdir -p ../build
$CC $OPT -c third_party/miniz/miniz.c -o ../build/miniz.o \
    -DMINIZ_NO_ZLIB_COMPATIBLE_NAMES -DMINIZ_NO_ARCHIVE_WRITING_APIS -DMINIZ_NO_TIME
LDFLAGS="-static"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) EXTRA="-lpsapi" ;;
  Darwin*) LDFLAGS=""; EXTRA="" ;;
  *) EXTRA="" ;;
esac
$CXX -std=c++17 $OPT -Wall -Wextra -Wno-unused-parameter \
    -Ithird_party/eigen -Ithird_party/miniz -I. \
    -DEIGEN_NO_DEBUG -DMINIZ_NO_ZLIB_COMPATIBLE_NAMES -DMINIZ_NO_ARCHIVE_WRITING_APIS -DMINIZ_NO_TIME \
    main.cpp ../build/miniz.o -o pathfinder $LDFLAGS $EXTRA
echo "built $(pwd)/pathfinder"
