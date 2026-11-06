cmake_minimum_required(VERSION 3.20)
project(nini_datacomp LANGUAGES C CXX)

# Biblioteca compartilhada com ABI em C (cpp/datacomp.h), carregada pelo Python via ctypes.
# Gera libdatacomp.so (Linux), libdatacomp.dylib (macOS) ou datacomp.dll (Windows) dentro do pacote nini_datacomp.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)       # só as funções DC_EXPORT ficam visíveis
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# MSVC: runtime do C++ embutido na DLL (/MT). Sem isto, datacomp.dll depende do msvcp140.dll, que o Python não traz e
# que só existe em máquinas com o Visual C++ Redistributable instalado.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

option(DC_NATIVE "Compilar para a CPU desta máquina (-march=native); nunca para wheels publicados" OFF)

# ReadStat (C, licença MIT) para ler SAS, Stata e SPSS (cpp/stat.cpp): só os leitores, compilados dentro da
# biblioteca, com um iconv próprio (third_party/shim: o mesmo resultado em qualquer sistema, nada a instalar no
# Windows) e o descompressor do miniz no lugar do zlib (os .zsav). Nada disso é exportado.
set(DC_TP "${CMAKE_CURRENT_SOURCE_DIR}/cpp/third_party")
file(GLOB_RECURSE DC_READSTAT CONFIGURE_DEPENDS "${DC_TP}/readstat/*.c")
add_library(dc_readstat OBJECT ${DC_READSTAT} "${DC_TP}/shim/iconv.c" "${DC_TP}/shim/zlib_shim.c"
            "${DC_TP}/miniz/miniz_tinfl.c")
set_target_properties(dc_readstat PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON
                      C_VISIBILITY_PRESET hidden)
# o shim vem antes: o <iconv.h> e o <zlib.h> do ReadStat são os nossos, não os do sistema
target_include_directories(dc_readstat BEFORE PRIVATE "${DC_TP}/shim" "${DC_TP}/readstat" "${DC_TP}/miniz")
target_compile_definitions(dc_readstat PRIVATE HAVE_ZLIB=1)
if(MSVC)
  # código de terceiros: sem os avisos (o ReadStat usa as funções "inseguras" do CRT)
  target_compile_options(dc_readstat PRIVATE /W0 /utf-8)
  target_compile_definitions(dc_readstat PRIVATE _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE)
else()
  target_compile_options(dc_readstat PRIVATE -O2 -w)
endif()

file(GLOB DC_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp")
add_library(datacomp SHARED ${DC_SOURCES} $<TARGET_OBJECTS:dc_readstat>)
target_include_directories(datacomp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
# o pool de threads do C++ (cpp/pool.cpp): -pthread no Linux; no Windows e no macOS, nada a mais
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(datacomp PRIVATE Threads::Threads)

set(DC_X86 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64")
  set(DC_X86 ON)
endif()

# Mesmo resultado em qualquer sistema: a tolerância |l - r| <= abs_tol + rel_tol * |r| não pode depender da CPU nem do
# compilador, inclusive exatamente na borda. Sem fusão de multiplicação e soma (-ffp-contract=off: FMA arredonda
# diferente, e o ARM e as CPUs com FMA fundiriam a*b + c por conta própria) e sem matemática "rápida".
# -fno-math-errno e -fno-trapping-math não mudam nenhum resultado (só dizem que ninguém lê errno nem captura exceções
# de ponto flutuante) e deixam os laços vetorizar.
if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # clang-cl (DC_CLANGCL=1 no Windows): opções do clang passadas com /clang:
  target_compile_options(datacomp PRIVATE /utf-8 /W4 /clang:-O3 /clang:-ffp-contract=off /clang:-fno-math-errno /clang:-fno-trapping-math)
  if(DC_NATIVE)
    target_compile_options(datacomp PRIVATE /clang:-march=native)
  elseif(DC_X86)
    target_compile_options(datacomp PRIVATE /clang:-march=x86-64-v2)
  endif()
  target_compile_definitions(datacomp PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(MSVC)
  # /utf-8: os comentários têm acentos; /O2 já vem do Release.
  # /fp:precise (padrão desde o VS 2022) não funde multiplicação e soma.
  target_compile_options(datacomp PRIVATE /utf-8 /W4 /permissive- /Zc:__cplusplus /fp:precise)
  target_compile_definitions(datacomp PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(datacomp PRIVATE -O3 -Wall -Wextra -Wpedantic -ffp-contract=off -fno-math-errno -fno-trapping-math)
  if(DC_NATIVE)
    target_compile_options(datacomp PRIVATE -march=native)
  elseif(DC_X86)
    # base dos wheels: x86-64-v2 (SSE4.2/POPCNT). AVX2 entra depois (fase 5) como segunda cópia escolhida na hora,
    # só nos kernels em que o perfil mandar; FMA fica de fora de propósito (veja acima).
    target_compile_options(datacomp PRIVATE -march=x86-64-v2)
  endif()
endif()

if(WIN32)
  set_target_properties(datacomp PROPERTIES PREFIX "")    # datacomp.dll
endif()

# scikit-build-core coloca o conteúdo de CMAKE_INSTALL_PREFIX na raiz do wheel
install(TARGETS datacomp
        LIBRARY DESTINATION nini_datacomp
        RUNTIME DESTINATION nini_datacomp)
