cmake_minimum_required(VERSION 3.20)
project(nini_datacomp LANGUAGES CXX)

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

file(GLOB DC_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp")
add_library(datacomp SHARED ${DC_SOURCES})
target_include_directories(datacomp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpp")

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
