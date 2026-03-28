cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "osrm")
set(EXT_NAME "osrm_ext")
project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT SKBUILD)
  message(WARNING
  "This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  ====================================================================="
  )
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

set(SRCS
  src/osrm_nb.cpp
  src/engineconfig_nb.cpp
  src/preprocessing_nb.cpp
  src/extractorconfig_nb.cpp
  src/contractorconfig_nb.cpp
  src/partitionerconfig_nb.cpp
  src/customizerconfig_nb.cpp
  src/utility/osrm_utility.cpp
  src/utility/param_utility.cpp

  src/parameters/baseparameter_nb.cpp
  src/parameters/routeparameter_nb.cpp
  src/parameters/matchparameter_nb.cpp
  src/parameters/nearestparameter_nb.cpp
  src/parameters/tableparameter_nb.cpp
  src/parameters/tileparameter_nb.cpp
  src/parameters/tripparameter_nb.cpp
  
  src/types/optional_nb.cpp
  src/types/coordinate_nb.cpp
  src/types/jsoncontainer_nb.cpp
  src/types/approach_nb.cpp
  src/types/bearing_nb.cpp
)
nanobind_add_module(
  ${EXT_NAME}
  NB_STATIC
  ${SRCS}
  ${UTIL}
  ${PARAMS}
)

target_include_directories(${EXT_NAME} 
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utility
    ${CMAKE_CURRENT_SOURCE_DIR}/include/parameters
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Build OSRM from source via FetchContent
include(FetchContent)
FetchContent_Declare(
  libosrm
  GIT_REPOSITORY https://github.com/Project-OSRM/osrm-backend
  GIT_TAG v6.0.0
)


# Suppress warnings from OSRM build + Lua compatibility flags
if (MSVC)
  # Silence noisy OSRM warnings on MSVC
  add_compile_options(
    /wd4267  # conversion from 'size_t' to smaller type
    /wd4365  # signed/unsigned mismatch
    /wd4244  # possible loss of data (int/float narrowing)
    /wd4371  # layout changed due to packing (std::pair, ABI)
    /wd4996  # deprecated (POSIX/CRT, getenv, strdup, etc.)
    /wd4146  # unary minus on unsigned (Boost)
    /wd4459  # local hides global (Boost, sol2)
    /wd4702  # unreachable code (template instantiations)
  )
  add_compile_options(/external:W0 /external:anglebrackets)
  add_compile_options(/experimental:external)

  # Remove GCC-style -Wall if it sneaks in
  string(REPLACE "-Wall" "" _cxxflags "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${_cxxflags}")

  # Lua compatibility for OSRM v6.0.0's sol2 library (MSVC syntax)
  add_compile_options(
    /ULUA_COMPAT_BITLIB
    /ULUA_COMPAT_5_2
    /ULUA_COMPAT_5_3
    /DLUA_COMPAT_BITLIB=0
    /DLUA_COMPAT_5_2=1
    /DLUA_COMPAT_5_3=1
  )
else()
  # GCC/Clang only
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-array-bounds -Wno-uninitialized -Wno-free-nonheap-object")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ULUA_COMPAT_BITLIB -ULUA_COMPAT_5_2 -ULUA_COMPAT_5_3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLUA_COMPAT_BITLIB=0 -DLUA_COMPAT_5_2=1 -DLUA_COMPAT_5_3=1")
endif()


# --- Lua: pin to 5.4 on macOS (keg-only, needs explicit prefix) ---
if(APPLE)
  execute_process(
    COMMAND brew --prefix lua@5.4
    OUTPUT_VARIABLE _brew_lua_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_brew_lua_prefix)
    list(PREPEND CMAKE_PREFIX_PATH "${_brew_lua_prefix}")
  endif()
endif()

# Find Lua and ensure include directory is set for OSRM's sol2
find_package(Lua REQUIRED)

# Fallback logic for Lua paths (Windows only)
if (WIN32)
  if (NOT LUA_LIBRARIES OR NOT LUA_INCLUDE_DIR)
    set(LUA_LIBRARIES "C:/vcpkg/installed/x64-windows/lib/lua.lib")
    set(LUA_INCLUDE_DIR "C:/vcpkg/installed/x64-windows/include")
  endif()
endif()

message(STATUS "LUA_LIBRARIES: ${LUA_LIBRARIES}")
message(STATUS "LUA_INCLUDE_DIR: ${LUA_INCLUDE_DIR}")

# Manually specify Lua paths if not found
if(NOT LUA_INCLUDE_DIR OR NOT LUA_LIBRARIES)
  set(LUA_INCLUDE_DIR "C:/tools/lua/include")
  set(LUA_LIBRARIES "C:/tools/lua/lib/lua.lib")
  message(WARNING "Lua paths manually set. Ensure these paths are correct.")
endif()

FetchContent_GetProperties(libosrm)
if(NOT libosrm_POPULATED)
  FetchContent_Populate(libosrm)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
  
  # Patch sol2 to handle empty LUA_COMPAT macros (common in Lua 5.3+)
  set(SOL2_FILE "${libosrm_SOURCE_DIR}/third_party/sol2/include/sol/sol.hpp")
  file(READ "${SOL2_FILE}" SOL2_CONTENT)
  string(REPLACE 
    "#if (SOL_LUA_VERSION_I_ == 502) || (defined(LUA_COMPAT_BITLIB) && (LUA_COMPAT_BITLIB != 0))"
    "#if (SOL_LUA_VERSION_I_ == 502) || (defined(LUA_COMPAT_BITLIB) && LUA_COMPAT_BITLIB+0 != 0)"
    SOL2_CONTENT "${SOL2_CONTENT}")
  string(REPLACE
    "(SOL_LUA_VERSION_I_ < 504 && (defined(LUA_COMPAT_5_2) && (LUA_COMPAT_5_2 != 0)))"
    "(SOL_LUA_VERSION_I_ < 504 && (defined(LUA_COMPAT_5_2) && LUA_COMPAT_5_2+0 != 0))"
    SOL2_CONTENT "${SOL2_CONTENT}")
  file(WRITE "${SOL2_FILE}" "${SOL2_CONTENT}")
  
  add_subdirectory(
    ${libosrm_SOURCE_DIR}
    ${libosrm_BINARY_DIR}
  )
endif()

# Silence warning spam from OSRM compilation on MSVC
if (MSVC)
  foreach(tgt
    osrm
    osrm_extract
    osrm_contract
    osrm_partition
    osrm_customize
  )
    if (TARGET ${tgt})
      target_compile_options(${tgt} PRIVATE /W0)
    endif()
  endforeach()
endif()

target_include_directories(${EXT_NAME} 
  PUBLIC
    ${libosrm_SOURCE_DIR}/include
    ${libosrm_SOURCE_DIR}/third_party/flatbuffers/include
)

target_link_libraries(${EXT_NAME}
  PRIVATE
    osrm
    osrm_extract
    osrm_contract
    osrm_partition
    osrm_customize
)

if (MSVC)
  add_definitions(-DBOOST_ALL_NO_LIB)
endif()

# Install Python source files
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/osrm/__init__.py
              ${CMAKE_CURRENT_SOURCE_DIR}/src/osrm/__main__.py
              ${CMAKE_CURRENT_SOURCE_DIR}/src/osrm/preprocessing.py
        DESTINATION osrm)

# Install Lua profiles as package data
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/osrm/profiles/
        DESTINATION osrm/profiles
        FILES_MATCHING PATTERN "*.lua"
        PATTERN "*.lua" PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)

# Install the extension module
install(TARGETS ${EXT_NAME} LIBRARY DESTINATION osrm)
