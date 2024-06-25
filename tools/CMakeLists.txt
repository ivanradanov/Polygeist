add_subdirectory(polygeist-merge-gpu-modules)
add_subdirectory(polygeist-opt)

if(POLYGEIST_ENABLE_POLYMER)
  add_subdirectory(polymer)
endif()
