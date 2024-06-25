add_llvm_pass_plugin(PolygeistCUDALaunchFixUp
  CUDALaunchFixUp.cpp

  DEPENDS
  intrinsics_gen
  BUILDTREE_ONLY
)

add_llvm_pass_plugin(PolygeistAffineFromLLVM
  AffineFromLLVM.cpp

  DEPENDS
  intrinsics_gen
  PolygeistTransforms
  BUILDTREE_ONLY
)
