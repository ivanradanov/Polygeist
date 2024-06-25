add_llvm_pass_plugin(PolygeistCUDALaunchFixUp
  CUDALaunchFixUp.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  intrinsics_gen
  BUILDTREE_ONLY
)

add_llvm_pass_plugin(PolygeistAffineFromLLVM
  AffineFromLLVM.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  intrinsics_gen
  MLIRPolygeist
  MLIRSupport
  MLIRIR
  MLIRAnalysis
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIROpenMPDialect
  MLIRGPUTransforms
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRFunctionInterfaces
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRMathToLLVM
  MLIRTargetLLVMIRImport
  MLIRSCFTransforms
  MLIRPolygeistTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToOpenMP
  MLIROpenMPToLLVM
  MLIROpenMPToLLVMIRTranslation
  MLIRGPUToNVVMTransforms
  MLIRGPUToGPURuntimeTransforms
  MLIRFuncAllExtensions
  MLIRLLVMIRToLLVMTranslation
  MLIRFromLLVMIRTranslationRegistration
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRCAPIRegisterEverything
  MLIRBuiltinToLLVMIRTranslation

  BUILDTREE_ONLY
)
