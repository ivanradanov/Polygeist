#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"


using namespace llvm;
using namespace mlir;


static void optUsingAffine(Module &M) {
  auto clonedModule = CloneModule(M);

  mlir::registerAllPasses();
  mlir::registerAllTranslations();
  mlir::DialectRegistry registry;
  mlir::registerOpenMPDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllFromLLVMIRTranslations(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  MLIRContext context(registry);

  auto mlirModule = mlir::translateLLVMIRToModule(std::move(clonedModule), &context);
  std::unique_ptr<Module> transformedModule = mlir::translateModuleToLLVMIR(mlirModule.get(), M.getContext());

  {
    M.getFunctionList().clear();
    SmallVector<GlobalVariable *> GVs;
    SmallVector<GlobalIFunc *> IFuncs;
    for (GlobalVariable &GV : M.globals())
      GVs.push_back(&GV);
    for (GlobalIFunc &IFunc : M.ifuncs())
      IFuncs.push_back(&IFunc);
    for (GlobalVariable *GV : GVs)
      GV->removeFromParent();
    for (GlobalIFunc *IFunc : IFuncs)
      IFunc->removeFromParent();
  }

  SmallVector<GlobalVariable *> GVs;
  SmallVector<Function *> Funcs;
  SmallVector<GlobalIFunc *> IFuncs;
  for (GlobalVariable &GV : transformedModule->globals())
    GVs.push_back(&GV);
  for (Function &F : *transformedModule)
    Funcs.push_back(&F);
  for (GlobalIFunc &IFunc : transformedModule->ifuncs())
    IFuncs.push_back(&IFunc);

  for (GlobalVariable *GV : GVs) {
    GV->removeFromParent();
    M.insertGlobalVariable(GV);
  }
  for (Function *F : Funcs) {
    F->removeFromParent();
    M.getFunctionList().insert(M.getFunctionList().end(), F);
  }
  for (GlobalIFunc *IFunc : IFuncs) {
    IFunc->removeFromParent();
    M.insertIFunc(IFunc);
  }
  // for (GlobalAlias *Alias : Aliases) {
  //   Alias->removeFromParent();
  //   M.insertAlias(Alias);
  // }
}

struct AffineFromLLVM : PassInfoMixin<AffineFromLLVM> {
  llvm::PreservedAnalyses run(Module &M, llvm::ModuleAnalysisManager &) {
    optUsingAffine(M);
    return llvm::PreservedAnalyses::none();
  }
};

llvm::PassPluginLibraryInfo getAffineFromLLVMPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CUDALaunchFixUp", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerOptimizerEarlyEPCallback(
                [](llvm::ModulePassManager &PM, OptimizationLevel Level) {
                  PM.addPass(AffineFromLLVM());
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::ModulePassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "affine-from-llvm") {
                    PM.addPass(AffineFromLLVM());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getAffineFromLLVMPluginInfo();
}
