#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"

using namespace llvm;


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
  registerDialects(Ctx, options.getCgeistOpts());

  auto mlirModule = mlir::translateLLVMIRToModule(clonedModule, context);
  auto transformedModule = mlir::translateModuleToLLVMIR(mlirModule, M.getContext());
}

struct AffineFromLLVM : PassInfoMixin<AffineFromLLVM> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    optUsingAffine(M);
    return PreservedAnalyses::none();
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
