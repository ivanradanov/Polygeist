#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include <cstring>
#include <polygeist/Tools/MergeHostDeviceGPUModules.h>

using namespace mlir;

namespace {
constexpr char gpuModuleName[] = "__polygeist_gpu_module";
constexpr char kernelPrefix[] = "__polygeist_launch_kernel_";
} // namespace

LogicalResult mlir::polygeist::mergeDeviceIntoHost(ModuleOp hostModule,
                                                   ModuleOp deviceModule) {
  if (hostModule->walk([](gpu::GPUModuleOp) { return WalkResult::interrupt(); })
          .wasInterrupted()) {
    return failure();
  }
  llvm::SmallVector<LLVM::LLVMFuncOp> launchFuncs;
  hostModule->walk([&](LLVM::LLVMFuncOp funcOp) {
    auto symName = funcOp.getName();
    if (symName.startswith(kernelPrefix))
      launchFuncs.push_back(funcOp);
  });

  auto ctx = hostModule.getContext();

  auto moduleBuilder = OpBuilder::atBlockBegin(hostModule.getBody());
  auto gpuModule = moduleBuilder.create<gpu::GPUModuleOp>(
      deviceModule->getLoc(), gpuModuleName);
  gpuModule.getRegion().takeBody(deviceModule.getRegion());
  // TODO get these target attrs from somewhere
  auto target = moduleBuilder.getAttr<NVVM::NVVMTargetAttr>(
      /*optLevel=*/2, /*triple=*/"nvptx64-nvidia-cuda", "sm_80", "+ptx60",
      /*flags=*/nullptr,
      /*linkLibs=*/nullptr);
  gpuModule.setTargetsAttr(moduleBuilder.getArrayAttr({target}));

  auto gpuModuleBuilder = OpBuilder::atBlockEnd(gpuModule.getBody());
  gpuModuleBuilder.create<gpu::ModuleEndOp>(gpuModule->getLoc());

  for (auto launchFunc : launchFuncs) {
    auto launchFuncUses = launchFunc.getSymbolUses(hostModule);
    for (auto use : *launchFuncUses) {
      if (auto callOp = dyn_cast<LLVM::CallOp>(use.getUser())) {
        auto loc = callOp->getLoc();
        OpBuilder builder(callOp);
        StringRef callee =
            cast<LLVM::AddressOfOp>(
                callOp.getCalleeOperands().front().getDefiningOp())
                .getGlobalName();
        int symbolLength = 0;
        if (callee.consume_front("_Z"))
          callee.consumeInteger(/*radix=*/10, symbolLength);
        const char stubPrefix[] = "__device_stub__";
        callee.consume_front(stubPrefix);

        // LLVM::LLVMFuncOp gpuFuncOp =
        // cast<LLVM::LLVMFuncOp>(deviceModule.lookupSymbol(callee));
        std::string deviceSymbol;
        if (symbolLength)
          deviceSymbol = "_Z" +
                         std::to_string(symbolLength - strlen(stubPrefix)) +
                         callee.str();
        else
          deviceSymbol = callee;
        SymbolRefAttr gpuFuncSymbol = SymbolRefAttr::get(
            StringAttr::get(ctx, gpuModuleName),
            {SymbolRefAttr::get(StringAttr::get(ctx, deviceSymbol.c_str()))});
        auto deviceFunc = dyn_cast_or_null<LLVM::LLVMFuncOp>(
            hostModule.lookupSymbol(gpuFuncSymbol));
        if (!deviceFunc)
          return deviceFunc.emitError();
        deviceFunc->setAttr("gpu.kernel", builder.getUnitAttr());
        deviceFunc->setAttr("nvvm.kernel", builder.getUnitAttr());
        auto shMemSize = builder.create<LLVM::TruncOp>(
            loc, builder.getI32Type(), callOp.getArgOperands()[7]);
        // TODO stream is arg 8
        llvm::SmallVector<Value> args;
        for (unsigned i = 9; i < callOp.getArgOperands().size(); i++)
          args.push_back(callOp.getArgOperands()[i]);
        builder.create<gpu::LaunchFuncOp>(
            loc, gpuFuncSymbol,
            gpu::KernelDim3(
                {builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[1]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[2]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[3])}),
            gpu::KernelDim3(
                {builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[4]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[5]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[6])}),
            shMemSize,
            // TODO need stream
            ValueRange(args));
        callOp->erase();
      }
    }
  }
  if (launchFuncs.size())
    hostModule->setAttr("gpu.container_module", OpBuilder(ctx).getUnitAttr());
  return success();
}
