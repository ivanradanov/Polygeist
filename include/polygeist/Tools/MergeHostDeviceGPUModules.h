#ifndef POLYGEIST_TOOLS_MERGEHOSTDEVICEGPUMODULES_H_
#define POLYGEIST_TOOLS_MERGEHOSTDEVICEGPUMODULES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::polygeist {
mlir::LogicalResult mergeDeviceIntoHost(mlir::ModuleOp hostModule,
                                                   mlir::ModuleOp deviceModule);
}


#endif // POLYGEIST_TOOLS_MERGEHOSTDEVICEGPUMODULES_H_
