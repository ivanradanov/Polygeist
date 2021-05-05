#map0 = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (1600, d0 * 32 + 32)>
#map2 = affine_map<(d0) -> (1800, 2400, d0 * 32 + 32)>
#map3 = affine_map<(d0) -> (1800, d0 * 32 + 32)>
#map4 = affine_map<(d0) -> (2400, d0 * 32 + 32)>
#map5 = affine_map<(d0) -> (2200, d0 * 32 + 32)>
#set0 = affine_set<(d0) : (-d0 + 56 >= 0, -d0 + 74 >= 0)>
#set1 = affine_set<(d0) : (-d0 + 56 >= 0, d0 - 75 >= 0)>
#set2 = affine_set<(d0) : (-d0 + 74 >= 0, d0 - 57 >= 0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00")
  llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
  llvm.mlir.global internal constant @str5("%0.2lf \00")
  llvm.mlir.global internal constant @str4("\0A\00")
  llvm.mlir.global internal constant @str3("D\00")
  llvm.mlir.global internal constant @str2("begin dump: %s\00")
  llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
  llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @str0("\00")
  llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c1600_i32 = constant 1600 : i32
    %c1800_i32 = constant 1800 : i32
    %c2200_i32 = constant 2200 : i32
    %c2400_i32 = constant 2400 : i32
    %c42_i32 = constant 42 : i32
    %c0_i64 = constant 0 : i64
    %true = constant true
    %false = constant false
    %cst = constant 1.500000e+00 : f64
    %cst_0 = constant 1.200000e+00 : f64
    %c3_i32 = constant 3 : i32
    %c2_i32 = constant 2 : i32
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c1800 = constant 1800 : index
    %cst_1 = constant 0.000000e+00 : f64
    %0 = memref.alloca() : memref<1xf64>
    %1 = memref.alloca() : memref<1xf64>
    %2 = memref.alloc() : memref<1600x1800xf64>
    %3 = memref.alloc() : memref<1600x2200xf64>
    %4 = memref.alloc() : memref<2200x1800xf64>
    %5 = memref.alloc() : memref<1800x2400xf64>
    %6 = memref.alloc() : memref<1600x2400xf64>
    %7 = memref.cast %6 : memref<1600x2400xf64> to memref<?x2400xf64>
    affine.store %cst, %0[0] : memref<1xf64>
    affine.store %cst_0, %1[0] : memref<1xf64>
    %8:2 = scf.while (%arg2 = %c0_i32) : (i32) -> (i32, i32) {
      %16 = cmpi slt, %arg2, %c1600_i32 : i32
      scf.condition(%16) %c0_i32, %arg2 : i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
      %16 = index_cast %arg3 : i32 to index
      %17 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
        %19 = cmpi slt, %arg4, %c2200_i32 : i32
        scf.condition(%19) %arg4 : i32
      } do {
      ^bb0(%arg4: i32):  // no predecessors
        %19 = index_cast %arg4 : i32 to index
        %20 = muli %arg3, %arg4 : i32
        %21 = addi %20, %c1_i32 : i32
        %22 = remi_signed %21, %c1600_i32 : i32
        %23 = sitofp %22 : i32 to f64
        %24 = sitofp %c1600_i32 : i32 to f64
        %25 = divf %23, %24 : f64
        memref.store %25, %3[%16, %19] : memref<1600x2200xf64>
        %26 = addi %arg4, %c1_i32 : i32
        scf.yield %26 : i32
      }
      %18 = addi %arg3, %c1_i32 : i32
      scf.yield %18 : i32
    }
    %9:2 = scf.while (%arg2 = %8#0) : (i32) -> (i32, i32) {
      %16 = cmpi slt, %arg2, %c2200_i32 : i32
      scf.condition(%16) %c0_i32, %arg2 : i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
      %16 = index_cast %arg3 : i32 to index
      %17 = scf.for %arg4 = %c0 to %c1800 step %c1 iter_args(%arg5 = %c0_i32) -> (i32) {
        %19 = index_cast %arg5 : i32 to index
        %20 = addi %arg5, %c1_i32 : i32
        %21 = muli %arg3, %20 : i32
        %22 = remi_signed %21, %c1800_i32 : i32
        %23 = sitofp %22 : i32 to f64
        %24 = sitofp %c1800_i32 : i32 to f64
        %25 = divf %23, %24 : f64
        memref.store %25, %4[%16, %19] : memref<2200x1800xf64>
        scf.yield %20 : i32
      }
      %18 = addi %arg3, %c1_i32 : i32
      scf.yield %18 : i32
    }
    %10:2 = scf.while (%arg2 = %9#0) : (i32) -> (i32, i32) {
      %16 = cmpi slt, %arg2, %c1800_i32 : i32
      scf.condition(%16) %c0_i32, %arg2 : i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
      %16 = index_cast %arg3 : i32 to index
      %17 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
        %19 = cmpi slt, %arg4, %c2400_i32 : i32
        scf.condition(%19) %arg4 : i32
      } do {
      ^bb0(%arg4: i32):  // no predecessors
        %19 = index_cast %arg4 : i32 to index
        %20 = addi %arg4, %c3_i32 : i32
        %21 = muli %arg3, %20 : i32
        %22 = addi %21, %c1_i32 : i32
        %23 = remi_signed %22, %c2400_i32 : i32
        %24 = sitofp %23 : i32 to f64
        %25 = sitofp %c2400_i32 : i32 to f64
        %26 = divf %24, %25 : f64
        memref.store %26, %5[%16, %19] : memref<1800x2400xf64>
        %27 = addi %arg4, %c1_i32 : i32
        scf.yield %27 : i32
      }
      %18 = addi %arg3, %c1_i32 : i32
      scf.yield %18 : i32
    }
    %11 = scf.while (%arg2 = %10#0) : (i32) -> i32 {
      %16 = cmpi slt, %arg2, %c1600_i32 : i32
      scf.condition(%16) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):  // no predecessors
      %16 = index_cast %arg2 : i32 to index
      %17 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
        %19 = cmpi slt, %arg3, %c2400_i32 : i32
        scf.condition(%19) %arg3 : i32
      } do {
      ^bb0(%arg3: i32):  // no predecessors
        %19 = index_cast %arg3 : i32 to index
        %20 = addi %arg3, %c2_i32 : i32
        %21 = muli %arg2, %20 : i32
        %22 = remi_signed %21, %c2200_i32 : i32
        %23 = sitofp %22 : i32 to f64
        %24 = sitofp %c2200_i32 : i32 to f64
        %25 = divf %23, %24 : f64
        memref.store %25, %6[%16, %19] : memref<1600x2400xf64>
        %26 = addi %arg3, %c1_i32 : i32
        scf.yield %26 : i32
      }
      %18 = addi %arg2, %c1_i32 : i32
      scf.yield %18 : i32
    }
    call @polybench_timer_start() : () -> ()
    %12 = affine.load %0[0] : memref<1xf64>
    %13 = affine.load %1[0] : memref<1xf64>
    affine.for %arg2 = 0 to 50 {
      affine.for %arg3 = 0 to 132 {
        affine.if #set0(%arg3) {
          affine.for %arg4 = #map0(%arg2) to min #map1(%arg2) {
            affine.for %arg5 = #map0(%arg3) to min #map2(%arg3) {
              %16 = affine.load %6[%arg4, %arg5] : memref<1600x2400xf64>
              %17 = mulf %16, %13 : f64
              affine.store %17, %6[%arg4, %arg5] : memref<1600x2400xf64>
              affine.store %cst_1, %2[%arg4, %arg5] : memref<1600x1800xf64>
            }
            affine.for %arg5 = 2400 to min #map3(%arg3) {
              affine.store %cst_1, %2[%arg4, %arg5] : memref<1600x1800xf64>
            }
            affine.for %arg5 = 1800 to min #map4(%arg3) {
              %16 = affine.load %6[%arg4, %arg5] : memref<1600x2400xf64>
              %17 = mulf %16, %13 : f64
              affine.store %17, %6[%arg4, %arg5] : memref<1600x2400xf64>
            }
          }
        }
        affine.if #set1(%arg3) {
          affine.for %arg4 = #map0(%arg2) to min #map1(%arg2) {
            affine.for %arg5 = #map0(%arg3) to min #map3(%arg3) {
              affine.store %cst_1, %2[%arg4, %arg5] : memref<1600x1800xf64>
            }
          }
        }
        affine.if #set2(%arg3) {
          affine.for %arg4 = #map0(%arg2) to min #map1(%arg2) {
            affine.for %arg5 = #map0(%arg3) to min #map4(%arg3) {
              %16 = affine.load %6[%arg4, %arg5] : memref<1600x2400xf64>
              %17 = mulf %16, %13 : f64
              affine.store %17, %6[%arg4, %arg5] : memref<1600x2400xf64>
            }
          }
        }
      }
    }
    affine.for %arg2 = 0 to 50 {
      affine.for %arg3 = 0 to 57 {
        affine.for %arg4 = 0 to 69 {
          affine.for %arg5 = #map0(%arg2) to min #map1(%arg2) {
            affine.for %arg6 = #map0(%arg4) to min #map5(%arg4) {
              affine.for %arg7 = #map0(%arg3) to min #map3(%arg3) {
                %16 = affine.load %2[%arg5, %arg7] : memref<1600x1800xf64>
                %17 = affine.load %3[%arg5, %arg6] : memref<1600x2200xf64>
                %18 = mulf %12, %17 : f64
                %19 = affine.load %4[%arg6, %arg7] : memref<2200x1800xf64>
                %20 = mulf %18, %19 : f64
                %21 = addf %16, %20 : f64
                affine.store %21, %2[%arg5, %arg7] : memref<1600x1800xf64>
              }
            }
          }
        }
        affine.for %arg4 = 0 to 75 {
          affine.for %arg5 = #map0(%arg2) to min #map1(%arg2) {
            affine.for %arg6 = #map0(%arg3) to min #map3(%arg3) {
              affine.for %arg7 = #map0(%arg4) to min #map4(%arg4) {
                %16 = affine.load %6[%arg5, %arg7] : memref<1600x2400xf64>
                %17 = affine.load %2[%arg5, %arg6] : memref<1600x1800xf64>
                %18 = affine.load %5[%arg6, %arg7] : memref<1800x2400xf64>
                %19 = mulf %17, %18 : f64
                %20 = addf %16, %19 : f64
                affine.store %20, %6[%arg5, %arg7] : memref<1600x2400xf64>
              }
            }
          }
        }
      }
    }
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    %14 = cmpi sgt, %arg0, %c42_i32 : i32
    %15 = scf.if %14 -> (i1) {
      %16 = llvm.getelementptr %arg1[%c0_i64] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
      %17 = llvm.load %16 : !llvm.ptr<ptr<i8>>
      %18 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
      %19 = llvm.getelementptr %18[%c0_i64, %c0_i64] : (!llvm.ptr<array<1 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %20 = llvm.call @strcmp(%17, %19) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
      %21 = trunci %20 : i32 to i1
      %22 = xor %21, %true : i1
      scf.yield %22 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %15 {
      call @print_array(%c1600_i32, %c2400_i32, %7) : (i32, i32, memref<?x2400xf64>) -> ()
    }
    memref.dealloc %2 : memref<1600x1800xf64>
    memref.dealloc %3 : memref<1600x2200xf64>
    memref.dealloc %4 : memref<2200x1800xf64>
    memref.dealloc %5 : memref<1800x2400xf64>
    memref.dealloc %6 : memref<1600x2400xf64>
    return %c0_i32 : i32
  }
  func private @polybench_timer_start()
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
  func private @S0(%arg0: memref<?x1800xf64>, %arg1: index, %arg2: index) attributes {scop.stmt} {
    %cst = constant 0.000000e+00 : f64
    affine.store %cst, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x1800xf64>
    return
  }
  func private @S1(%arg0: memref<?x1800xf64>, %arg1: index, %arg2: index, %arg3: memref<?x1800xf64>, %arg4: index, %arg5: f64, %arg6: memref<?x2200xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x1800xf64>
    %1 = affine.load %arg6[symbol(%arg1), symbol(%arg4)] : memref<?x2200xf64>
    %2 = mulf %arg5, %1 : f64
    %3 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<?x1800xf64>
    %4 = mulf %2, %3 : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x1800xf64>
    return
  }
  func private @S2(%arg0: memref<?x2400xf64>, %arg1: index, %arg2: index, %arg3: f64) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2400xf64>
    %1 = mulf %0, %arg3 : f64
    affine.store %1, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2400xf64>
    return
  }
  func private @S3(%arg0: memref<?x2400xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2400xf64>, %arg4: index, %arg5: memref<?x1800xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2400xf64>
    %1 = affine.load %arg5[symbol(%arg1), symbol(%arg4)] : memref<?x1800xf64>
    %2 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<?x2400xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    affine.store %4, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2400xf64>
    return
  }
  func private @print_array(%arg0: i32, %arg1: i32, %arg2: memref<?x2400xf64>) {
    %c0_i64 = constant 0 : i64
    %c0_i32 = constant 0 : i32
    %c20_i32 = constant 20 : i32
    %c1_i32 = constant 1 : i32
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %2 = llvm.mlir.addressof @str1 : !llvm.ptr<array<23 x i8>>
    %3 = llvm.getelementptr %2[%c0_i64, %c0_i64] : (!llvm.ptr<array<23 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %4 = llvm.call @fprintf(%1, %3) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
    %5 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %6 = llvm.load %5 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %7 = llvm.mlir.addressof @str2 : !llvm.ptr<array<15 x i8>>
    %8 = llvm.getelementptr %7[%c0_i64, %c0_i64] : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %9 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
    %10 = llvm.getelementptr %9[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %11 = llvm.call @fprintf(%6, %8, %10) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    %12 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
      %25 = cmpi slt, %arg3, %arg0 : i32
      scf.condition(%25) %arg3 : i32
    } do {
    ^bb0(%arg3: i32):  // no predecessors
      %25 = index_cast %arg3 : i32 to index
      %26 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
        %28 = cmpi slt, %arg4, %arg1 : i32
        scf.condition(%28) %arg4 : i32
      } do {
      ^bb0(%arg4: i32):  // no predecessors
        %28 = index_cast %arg4 : i32 to index
        %29 = muli %arg3, %arg0 : i32
        %30 = addi %29, %arg4 : i32
        %31 = remi_signed %30, %c20_i32 : i32
        %32 = cmpi eq, %31, %c0_i32 : i32
        scf.if %32 {
          %40 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
          %41 = llvm.load %40 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
          %42 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
          %43 = llvm.getelementptr %42[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %44 = llvm.call @fprintf(%41, %43) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
        }
        %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
        %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
        %35 = llvm.mlir.addressof @str5 : !llvm.ptr<array<8 x i8>>
        %36 = llvm.getelementptr %35[%c0_i64, %c0_i64] : (!llvm.ptr<array<8 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %37 = memref.load %arg2[%25, %28] : memref<?x2400xf64>
        %38 = llvm.call @fprintf(%34, %36, %37) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, f64) -> i32
        %39 = addi %arg4, %c1_i32 : i32
        scf.yield %39 : i32
      }
      %27 = addi %arg3, %c1_i32 : i32
      scf.yield %27 : i32
    }
    %13 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %14 = llvm.load %13 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %15 = llvm.mlir.addressof @str6 : !llvm.ptr<array<17 x i8>>
    %16 = llvm.getelementptr %15[%c0_i64, %c0_i64] : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = llvm.mlir.addressof @str3 : !llvm.ptr<array<2 x i8>>
    %18 = llvm.getelementptr %17[%c0_i64, %c0_i64] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %19 = llvm.call @fprintf(%14, %16, %18) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
    %20 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %21 = llvm.load %20 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
    %22 = llvm.mlir.addressof @str7 : !llvm.ptr<array<23 x i8>>
    %23 = llvm.getelementptr %22[%c0_i64, %c0_i64] : (!llvm.ptr<array<23 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %24 = llvm.call @fprintf(%21, %23) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> i32
    return
  }
}

