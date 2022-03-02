// RUN: polygeist-opt --cpuify="method=distribute" --split-input-file %s | FileCheck %s

module {
  func private @capture(%a : memref<i32>) 
  func private @use(%a : memref<?xi32>, %b : f32, %d : i32, %e : f32)
  func @main() {
    %c0 = arith.constant 0 : index
    %cc1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    scf.parallel (%arg2) = (%c0) to (%c5) step (%cc1) {
      %a1 = memref.alloca() : memref<2xi32>
      %a2 = memref.cast %a1 : memref<2xi32> to memref<?xi32>
      %b1 = memref.alloca() : memref<f32>
      %c1 = memref.alloca() : memref<i32>
      %d1 = memref.alloca() : memref<1xi32>
      %b2 = memref.load %b1[] : memref<f32>
      call @capture(%c1) : (memref<i32>) -> ()
      %d2 = memref.cast %d1 : memref<1xi32> to memref<?xi32>
      
      %e1 = memref.alloca() : memref<1xf32>
      %e2 = memref.cast %e1 : memref<1xf32> to memref<?xf32>
      %e3 = memref.load %e2[%c0] : memref<?xf32>

      "polygeist.barrier"(%arg2) : (index) -> ()
      
      %d3 = memref.load %d2[%c0] : memref<?xi32>
      call @use(%a2, %b2, %d3, %e3) : (memref<?xi32>, f32, i32, f32) -> ()
      scf.yield
    }
    return
  }
}

// CHECK:  func @main() {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c5 = arith.constant 5 : index
// CHECK-NEXT:    %0 = memref.alloc(%c5) : memref<?xf32>
// CHECK-NEXT:    %1 = memref.alloc(%c5) : memref<?xf32>
// CHECK-NEXT:    %2 = memref.alloc(%c5) : memref<?x2xi32>
// CHECK-NEXT:    %3 = memref.alloc(%c5) : memref<?xi32>
// CHECK-NEXT:    %4 = memref.alloc(%c5) : memref<?x1xi32>
// CHECK-NEXT:    scf.parallel (%arg0) = (%c0) to (%c5) step (%c1) {
// CHECK-NEXT:      %5 = memref.alloca() : memref<f32>
// CHECK-NEXT:      %6 = memref.load %5[] : memref<f32>
// CHECK-NEXT:      memref.store %6, %1[%arg0] : memref<?xf32>
// CHECK-NEXT:      %7 = "polygeist.subindex"(%3, %arg0) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:      call @capture(%7) : (memref<i32>) -> ()
// CHECK-NEXT:      %8 = memref.alloca() : memref<1xf32>
// CHECK-NEXT:      %9 = memref.load %8[%c0] : memref<1xf32>
// CHECK-NEXT:      memref.store %9, %0[%arg0] : memref<?xf32>
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.parallel (%arg0) = (%c0) to (%c5) step (%c1) {
// CHECK-NEXT:      %5 = memref.load %1[%arg0] : memref<?xf32>
// CHECK-NEXT:      %6 = memref.load %0[%arg0] : memref<?xf32>
// CHECK-NEXT:      %7 = "polygeist.subindex"(%4, %arg0) : (memref<?x1xi32>, index) -> memref<1xi32>
// CHECK-NEXT:      %8 = "polygeist.subindex"(%2, %arg0) : (memref<?x2xi32>, index) -> memref<2xi32>
// CHECK-NEXT:      %9 = memref.cast %8 : memref<2xi32> to memref<?xi32>
// CHECK-NEXT:      %10 = memref.load %7[%c0] : memref<1xi32>
// CHECK-NEXT:      call @use(%9, %5, %10, %6) : (memref<?xi32>, f32, i32, f32) -> ()
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %0 : memref<?xf32>
// CHECK-NEXT:    memref.dealloc %1 : memref<?xf32>
// CHECK-NEXT:    memref.dealloc %2 : memref<?x2xi32>
// CHECK-NEXT:    memref.dealloc %3 : memref<?xi32>
// CHECK-NEXT:    memref.dealloc %4 : memref<?x1xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

