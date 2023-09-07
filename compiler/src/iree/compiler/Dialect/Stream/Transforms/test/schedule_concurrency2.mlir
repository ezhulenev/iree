// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-stream-schedule-concurrency))" %s | FileCheck %s

// Tests that tied operands properly trigger hazard detection.
// Here @dispatch_1 has a read/write hazard on %capture0 with @dispatch_0 and
// should not be placed into the same concurrency group.

// CHECK-LABEL: @keepTiedOpsSeparate
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<external>)
func.func @keepTiedOpsSeparate(%arg0: !stream.resource<external>) -> (!stream.resource<external>, !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // CHECK: stream.async.execute
  // CHECK-SAME: with(%[[ARG0]] as %[[CAPTURE0:.+]]: !stream.resource<external>{%c4}) ->
  %results:2, %result_timepoint = stream.async.execute with(%arg0 as %capture0: !stream.resource<external>{%c4}) -> (!stream.resource<external>{%c4}, %arg0{%c2}) {
    %subview = stream.resource.subview %capture0[%c2] : !stream.resource<external>{%c4} -> !stream.resource<external>{%c2}
    // CHECK-NOT: stream.async.concurrent
    // CHECK-NEXT: stream.async.dispatch @ex::@dispatch_0
    %1 = stream.async.dispatch @ex::@dispatch_0(%capture0[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}) -> !stream.resource<external>{%c4}
    // CHECK-NEXT: stream.async.dispatch @ex::@dispatch_1
    %2 = stream.async.dispatch @ex::@dispatch_1(%subview[%c0 to %c2 for %c2]) : (!stream.resource<external>{%c2}) -> %subview{%c2}
    // CHECK-NEXT: stream.yield
    stream.yield %1, %2 : !stream.resource<external>{%c4}, !stream.resource<external>{%c2}
  } => !stream.timepoint
  return %results#0, %results#1 : !stream.resource<external>, !stream.resource<external>
}