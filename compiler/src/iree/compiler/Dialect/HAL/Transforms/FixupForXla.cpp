// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iree/compiler/Dialect/HAL/IR/HALTypes.h>
#include <iree/compiler/Dialect/Util/IR/UtilOps.h>
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {
void moveAllocations(func::FuncOp func, CFGLoopInfo &loopInfo) {
  for (auto alloc : llvm::make_early_inc_range(
           func.getOps<IREE::HAL::AllocatorAllocateOp>())) {

    auto *loop = loopInfo.getLoopFor(alloc->getBlock());
    if (!loop)
      continue;

    auto *predecessor = loop->getOutermostLoop()->getLoopPredecessor();
    if (!predecessor)
      continue;

    alloc->moveBefore(predecessor->getTerminator());
  }
}

void moveCommandBuffers(func::FuncOp func, CFGLoopInfo &loopInfo) {
  llvm::SmallVector<std::pair<Operation *, Block *>> moves;

  for (Operation &op : llvm::make_early_inc_range(func.getOps())) {
    if (isa<IREE::HAL::CommandBufferCreateOp>(&op) ||
        isa<IREE::HAL::CommandBufferPushDescriptorSetOp>(&op) ||
        isa<IREE::HAL::CommandBufferCopyBufferOp>(&op) ||
        isa<IREE::HAL::CommandBufferDispatchOp>(&op) ||
        isa<IREE::HAL::CommandBufferExecutionBarrierOp>(&op) ||
        isa<IREE::HAL::CommandBufferFinalizeOp>(&op)) {
      auto *loop = loopInfo.getLoopFor(op.getBlock());
      if (!loop)
        continue;

      auto *predecessor = loop->getOutermostLoop()->getLoopPredecessor();
      if (!predecessor)
        continue;

      moves.emplace_back(&op, predecessor);
    }
  }

  for (auto [op, block] : moves)
    op->moveBefore(block->getTerminator());
}

void insertBarriers(func::FuncOp func) {
  struct Resource {
    Value value;
    bool write;
    int64_t begin;
    int64_t end;
  };

  // Pipeline global name -> bindings.
  llvm::StringMap<ArrayAttr> bindings;

  auto module = func->getParentOfType<ModuleOp>();
  module->walk([&](IREE::Util::GlobalStoreOp store) {
    if (!store.getValue().getType().isa<IREE::HAL::PipelineLayoutType>())
      return;
    auto create = cast<IREE::HAL::PipelineLayoutCreateOp>(
        store.getValue().getDefiningOp());
    auto layout = create.getSetLayouts().front();
    auto desc =
        cast<IREE::HAL::DescriptorSetLayoutCreateOp>(layout.getDefiningOp());
    bindings[store.getGlobal()] = desc.getBindings();
  });

  llvm::DenseMap<Value, llvm::SmallVector<Resource>> resources;

  auto is_overlap = [](Resource a, Resource b) {
    if (a.value != b.value)
      return false;
    if (!a.write && !b.write)
      return false;
    return std::max(a.begin, b.begin) <= std::min(a.end, b.end);
  };

  for (Operation &op : func.getOps()) {
    if (auto barrier =
            dyn_cast<IREE::HAL::CommandBufferExecutionBarrierOp>(&op)) {
      resources[barrier.getCommandBuffer()].clear();
    }

    if (auto push =
            dyn_cast<IREE::HAL::CommandBufferPushDescriptorSetOp>(&op)) {

      auto load = cast<IREE::Util::GlobalLoadOp>(
          push.getPipelineLayout().getDefiningOp());
      auto binding = bindings[load.getGlobal()];

      auto is_write = [&](int index) {
        auto attr = cast<IREE::HAL::DescriptorSetBindingAttr>(binding[index]);
        return !attr.getFlags().has_value();
      };

      int index = 0;
      for (auto [buffer, offset, length] :
           llvm::zip(push.getBindingBuffers(), push.getBindingOffsets(),
                     push.getBindingLengths())) {
        llvm::APInt offset_value;
        llvm::APInt length_value;
        if (matchPattern(offset, m_ConstantInt(&offset_value)) ||
            matchPattern(length, m_ConstantInt(&length_value))) {
          Resource resource{
              buffer, is_write(index), offset_value.getSExtValue(),
              offset_value.getSExtValue() + length_value.getSExtValue()};

          bool overlap =
              llvm::any_of(resources[push.getCommandBuffer()],
                           [&](Resource a) { return is_overlap(a, resource); });

          ++index;

          if (overlap) {
            llvm::errs() << "Add extra barrier\n";
            OpBuilder b(push);

            b.create<IREE::HAL::CommandBufferExecutionBarrierOp>(
                push->getLoc(), push.getCommandBuffer(),
                IREE::HAL::ExecutionStageBitfield::CommandRetire |
                    IREE::HAL::ExecutionStageBitfield::Dispatch |
                    IREE::HAL::ExecutionStageBitfield::Transfer,
                IREE::HAL::ExecutionStageBitfield::CommandIssue |
                    IREE::HAL::ExecutionStageBitfield::Dispatch |
                    IREE::HAL::ExecutionStageBitfield::Transfer,
                IREE::HAL::ExecutionBarrierFlagBitfield::None);

            resources[push.getCommandBuffer()].clear();
            break;

          } else {

            resources[push.getCommandBuffer()].push_back(resource);
          }
        }
      }
    }
  }
}

struct FixupForXlaPass
    : public PassWrapper<FixupForXlaPass, OperationPass<mlir::ModuleOp>> {
  StringRef getArgument() const override { return "iree-hal-fixup-for-xla"; }

  StringRef getDescription() const override { return ""; }

  void runOnOperation() override {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      // Skip callable regions with empty or trivial CFG.
      Region *region = func.getCallableRegion();
      if (!region || region->hasOneBlock())
        continue;

      auto &domInfo = getAnalysis<DominanceInfo>();
      CFGLoopInfo loopInfo(domInfo.getDomTree(region));

      moveAllocations(func, loopInfo);
      moveCommandBuffers(func, loopInfo);
      insertBarriers(func);
    }
  };
};

struct PreFixupForXlaPass
    : public PassWrapper<FixupForXlaPass, OperationPass<mlir::ModuleOp>> {
  StringRef getArgument() const override {
    return "iree-hal-pre-fixup-for-xla";
  }

  StringRef getDescription() const override { return ""; }

  void runOnOperation() override {
    llvm::SmallVector<Util::OptimizationBarrierOp> barriers;

    getOperation()->walk([&](Util::OptimizationBarrierOp barrier) {
      if (llvm::all_of(barrier->getOperandTypes(),
                       [](Type type) { return type.isIndex(); }))
        barriers.push_back(barrier);
    });

    for (auto barrier : barriers) {
      for (auto [from, to] :
           llvm::zip(barrier->getResults(), barrier.getOperands())) {
        from.replaceAllUsesWith(to);
      }
      barrier->erase();
    }
  };
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createFixupForXlaPass() {
  return std::make_unique<FixupForXlaPass>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> createPreFixupForXlaPass() {
  return std::make_unique<PreFixupForXlaPass>();
}

static PassRegistration<FixupForXlaPass> pass;
static PassRegistration<PreFixupForXlaPass> prePass;

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
