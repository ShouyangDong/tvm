/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file extract_gpu_resource_usage.cc
 * \brief Analyze and extract actual GPU resource usage from a TIR GPU kernel.
 *        It collects statistics such as:
 *        - thread block dimensions (threadIdx.x/y/z)
 *        - total threads per block
 *        - shared memory and local memory consumption (in bytes)
 *        - vector access widths
 *        This information can be used for hardware-aware scheduling,
 *        cost modeling, or diagnostic reporting.
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../../runtime/thread_storage_scope.h"
#include "../transforms/ir_utils.h"

namespace tvm {
namespace tir {

class GPUResourceExtractor : public StmtExprVisitor {
 public:
  Map<String, ObjectRef> Extract(Stmt stmt) {
    Reset_();
    this->VisitStmt(stmt);
    return BuildResult_();
  }

  void VisitStmt_(const AllocateNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto scope = GetPtrStorageScope(op->buffer_var);
    runtime::StorageScope storage_scope = runtime::StorageScope::Create(scope);

    size_t size_bytes = static_cast<size_t>(op->ConstantAllocationSize()) *
                        op->dtype.bytes() * op->dtype.lanes();

    if (storage_scope.rank == runtime::StorageRank::kLocal) {
      local_memory_bytes_ += size_bytes;
    } else if (storage_scope.rank == runtime::StorageRank::kShared) {
      shared_memory_bytes_ += size_bytes;
    }

    // Record vector usage
    if (op->dtype.is_vector()) {
      vector_alloc_sizes_.push_back(size_bytes);
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
      if (nest_level_ == 0) {
        // New kernel
        if (kernels_launched_ > 0) {
          // TODO: support multi-kernel? For now assume single kernel.
        }
        kernels_launched_++;
        ResetKernelStats_();
      }

      Var var = op->node.as<IterVarNode>()->var;
      const auto* extent = op->value.as<IntImmNode>();
      ICHECK(extent) << "Thread extent must be constant for analysis";

      std::string name = var.get()->name_hint;
      int64_t length = extent->value;

      if (name == "threadIdx.x") {
        thread_x_ = length;
        visited_threads_.insert(name);
      } else if (name == "threadIdx.y") {
        thread_y_ = length;
        visited_threads_.insert(name);
      } else if (name == "threadIdx.z") {
        thread_z_ = length;
        visited_threads_.insert(name);
      }
      // ignore vthread for resource counting (it's virtual)

      nest_level_++;
      StmtVisitor::VisitStmt_(op);
      nest_level_--;

      if (nest_level_ == 0) {
        threads_per_block_ = thread_x_ * thread_y_ * thread_z_;
      }
    } else {
      StmtVisitor::VisitStmt_(op);
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    if (op->dtype.is_vector()) {
      int64_t vec_bytes = op->dtype.bytes() * op->dtype.lanes();
      vector_load_sizes_.push_back(vec_bytes);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    if (op->value->dtype.is_vector()) {
      int64_t vec_bytes = op->value->dtype.bytes() * op->value->dtype.lanes();
      vector_store_sizes_.push_back(vec_bytes);
    }
    StmtVisitor::VisitStmt_(op);
  }

 private:
  int nest_level_ = 0;
  int64_t thread_x_ = 1, thread_y_ = 1, thread_z_ = 1;
  int64_t threads_per_block_ = 1;
  int64_t shared_memory_bytes_ = 0;
  int64_t local_memory_bytes_ = 0;
  int64_t kernels_launched_ = 0;
  std::unordered_set<std::string> visited_threads_;

  std::vector<int64_t> vector_alloc_sizes_;
  std::vector<int64_t> vector_load_sizes_;
  std::vector<int64_t> vector_store_sizes_;

  void Reset_() {
    ResetKernelStats_();
    kernels_launched_ = 0;
    shared_memory_bytes_ = 0;
    local_memory_bytes_ = 0;
    vector_alloc_sizes_.clear();
    vector_load_sizes_.clear();
    vector_store_sizes_.clear();
  }

  void ResetKernelStats_() {
    thread_x_ = 1;
    thread_y_ = 1;
    thread_z_ = 1;
    threads_per_block_ = 1;
    visited_threads_.clear();
  }

  Map<String, ObjectRef> BuildResult_() {
    Map<String, ObjectRef> result;

    result.Set("thread_x", Integer(thread_x_));
    result.Set("thread_y", Integer(thread_y_));
    result.Set("thread_z", Integer(thread_z_));
    result.Set("threads_per_block", Integer(threads_per_block_));
    result.Set("shared_memory_bytes", Integer(static_cast<int64_t>(shared_memory_bytes_)));
    result.Set("local_memory_bytes", Integer(static_cast<int64_t>(local_memory_bytes_)));
    result.Set("num_kernels", Integer(kernels_launched_));

    // Optional: add vector info as arrays
    Array<Integer> load_vecs;
    for (auto sz : vector_load_sizes_) load_vecs.push_back(Integer(sz));
    result.Set("vector_load_bytes", load_vecs);

    Array<Integer> store_vecs;
    for (auto sz : vector_store_sizes_) store_vecs.push_back(Integer(sz));
    result.Set("vector_store_bytes", store_vecs);

    return result;
  }
};

Map<String, ObjectRef> ExtractGPUResourceUsage(const PrimFunc& func) {
  GPUResourceExtractor extractor;
  return extractor.Extract(func->body);
}

TVM_REGISTER_GLOBAL("tir.analysis.ExtractGPUResourceUsage").set_body_typed(ExtractGPUResourceUsage);

}  // namespace tir
}  // namespace tvm
