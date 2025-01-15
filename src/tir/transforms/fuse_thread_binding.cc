#include <tvm/ir/module.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/runtime/registry.h>
#include <unordered_map>
#include <string>

namespace tvm {
namespace tir {

// Mutator to precompute maximum extent for each thread binding type.
class ThreadBindingExtentCollector : public StmtVisitor {
 public:
  void VisitStmt_(const SeqStmtNode* op) override {
    std::unordered_set<std::string> thread_var_names;

    // 遍历 `SeqStmt` 里的 `For` 循环，检查是否有重复 thread binding
    for (const Stmt& stmt : op->seq) {
      if (const ForNode* for_loop = stmt.as<ForNode>()) {
        if (for_loop->thread_binding.defined()) {
          std::string thread_var = for_loop->thread_binding.value()->var->name_hint;

          // 如果发现重复的 thread var，标记 has_duplicate
          if (thread_var_names.count(thread_var)) {
            has_duplicate = true;
          }
          thread_var_names.insert(thread_var);
        }
      }
    }

    // 继续递归访问子节点
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode* loop) final {
    if (loop->thread_binding.defined()) {
      std::string thread_tag = loop->thread_binding.value()->thread_tag;
      IterVar thread_binding = loop->thread_binding.value();
      // 安全获取 `extent`，确保它是整数
      if (const IntImmNode* extent_imm = loop->extent.as<IntImmNode>()) {
        int extent = extent_imm->value;

        // 更新最大 extent 记录
        if (max_extents.find(thread_tag) == max_extents.end()) {
          max_extents[thread_tag] = extent;
          thread_bindings[thread_tag] = thread_binding;
          loop_vars.push_back(loop->loop_var);
        } else {
          max_extents[thread_tag] = std::max(max_extents[thread_tag], extent);
        }
      }
    }
    
    // 递归访问子节点
    StmtVisitor::VisitStmt_(loop);
  }

  // 记录最大 thread extent
  std::unordered_map<std::string, int> max_extents;
  std::unordered_map<std::string, IterVar> thread_bindings;
  Array<Var> loop_vars;
  bool has_duplicate = false;
};



class MarkParellelThread : public StmtMutator {
 public:
  bool has_duplicate = false;

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    std::unordered_set<std::string> thread_var_names;

    // 遍历 `SeqStmt` 里的 `For` 循环，检查是否有重复 thread binding
    for (const Stmt& stmt : op->seq) {
      if (const ForNode* for_loop = stmt.as<ForNode>()) {
        if (for_loop->thread_binding.defined()) {
          std::string thread_var = for_loop->thread_binding.value()->var->name_hint;

          // 如果发现重复的 thread var，标记 has_duplicate
          if (thread_var_names.count(thread_var)) {
            has_duplicate = true;
          }
          thread_var_names.insert(thread_var);
        }
      }
    }
    if (!has_duplicate) {
        return StmtMutator::VisitStmt_(op);
    }
    Array<Stmt> filtered;
    for (Stmt stmt : op->seq) {
      if (!is_no_op(stmt)) {
        filtered.push_back(std::move(stmt));
      }
    }
    tir::Stmt seq_stmt = SeqStmt::Flatten(filtered);
    auto zero = make_zero(DataType::Int(32));
    return tvm::tir::AttrStmt(zero, "parallel_thread", zero, seq_stmt);
  }
};

// Mutator to optimize thread binding: for each For loop with thread binding,
// if it is not the first occurrence (or its extent is less than the maximum extent),
// wrap its body in an if statement so that only when the loop variable equals (max-1)
// the loop body is executed.
class FuseThreadBindingsMutator : public StmtMutator {
 public:
  std::unordered_map<std::string, int> max_extents;
  std::unordered_map<std::string, int> occurrence_count;
  std::unordered_map<std::string, IterVar> thread_bindings;

  explicit FuseThreadBindingsMutator(
      const std::unordered_map<std::string, int>& max_extents,
      const std::unordered_map<std::string, IterVar>& thread_bindings)
      : max_extents(max_extents), thread_bindings(thread_bindings) {}

  Stmt VisitStmt_(const ForNode* loop_node) final {
    // If this For is not thread binding attribute, return as usual.
    if (loop_node->kind != ForKind::kThreadBinding) {
      return StmtMutator::VisitStmt_(loop_node);
    }
    std::string tag = loop_node->thread_binding.value()->thread_tag;
    IterVar iter_var = loop_node->thread_binding.value();
    Var loop_var = loop_node->loop_var;
    occurrence_count[tag] += 1;
    int occ = occurrence_count[tag];
    if (occ == 1) {
        return StmtMutator::VisitStmt_(loop_node);
    } else{
        int max_extent = max_extents[tag];
        if (Downcast<Integer>(loop_node->extent)->value == max_extent) {
            Stmt body = loop_node->body;
            Map<Var, Var> vmap;

            vmap.Set(loop_var, thread_bindings[tag]->var);
            body = Substitute(body, vmap);
            // 直接访问其 body
            return VisitStmt(body);
        } else {
            // 如果 extent 小于最大值，则包裹在 if 条件下
            PrimExpr cond = LT(loop_node->loop_var, loop_node->extent);
            Stmt if_stmt = IfThenElse(cond, loop_node->body);
            Map<Var, Var> vmap;
            vmap.Set(loop_var, thread_bindings[tag]->var);
            if_stmt = Substitute(if_stmt, vmap);
        return VisitStmt(if_stmt);
        }
    }
  }
};

class RewriteThreadBindingsMutator : public StmtMutator {
 public:
  std::unordered_map<std::string, int> max_extents;
  std::unordered_map<std::string, IterVar> thread_bindings;
  Array<Var> loop_vars;
  bool has_duplicate = false;

  explicit RewriteThreadBindingsMutator(
      const std::unordered_map<std::string, int>& max_extents,
      const std::unordered_map<std::string, IterVar>& thread_bindings,
      const Array<Var>& loop_vars)
      : max_extents(max_extents), thread_bindings(thread_bindings), loop_vars(loop_vars) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == "parallel_thread") {
        Stmt body = op->body;
        for (const auto& kv : thread_bindings) {
        const std::string& thread_name = kv.first;
        const IterVar& thread_binding = kv.second;
        body = For(thread_binding->var, tvm::tir::make_const(thread_binding->var->dtype,0), 
                        tvm::tir::make_const(thread_binding->var->dtype, max_extents.at(thread_name)), 
                        ForKind::kThreadBinding, body, thread_binding);
        }

        return body;
    }
    return StmtMutator::VisitStmt_(op);
  }
};


namespace transform {

// Pass entry: first collect max extents, then apply optimization.
Pass FuseThreadBindings() {
    auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
        ThreadBindingExtentCollector collector;
        collector(f->body);
        if (!collector.has_duplicate) {
            return f;
        }
        auto* n = f.CopyOnWrite();
        n->body = MarkParellelThread()(std::move(n->body));
        n->body = RewriteThreadBindingsMutator(collector.max_extents, collector.thread_bindings, collector.loop_vars)(std::move(n->body));
        n->body = FuseThreadBindingsMutator(collector.max_extents, collector.thread_bindings)(std::move(n->body));
        return f;
    };
    return CreatePrimFuncPass(pass_func, 0, "tir.FuseThreadBindings", {});
}

TVM_REGISTER_GLOBAL("tir.transform.FuseThreadBindings")
    .set_body_typed(FuseThreadBindings);
}  // namespace transform
}  // namespace tir
}  // namespace tvm
