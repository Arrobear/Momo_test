"""
蜕变测试模块 (Metamorphic Testing)

不依赖"正确答案"，通过检查 API 行为是否满足数学性质来发现 bug。
- Tier 1 (自动): NonMutation, Repeatable — 零 LLM，基于参数类型和 API 名判断
- Tier 2 (启发式): Idempotent — 基于 API 名模式判断
- Tier 2 (缓存LLM): Decompose — 每个 API 一次 LLM 调用，结果缓存

完全库无关 — 不硬编码任何 torch/glom/tf 逻辑。
"""

import collections.abc
import copy
import json
import os
import re
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ── 复用现有模块 ─────────────────────────────────────────
from config import root_path, lib_name, API_KEY, BASE_URL
from stage_1_function import read_json_api, read_file, get_doc
from stage_1_approch import _load_run_api, _eval_param_by_type


# ============================================================
# 1. 基础类型
# ============================================================

@dataclass
class Verdict:
    relation: str
    result: str       # "PASS" | "VIOLATION" | "SKIP" | "ERROR"
    reason: str = ""
    details: dict = field(default_factory=dict)

    def is_violation(self) -> bool:
        return self.result == "VIOLATION"


# ============================================================
# 2. 蜕变关系基类
# ============================================================

class MetamorphicRelation(ABC):
    """蜕变关系基类 — 库无关"""

    name: str = ""
    description: str = ""
    tier: int = 1

    @abstractmethod
    def applies_to(
        self,
        api_name: str,
        conditions: dict | None,
        api_boundary: list | None,
    ) -> tuple[bool, str]:
        """
        判断此关系是否适用于该 API。
        返回 (适用?, 原因)。
        """
        ...

    @abstractmethod
    def check(
        self,
        api_name: str,
        run_api: Callable,
        param_names: list[str],
        param_values: list[Any],
        param_types: list[str],
    ) -> Verdict:
        """
        对一组具体的参数值执行蜕变检查。
        param_names:  参数名列表
        param_values: 已 eval 的参数值列表
        param_types:  参数类型标签列表 ("code" | "literal")
        返回 Verdict。
        """
        ...


# ============================================================
# 3. Tier 1 — 非突变性 (NonMutation)
# ============================================================

class NonMutation(MetamorphicRelation):
    """
    检查 API 是否静默修改了传入的可变对象参数。

    适用条件: 参数中有 dict/list/set/bytearray 或自定义对象
    例外条件: API 名含 update/set/modify/mutate/append/extend/pop/delete/...
    """

    name = "non_mutation"
    description = "API 不应静默修改传入的可变对象参数"
    tier = 1

    _MUTATOR_PATTERNS = re.compile(
        r"\b(update|set|modify|mutate|append|extend|pop|remove|"
        r"delete|clear|insert|sort|reverse|write|save|dump|store|"
        r"register|assign|replace)\b",
        re.IGNORECASE,
    )

    _MUTABLE_INDICATORS = (
        "dict", "list", "set", "bytearray", "object",
        "instance", "module", "tensor", "ndarray", "array",
        "mutable", "sequence", "mapping",
    )

    def applies_to(self, api_name, conditions, api_boundary):
        if self._MUTATOR_PATTERNS.search(api_name):
            return False, f"API 名含突变语义: {api_name}"

        if not conditions or "Parameter type" not in conditions:
            return False, "无条件数据"

        param_types = conditions["Parameter type"]
        has_mutable = False
        for pname, pdesc in param_types.items():
            desc_lower = str(pdesc).lower()
            if any(ind in desc_lower for ind in self._MUTABLE_INDICATORS):
                has_mutable = True
                break

        if not has_mutable:
            return False, "无可变参数类型"

        return True, "存在可变类型参数"

    def check(self, api_name, run_api, param_names, param_values, param_types):
        # 构建参数字典 → 深拷贝 → 执行 → 对比
        params_before = dict(zip(param_names, param_values))
        try:
            params_copy = copy.deepcopy(params_before)
        except Exception:
            return Verdict(
                self.name, "SKIP",
                reason="参数无法 deepcopy，跳过非突变检查",
            )

        try:
            run_api(**params_before)
            status = "success"
        except Exception:
            status = "error"

        # 对比执行前后的参数
        mutations = []
        for key in param_names:
            before = params_copy.get(key)
            after = params_before.get(key)
            if not self._deep_equal(before, after):
                mutations.append({
                    "param": key,
                    "before": self._safe_repr(before),
                    "after": self._safe_repr(after),
                })

        if mutations:
            if status == "error":
                return Verdict(
                    self.name, "SKIP",
                    reason="API 调用报错，参数修改可能由异常导致，不做判断",
                    details={
                        "api_status": status,
                        "mutations": mutations,
                    },
                )
            # 仅 self 被修改：builder/fluent 模式的设计行为，不是 bug
            if all(m["param"] == "self" for m in mutations):
                return Verdict(
                    self.name, "SKIP",
                    reason="仅 self 参数被修改，builder/fluent 模式的设计行为",
                    details={
                        "api_status": status,
                        "mutations": mutations,
                    },
                )
            return Verdict(
                self.name, "VIOLATION",
                reason=f"API 静默修改了 {len(mutations)} 个参数: "
                       f"{[m['param'] for m in mutations]}",
                details={
                    "api_status": status,
                    "mutations": mutations,
                },
            )

        return Verdict(self.name, "PASS")

    @staticmethod
    def _deep_equal(a, b) -> bool:
        """递归比较两个对象是否相等（处理 Tensor/ndarray 等）"""
        if a is b:
            return True
        if type(a) is not type(b):
            return False
        # 尝试直接比较（必须严格返回 True，防止 __eq__ 返回 Spec 等非 bool 对象）
        try:
            eq = a == b
            if isinstance(eq, bool) and eq:
                return True
        except Exception:
            pass
        # Tensor 特殊处理
        if hasattr(a, "shape") and hasattr(b, "shape"):
            try:
                if a.shape != b.shape:
                    return False
                # 尝试 allclose
                if hasattr(a, "allclose"):
                    return bool(a.allclose(b))
            except Exception:
                pass
        # numpy 特殊处理
        if "numpy" in str(type(a)) and "ndarray" in str(type(a)):
            try:
                import numpy as np
                return bool(np.array_equal(a, b))
            except Exception:
                pass
        # dict 递归比较
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            return all(NonMutation._deep_equal(a[k], b[k]) for k in a)
        # list/tuple 递归比较
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(NonMutation._deep_equal(x, y) for x, y in zip(a, b))
        # 裸 object() 哨兵：无属性、无状态，同类型即等价
        if type(a) is object and type(b) is object:
            return True
        # 最后兜底：repr 比较
        try:
            if repr(a) == repr(b):
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _safe_repr(obj, max_len=200):
        try:
            s = repr(obj)
            return s[:max_len] + ("..." if len(s) > max_len else "")
        except Exception:
            return f"<{type(obj).__name__}>"


# ============================================================
# 4. Tier 1 — 可重复性 (Repeatable)
# ============================================================

class Repeatable(MetamorphicRelation):
    """
    检查相同输入调用两次是否返回相同结果。

    适用条件: 所有 API（除非有随机性）
    例外条件: API 名/文档含 random/shuffle/sample/seed/generator/stochastic
    """

    name = "repeatable"
    description = "相同输入连续调用两次应返回相同结果"
    tier = 1

    _RANDOM_PATTERNS = re.compile(
        r"\b(random|shuffle|sample|permute|seed|generator|"
        r"stochastic|noise|dropout|bernoulli|poisson)\b",
        re.IGNORECASE,
    )

    def applies_to(self, api_name, conditions, api_boundary):
        if self._RANDOM_PATTERNS.search(api_name):
            return False, f"API 名含随机性: {api_name}"
        # 也检查文档
        if conditions:
            for key in conditions:
                val_str = str(conditions[key]).lower()
                if self._RANDOM_PATTERNS.search(val_str):
                    return False, "文档含随机性描述"
        return True, "API 无随机性"

    def check(self, api_name, run_api, param_names, param_values, param_types):
        params = dict(zip(param_names, param_values))

        # 第一次调用
        try:
            output1 = run_api(**params)
            status1 = "success"
        except Exception as e:
            status1 = "error"
            output1 = f"{type(e).__name__}: {str(e)}"

        # 第二次调用
        try:
            output2 = run_api(**params)
            status2 = "success"
        except Exception as e:
            status2 = "error"
            output2 = f"{type(e).__name__}: {str(e)}"

        # 状态不一致
        if status1 != status2:
            return Verdict(
                self.name, "VIOLATION",
                reason=f"调用状态不稳定: 第1次={status1}, 第2次={status2}",
                details={"output1": str(output1)[:500], "output2": str(output2)[:500]},
            )

        # 都崩溃 → 跳过（无法判断是否可重复）
        if status1 == "error":
            return Verdict(self.name, "SKIP", reason="两次调用均崩溃，无法判断")

        # 都成功 → 比较结果
        if not self._outputs_equal(output1, output2):
            return Verdict(
                self.name, "VIOLATION",
                reason="相同输入两次输出不同",
                details={
                    "output1": self._safe_repr(output1),
                    "output2": self._safe_repr(output2),
                },
            )

        return Verdict(self.name, "PASS")

    @staticmethod
    def _outputs_equal(a, b) -> bool:
        """比较两个输出是否'足够相等'"""
        # 快速路径
        if a is b:
            return True
        if type(a) is not type(b):
            # 允许 int/float 交叉
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return abs(a - b) < 1e-10
            return False
        try:
            eq = a == b
            if isinstance(eq, bool) and eq:
                return True
        except Exception:
            pass
        # dict 递归
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            return all(
                Repeatable._outputs_equal(a[k], b[k]) for k in a
            )
        # list/tuple
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(
                Repeatable._outputs_equal(x, y) for x, y in zip(a, b)
            )
        # float 容差
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b) < 1e-10
        # generator / iterator：每次调用创建新对象，无法用值比较，同类型即视为等价
        if isinstance(a, collections.abc.Iterator) and isinstance(b, collections.abc.Iterator):
            return True
        # 最后兜底：repr 比较
        try:
            if repr(a) == repr(b):
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _safe_repr(obj, max_len=300):
        try:
            s = repr(obj)
            return s[:max_len]
        except Exception:
            return f"<{type(obj).__name__}>"


# ============================================================
# 5. Tier 2 — 幂等性 (Idempotent)
# ============================================================

class Idempotent(MetamorphicRelation):
    """
    检查"读"类操作的幂等性: f(x) = y → f(y) 应 ≈ y  (对兼容参数)

    适用: 名字模式匹配 get/read/query/find/is/has/check 等的 API
    """

    name = "idempotent"
    description = "读操作重复执行结果不变 (f(f(x)) ≈ f(x))"
    tier = 2

    _READER_PATTERNS = re.compile(
        r"(\bget\b|^get_|\bfind\b|^find_|\bquery\b|^query_|"
        r"\bread\b|^read_|\bload\b|^load_|"
        r"\bis_\b|^is_|\bhas_\b|^has_|\bcheck\b|^check_|"
        r"\btest\b|^test_|\bcontains\b|^contains_|"
        r"\bequals\b|^equals_|\bview\b|^view_|"
        r"\bsearch\b|^search_|\blookup\b|^lookup_)",
        re.IGNORECASE,
    )

    def applies_to(self, api_name, conditions, api_boundary):
        if self._READER_PATTERNS.search(api_name):
            return True, f"API 名匹配读取模式"
        return False, "API 名不匹配读取模式"

    def check(self, api_name, run_api, param_names, param_values, param_types):
        params = dict(zip(param_names, param_values))

        # 第一次调用
        try:
            output1 = run_api(**params)
            status1 = "success"
        except Exception as e:
            return Verdict(self.name, "SKIP",
                           reason=f"第一次调用崩溃: {type(e).__name__}")

        # 尝试将 output1 作为某个参数的输入
        # 策略: 找第一个取值与 output1 类型兼容的参数，替换之
        second_params = dict(params)
        fed_back = False
        for pname, pval in params.items():
            # 简单类型检查
            if self._type_compatible(output1, pval):
                second_params[pname] = output1
                fed_back = True
                break

        if not fed_back:
            # 无法将输出反哺为输入
            return Verdict(
                self.name, "SKIP",
                reason="输出类型无法匹配任何输入参数",
                details={"output_type": str(type(output1).__name__)},
            )

        # 第二次调用
        try:
            output2 = run_api(**second_params)
            status2 = "success"
        except Exception as e:
            return Verdict(
                self.name, "SKIP",
                reason=f"第二次调用(输出反哺)崩溃: {type(e).__name__}",
            )

        # 比较 output1 和 output2
        if not Repeatable._outputs_equal(output1, output2):
            return Verdict(
                self.name, "VIOLATION",
                reason="幂等性违反: f(x) ≠ f(f(x))",
                details={
                    "first_output": Repeatable._safe_repr(output1),
                    "second_output": Repeatable._safe_repr(output2),
                    "fed_back_param": f"{pname}={Repeatable._safe_repr(output1)}",
                },
            )

        return Verdict(self.name, "PASS")

    @staticmethod
    def _type_compatible(value, reference) -> bool:
        """检查 value 是否与 reference 类型兼容（可否替换）"""
        if value is None:
            return False
        if isinstance(reference, type(value)):
            return True
        if isinstance(reference, (list, tuple)) and isinstance(value, (list, tuple)):
            return True
        if isinstance(reference, dict) and isinstance(value, dict):
            return True
        if isinstance(reference, str) and isinstance(value, str):
            return True
        return False


# ============================================================
# 6. Tier 2 — 可分解性 (Decompose)
# ============================================================

class Decompose(MetamorphicRelation):
    """
    检查复合操作能否正确分解为原子步骤。

    分解方案由 LLM 一次性生成并缓存。无缓存时跳过。

    常见模式 (从参数结构推导):
    - dict 参数: api(x, {k1:v1, k2:v2}) == {k1: api(x, v1), k2: api(x, v2)}
    - 路径参数: api(x, 'a.b') == api(api(x, 'a'), 'b')
    - 管道参数: api(x, (s1, s2)) == s2(s1(x))
    """

    name = "decompose"
    description = "复合操作应能正确分解为原子步骤"
    tier = 2

    _COMPOSITE_INDICATORS = (
        "dict", "spec", "path", "query", "pattern",
        "tuple of", "list of", "sequence",
    )

    def __init__(self, decompose_plans: dict | None = None):
        self._plans = decompose_plans or {}

    def applies_to(self, api_name, conditions, api_boundary):
        # 先检查是否有缓存的 LLM 分解方案
        if api_name in self._plans:
            plan = self._plans[api_name]
            if plan.get("applies"):
                return True, f"已有分解方案: {len(plan.get('patterns', []))} 种"
            return False, f"LLM 判定不适用: {plan.get('reason', '')}"

        # 启发式: 检查参数是否含复合结构
        if not conditions or "Parameter type" not in conditions:
            return False, "无条件数据"

        param_types = conditions["Parameter type"]
        for pname, pdesc in param_types.items():
            desc_lower = str(pdesc).lower()
            if any(ind in desc_lower for ind in self._COMPOSITE_INDICATORS):
                return True, f"参数 '{pname}' 含复合结构 ({pdesc})"
            # 也检查 boundary 中的类型
            if api_boundary:
                for b in api_boundary:
                    params = b.get("api_input", {}).get("params", {})
                    if pname in params:
                        ptype = params[pname].get("type", "")
                        if ptype in ("dict", "object", "complex"):
                            return True, f"参数 '{pname}' boundary 类型={ptype}"

        return False, "无复合结构参数"

    def check(self, api_name, run_api, param_names, param_values, param_types):
        params = dict(zip(param_names, param_values))

        # 如果有 LLM 缓存的分解方案，优先使用
        if api_name in self._plans and self._plans[api_name].get("applies"):
            return self._check_with_llm_plan(
                api_name, run_api, params, self._plans[api_name]
            )

        # 否则用启发式分解
        return self._check_heuristic(api_name, run_api, params)

    def _check_with_llm_plan(
        self, api_name, run_api, params, plan
    ) -> Verdict:
        """使用 LLM 缓存的分解方案执行检查"""
        violations = []
        for pattern in plan.get("patterns", []):
            condition = pattern.get("condition", "")
            composite_code = pattern.get("composite_code", "")
            decomposed_code = pattern.get("decomposed_code", "")
            pattern_name = pattern.get("name", "unnamed")

            if not composite_code or not decomposed_code:
                continue

            # 在 params 上下文中执行
            local_env = dict(params)
            local_env["run_api"] = run_api

            # 条件检查
            if condition:
                try:
                    if not eval(condition, {}, local_env):
                        continue
                except Exception:
                    continue

            # 执行复合调用
            try:
                composite_result = eval(composite_code, {}, local_env)
            except Exception as e:
                continue  # 复合调用失败 → 跳过此模式

            # 执行分解调用
            try:
                decomposed_result = eval(decomposed_code, {}, local_env)
            except Exception as e:
                violations.append({
                    "pattern": pattern_name,
                    "issue": "分解调用失败",
                    "composite_result": Repeatable._safe_repr(composite_result),
                    "error": str(e),
                })
                continue

            # 比较
            if not Repeatable._outputs_equal(composite_result, decomposed_result):
                violations.append({
                    "pattern": pattern_name,
                    "issue": "复合结果 ≠ 分解结果",
                    "composite": Repeatable._safe_repr(composite_result),
                    "decomposed": Repeatable._safe_repr(decomposed_result),
                })

        if violations:
            return Verdict(
                self.name, "VIOLATION",
                reason=f"{len(violations)} 种分解模式不一致",
                details={"violations": violations},
            )
        return Verdict(self.name, "PASS")

    # 上下文/环境类参数名，拆解后语义不成立
    _CONTEXT_PARAM_NAMES = {
        "scope", "context", "env", "environment",
        "config", "settings", "options", "kwargs",
    }

    @staticmethod
    def _is_extra_nesting(dict_val: dict, composite, decomposed: dict) -> bool:
        """检测拆解是否引入了额外嵌套（dict 参数被 API 当作整体输入）。"""
        # 模式 A: composite 是 dict，decomposed[k] 多嵌套一层 {k: composite[k]}
        if isinstance(composite, dict):
            for k in dict_val:
                dv = decomposed.get(k)
                if not (isinstance(dv, dict) and k in dv):
                    return False
                if k in composite and not Repeatable._outputs_equal(dv[k], composite[k]):
                    return False
            return True
        # 模式 B: composite 不是 dict 但 decomposed 是
        # （拆解框架强行加了一层 dict 结构）
        if isinstance(decomposed, dict) and not isinstance(composite, dict):
            return True
        return False

    def _check_heuristic(self, api_name, run_api, params) -> Verdict:
        """启发式分解检查（无 LLM 方案时的后备）"""
        # 找到 dict 参数，排除上下文/环境类参数
        dict_params = {k: v for k, v in params.items()
                       if isinstance(v, dict) and v
                       and k not in self._CONTEXT_PARAM_NAMES}

        for pname, pval in dict_params.items():
            verdict = self._try_dict_decompose(
                api_name, run_api, params, pname, pval
            )
            if verdict.is_violation():
                return verdict

        return Verdict(self.name, "SKIP",
                       reason="无LLM方案且无dict参数可分解")

    def _try_dict_decompose(
        self, api_name, run_api, params, dict_key, dict_val
    ) -> Verdict:
        """尝试 {k: v} 分解"""
        # 复合调用
        try:
            composite = run_api(**params)
        except Exception:
            return Verdict(self.name, "SKIP",
                           reason="复合调用崩溃")

        # 分解: 对每个 key 单独调用
        decomposed = {}
        success = True
        for k, v in dict_val.items():
            # 构造单key输入: 把 dict 参数替换为 {k: v}
            sub_params = dict(params)
            sub_params[dict_key] = {k: v}
            try:
                decomposed[k] = run_api(**sub_params)
            except Exception as e:
                decomposed[k] = f"ERROR: {type(e).__name__}: {e}"
                success = False

        if not success:
            return Verdict(self.name, "SKIP",
                           reason="部分分解调用崩溃")

        # 比较
        if not Repeatable._outputs_equal(composite, decomposed):
            # 如果每个子调用结果都等于复合结果，说明 dict 被当作原子单元
            # （如 glom 的 Spec dict），拆解假设不成立
            if all(Repeatable._outputs_equal(composite, v)
                   for v in decomposed.values()
                   if not isinstance(v, str) or not v.startswith("ERROR:")):
                return Verdict(
                    self.name, "SKIP",
                    reason=f"dict参数 '{dict_key}' 表现为原子单元，拆解不适用",
                    details={
                        "composite": Repeatable._safe_repr(composite),
                        "decomposed": Repeatable._safe_repr(decomposed),
                    },
                )
            # 检查是否拆解引入了额外嵌套（API 把 dict 当整体输入处理）
            if Decompose._is_extra_nesting(dict_val, composite, decomposed):
                return Verdict(
                    self.name, "SKIP",
                    reason=f"dict参数 '{dict_key}' 拆解引入额外嵌套，拆解假设不成立",
                    details={
                        "composite": Repeatable._safe_repr(composite),
                        "decomposed": Repeatable._safe_repr(decomposed),
                    },
                )
            return Verdict(
                self.name, "VIOLATION",
                reason=f"dict参数 '{dict_key}' 分解不一致",
                details={
                    "composite": Repeatable._safe_repr(composite),
                    "decomposed": Repeatable._safe_repr(decomposed),
                },
            )

        return Verdict(self.name, "PASS")


# ============================================================
# 7. LLM 分解方案生成器 (每个 API 一次，缓存复用)
# ============================================================

def generate_decompose_plans(
    api_names: list[str],
    conditions: dict,
    cache_path: str | None = None,
    client: Any = None,
) -> dict:
    """
    用 LLM 为每个 API 生成分解方案。
    每个 API 调用一次，结果缓存到 decompose_plans.json。
    """
    if cache_path is None:
        cache_path = (
            f"{root_path}/haoyahui/documentation/arg_boundary/"
            f"{lib_name}_decompose_plans.json"
        )

    # 加载缓存
    cached = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except Exception:
            cached = {}

    # 筛选需要生成方案的 API
    to_generate = []
    for name in api_names:
        if name not in cached and name in conditions:
            param_types = conditions[name].get("Parameter type", {})
            # 只对有复合参数结构的 API 生成
            has_composite = any(
                ind in str(v).lower()
                for v in param_types.values()
                for ind in ("dict", "spec", "path", "query", "tuple", "list of")
            )
            if has_composite:
                to_generate.append(name)

    if not to_generate or client is None:
        return cached

    print(f"[Decompose] 需要为 {len(to_generate)} 个 API 生成分解方案...")

    for i, api_name in enumerate(to_generate):
        if api_name in cached:
            continue

        api_doc = get_doc(api_name)
        if not api_doc:
            api_doc = "无文档"

        param_types = conditions[api_name].get("Parameter type", {})

        prompt = f"""You are analyzing a Python API for metamorphic testing.

API: {api_name}
Library: {lib_name}
Documentation: {api_doc}
Parameters: {json.dumps(param_types, ensure_ascii=False)}

Determine if this API supports DECOMPOSITION testing:
Can a call with compound parameters be correctly broken down into simpler sub-calls?

Examples of decomposition:
- api(x, {{'k1': v1, 'k2': v2}}) == {{'k1': api(x, v1), 'k2': api(x, v2)}}
- api(x, 'a.b.c') == api(api(api(x, 'a'), 'b'), 'c')
- api(x, (step1, step2)) == step2(step1(x))

Output a JSON object:
If decomposition applies:
{{"applies": true, "patterns": [
  {{"name": "<pattern_name>",
   "condition": "<Python expression that must be True for this pattern, or empty string>",
   "composite_code": "<Python expression for composite call using run_api(...)>",
   "decomposed_code": "<Python expression for decomposed calls using run_api(...)>"}}
]}}
If NOT applicable:
{{"applies": false, "reason": "<one sentence why>"}}

Rules:
- Use `run_api` as the function name (it's already defined)
- Use the exact parameter names from the Parameters list
- composite_code and decomposed_code must be valid Python
- Output ONLY the JSON object, no markdown, no explanation."""

        try:
            response = client.chat.completions.create(
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "You generate metamorphic test decomposition plans. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                stream=False,
            )
            text = response.choices[0].message.content

            # 提取 JSON
            json_start = text.find("{")
            json_end = text.rfind("}")
            if json_start >= 0 and json_end > json_start:
                plan = json.loads(text[json_start:json_end + 1])
                cached[api_name] = plan
                print(f"  [{i+1}/{len(to_generate)}] {api_name}: "
                      f"applies={plan.get('applies')}")
            else:
                cached[api_name] = {"applies": False,
                                    "reason": "LLM 输出无法解析"}
        except Exception as e:
            print(f"  [{i+1}/{len(to_generate)}] {api_name}: 失败 - {e}")
            cached[api_name] = {"applies": False, "reason": str(e)}

        time.sleep(0.1)  # 谦让 API

    # 保存缓存
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, indent=2, ensure_ascii=False)
    print(f"[Decompose] 方案已缓存至 {cache_path}")

    return cached


# ============================================================
# 8. 输入采样器
# ============================================================

class InputSampler:
    """从 inputs_dict 中采样具体参数值"""

    def __init__(self, inputs_dict: dict, seed: int = 42):
        import random
        self._rng = random.Random(seed)
        self._inputs = inputs_dict  # {param: {type, values: [...]}}

    def sample_one(self) -> tuple[list[str], list[Any], list[str]]:
        """
        从每个参数的候选值中随机选一个。
        返回 (参数名列表, 参数值列表, 类型标签列表)。
        """
        names = []
        values = []
        types = []
        for pname, pinfo in self._inputs.items():
            if isinstance(pinfo, dict):
                candidates = pinfo.get("values", [])
                ptype = pinfo.get("type", "literal")
            else:
                candidates = pinfo
                ptype = "literal"

            if not candidates:
                continue

            val = self._rng.choice(candidates)
            names.append(pname)
            values.append(val)
            types.append(ptype)

        return names, values, types

    def sample_many(self, n: int) -> list[tuple[list[str], list[Any], list[str]]]:
        return [self.sample_one() for _ in range(n)]


# ============================================================
# 9. 蜕变测试执行器
# ============================================================

class MetamorphicRunner:
    """蜕变测试主执行器 — 完全库无关"""

    def __init__(
        self,
        lib: str | None = None,
        num_samples_per_api: int = 15,
        decompose_plans: dict | None = None,
    ):
        self.lib = lib or lib_name
        self.num_samples = num_samples_per_api
        self._decompose_plans = decompose_plans or {}

        # 注册所有蜕变关系 (按优先级)
        self._relations: list[MetamorphicRelation] = [
            NonMutation(),
            Repeatable(),
            Idempotent(),
            Decompose(decompose_plans=self._decompose_plans),
        ]

        # 结果收集
        self._violations: list[dict] = []     # 违规 (疑似 bug)
        self._passes: list[dict] = []         # 通过
        self._skips: list[dict] = []          # 跳过
        self._errors: list[dict] = []         # 执行错误

        # 统计
        self._stats = Counter()

    @staticmethod
    def _safe_repr(obj: Any) -> str:
        """安全的 repr，防止被测对象的 __repr__ 有 bug 导致崩溃。"""
        try:
            s = repr(obj)
            return s[:200]
        except Exception:
            return f"<{type(obj).__name__}: repr failed>"

    # ── 公共入口 ────────────────────────────────────────

    def run(
        self,
        api_names: list[str] | None = None,
        verbose: bool = True,
    ) -> dict:
        """对所有 API 执行蜕变测试。返回汇总报告 dict。"""
        if api_names is None:
            api_names = self._load_api_list()

        total = len(api_names)
        for i, api_name in enumerate(api_names):
            if verbose:
                print(f"\n{'='*60}")
                print(f"[{i+1}/{total}] {api_name}")
                print(f"{'='*60}")

            result = self.run_one(api_name, verbose=verbose)
            self._stats["apis_tested"] += 1

            if verbose:
                counts = Counter(v.get("relation") for v in result["violations"] if result.get("violations"))
                stats_str = f"  PASS={result['passes']} VIOLATION={result['violations_len']} SKIP={result['skips']}"
                if result["violations_len"] > 0:
                    stats_str += "  ⚠️"
                print(stats_str)

        return self._build_report(total)

    def run_one(self, api_name: str, verbose: bool = False) -> dict:
        """对单个 API 执行蜕变测试"""
        # 1. 加载 run_api
        run_api = _load_run_api(api_name)
        if not run_api:
            self._stats["no_run_api"] += 1
            if verbose:
                print(f"  [跳过] 无 run_api")
            return {"api_name": api_name, "passes": 0, "violations_len": 0,
                    "skips": 0, "violations": []}

        # 2. 加载输入候选值
        inputs_dict = read_json_api(
            api_name=api_name,
            file_path=f"../documentation/api_input/",
            read_mode="inputs",
        )
        if not inputs_dict:
            self._stats["no_inputs"] += 1
            if verbose:
                print(f"  [跳过] 无测试输入")
            return {"api_name": api_name, "passes": 0, "violations_len": 0,
                    "skips": 0, "violations": []}

        # 3. 加载条件和边界
        conditions = read_json_api(
            api_name=api_name,
            file_path=f"../documentation/conditions/",
            read_mode="conditions",
        )
        api_boundary = read_json_api(
            api_name=api_name,
            file_path=f"../documentation/arg_boundary/",
            read_mode="boundary",
        )

        # 4. 确定适用的蜕变关系
        applicable = []
        for rel in self._relations:
            ok, reason = rel.applies_to(api_name, conditions, api_boundary)
            if ok:
                applicable.append(rel)
                self._stats[f"relation_{rel.name}"] += 1
                if verbose:
                    print(f"  ✓ {rel.name}: {reason}")
            else:
                if verbose:
                    print(f"  - {rel.name}: {reason}")

        if not applicable:
            self._stats["no_applicable_relations"] += 1
            return {"api_name": api_name, "passes": 0, "violations_len": 0,
                    "skips": 0, "violations": []}

        # 5. 采样输入并执行检查
        sampler = InputSampler(inputs_dict)
        violations = []
        local_passes = 0
        local_skips = 0

        for sample_idx in range(self.num_samples):
            names, raw_values, types = sampler.sample_one()

            # eval 参数值
            try:
                values = [
                    _eval_param_by_type(v, t)
                    for v, t in zip(raw_values, types)
                ]
            except Exception as e:
                self._stats["eval_error"] += 1
                continue

            # 对每条适用关系执行检查
            for rel in applicable:
                try:
                    verdict = rel.check(api_name, run_api, names, values, types)
                except Exception as e:
                    verdict = Verdict(
                        rel.name, "ERROR",
                        reason=f"检查执行异常: {traceback.format_exc()[:500]}",
                    )

                self._stats[f"check_{verdict.result}"] += 1

                entry = {
                    "api_name": api_name,
                    "relation": rel.name,
                    "sample_index": sample_idx,
                    "verdict": verdict.result,
                    "reason": verdict.reason,
                    "details": verdict.details,
                    "inputs": dict(zip(names, [self._safe_repr(v) for v in values])),
                }

                if verdict.is_violation():
                    violations.append(entry)
                    if verbose:
                        print(f"    ⚠️ [{rel.name}] VIOLATION: {verdict.reason[:100]}")
                elif verdict.result == "PASS":
                    local_passes += 1
                elif verdict.result == "SKIP":
                    local_skips += 1
                    if verbose and sample_idx < 2:
                        print(f"    ⊘ [{rel.name}] SKIP: {verdict.reason[:80]}")

        self._violations.extend(violations)

        return {
            "api_name": api_name,
            "passes": local_passes,
            "violations_len": len(violations),
            "skips": local_skips,
            "violations": violations,
        }

    # ── 报告生成 ────────────────────────────────────────

    def _build_report(self, total_apis: int) -> dict:
        """生成汇总报告"""
        # 按严重度分类
        high_severity = []
        medium_severity = []
        low_severity = []

        for v in self._violations:
            if v["relation"] == "non_mutation":
                high_severity.append(v)
            elif v["relation"] in ("decompose",):
                medium_severity.append(v)
            else:
                low_severity.append(v)

        return {
            "library": self.lib,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_apis": total_apis,
                "apis_tested": self._stats["apis_tested"],
                "apis_no_run_api": self._stats.get("no_run_api", 0),
                "apis_no_inputs": self._stats.get("no_inputs", 0),
                "total_checks": self._stats.get("check_PASS", 0)
                + self._stats.get("check_VIOLATION", 0)
                + self._stats.get("check_SKIP", 0),
                "passes": self._stats.get("check_PASS", 0),
                "violations": len(self._violations),
                "skips": self._stats.get("check_SKIP", 0),
                "errors": self._stats.get("check_ERROR", 0),
            },
            "violations_by_severity": {
                "high": len(high_severity),
                "medium": len(medium_severity),
                "low": len(low_severity),
            },
            "relation_stats": {
                "non_mutation": self._stats.get("relation_non_mutation", 0),
                "repeatable": self._stats.get("relation_repeatable", 0),
                "idempotent": self._stats.get("relation_idempotent", 0),
                "decompose": self._stats.get("relation_decompose", 0),
            },
            "high_severity_violations": high_severity[:30],
            "medium_severity_violations": medium_severity[:30],
            "low_severity_violations": low_severity[:10],
        }

    def print_report(self, report: dict | None = None):
        """打印可读报告"""
        if report is None:
            report = self._build_report(0)

        s = report["summary"]
        print("\n" + "=" * 60)
        print("  蜕变测试报告 (Metamorphic Testing)")
        print("=" * 60)
        print(f"  库:       {report['library']}")
        print(f"  时间:     {report['timestamp']}")
        print(f"  API 总数: {s['total_apis']}")
        print(f"  已测试:   {s['apis_tested']}")
        print(f"  跳过(无run_api): {s['apis_no_run_api']}")
        print(f"  跳过(无输入):   {s['apis_no_inputs']}")
        print(f"  ─────────────────────────────")
        print(f"  总检查:   {s['total_checks']}")
        print(f"  通过:     {s['passes']}")
        print(f"  违规:     {s['violations']}  ⚠️")
        print(f"  跳过:     {s['skips']}")
        print(f"  错误:     {s['errors']}")
        print(f"  ─────────────────────────────")
        print(f"  严重度分布:")
        sev = report["violations_by_severity"]
        print(f"    HIGH:   {sev['high']}")
        print(f"    MEDIUM: {sev['medium']}")
        print(f"    LOW:    {sev['low']}")
        print(f"  ─────────────────────────────")
        print(f"  关系覆盖:")
        for rel, count in report["relation_stats"].items():
            print(f"    {rel}: {count} APIs")
        print("=" * 60)

        if sev["high"] > 0:
            print(f"\n  ⚠️ HIGH 严重度违规 (前10):")
            for v in report.get("high_severity_violations", [])[:10]:
                print(f"    [{v['api_name']}] {v['relation']}: {v['reason'][:120]}")

        if sev["medium"] > 0:
            print(f"\n  ⚠️ MEDIUM 严重度违规 (前10):")
            for v in report.get("medium_severity_violations", [])[:10]:
                print(f"    [{v['api_name']}] {v['relation']}: {v['reason'][:120]}")

    def save_report(self, report: dict, path: str | None = None):
        """保存报告为 JSON"""
        if path is None:
            path = (
                f"{root_path}/haoyahui/documentation/results/"
                f"{self.lib}_metamorphic_report.json"
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存至: {path}")

    # ── 辅助 ───────────────────────────────────────────

    def _load_api_list(self) -> list[str]:
        api_path = f"../documentation/lib_api/{self.lib}_APIdef.txt"
        return read_file(api_path)


# ============================================================
# 10. 便捷入口
# ============================================================

def run_metamorphic_tests(
    lib: str | None = None,
    num_samples: int = 15,
    use_llm_decompose: bool = False,
    verbose: bool = True,
) -> dict:
    """
    蜕变测试统一入口。

    参数:
        lib: 库名称，默认使用 config.py 中的 lib_name
        num_samples: 每个 API 采样测试次数
        use_llm_decompose: 是否使用 LLM 生成分解方案（需要 API）
        verbose: 是否打印详细进度
    """
    lib = lib or lib_name

    decompose_plans = {}
    if use_llm_decompose:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        api_names = read_file(f"../documentation/lib_api/{lib}_APIdef.txt")
        conditions_path = (
            f"{root_path}/haoyahui/documentation/conditions/"
            f"{lib}_conditions.json"
        )
        conditions = {}
        if os.path.exists(conditions_path):
            with open(conditions_path, "r", encoding="utf-8") as f:
                conditions = json.load(f)
        decompose_plans = generate_decompose_plans(
            api_names=api_names,
            conditions=conditions,
            client=client,
        )

    runner = MetamorphicRunner(
        lib=lib,
        num_samples_per_api=num_samples,
        decompose_plans=decompose_plans,
    )

    report = runner.run(verbose=verbose)
    runner.print_report(report)
    runner.save_report(report)

    return report


# ============================================================
# 11. 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="蜕变测试 (Metamorphic Testing)")
    parser.add_argument("--lib", type=str, default=None,
                        help="库名称 (默认: config.lib_name)")
    parser.add_argument("--samples", type=int, default=15,
                        help="每个 API 采样数 (默认: 15)")
    parser.add_argument("--llm-decompose", action="store_true",
                        help="使用 LLM 生成分解方案")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式")
    args = parser.parse_args()

    run_metamorphic_tests(
        lib=args.lib,
        num_samples=args.samples,
        use_llm_decompose=args.llm_decompose,
        verbose=not args.quiet,
    )
