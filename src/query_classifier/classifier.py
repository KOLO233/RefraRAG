"""四级查询分类器。

将用户问题分为 L1-L4 四个复杂度级别，并路由到对应的检索/生成策略。

级别定义（基于开题报告）：
- L1 显性事实：答案直接出现在文档中（"什么是X？"）
- L2 隐性事实：需要跨段落/文档推理（"比较A和B"）
- L3 可解释原理：需要领域知识推理（"为什么X会导致Y？"）
- L4 隐藏原理：需要深层推理和假设（"如果改变X会怎样？"）
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.core.types import QueryClassification
from src.core.settings import QueryClassifierSettings

logger = logging.getLogger(__name__)


# ===========================================================================
# 规则基线分类器
# ===========================================================================

# 各级别的匹配规则，每条规则有权重
# 高权重规则（模式匹配）优先于低权重规则（关键词匹配）
_LEVEL_RULES = [
    # ===== L1: 显性事实（定义、列举）=====
    # 高权重模式："什么是X" / "X是什么" 开头的定义类问题
    {"level": "L1", "weight": 3.0, "type": "pattern",
     "pattern": r"^什么是"},
    {"level": "L1", "weight": 3.0, "type": "pattern",
     "pattern": r"是什么[？?。]?$"},
    {"level": "L1", "weight": 3.0, "type": "pattern",
     "pattern": r"^列举|^列出|^请列出"},
    {"level": "L1", "weight": 3.0, "type": "pattern",
     "pattern": r"有哪些[？?。]?$"},
    # 中权重关键词
    {"level": "L1", "weight": 1.5, "type": "keyword",
     "keywords": ["定义", "概念", "含义", "意思", "介绍", "定义是", "what is", "define"]},

    # ===== L2: 隐性事实（比较、区别）=====
    {"level": "L2", "weight": 3.0, "type": "pattern",
     "pattern": r"和.{1,10}(区别|异同|不同|对比|比较)"},
    {"level": "L2", "weight": 3.0, "type": "pattern",
     "pattern": r"比较.{1,10}(与|和)"},
    {"level": "L2", "weight": 3.0, "type": "pattern",
     "pattern": r"相比"},
    {"level": "L2", "weight": 1.5, "type": "keyword",
     "keywords": ["比较", "区别", "异同", "对比", "优劣", "不同", "相似", "差异", "compare", "versus"]},

    # ===== L3: 可解释原理（因果推理）=====
    {"level": "L3", "weight": 3.0, "type": "pattern",
     "pattern": r"^为什么"},
    {"level": "L3", "weight": 3.0, "type": "pattern",
     "pattern": r"的原因是什么"},
    {"level": "L3", "weight": 2.0, "type": "pattern",
     "pattern": r"^解释.{2,15}(原理|机制|原因)"},
    {"level": "L3", "weight": 1.0, "type": "keyword",
     "keywords": ["为什么", "原因", "机理", "原理", "机制", "推导", "如何理解", "本质", "why", "mechanism"]},

    # ===== L4: 隐藏原理（假设推理）=====
    {"level": "L4", "weight": 3.0, "type": "pattern",
     "pattern": r"^如果.{2,20}(会|将|怎样)"},
    {"level": "L4", "weight": 3.0, "type": "pattern",
     "pattern": r"^假设.{2,20}(会|那么)"},
    {"level": "L4", "weight": 1.5, "type": "keyword",
     "keywords": ["假设", "推测", "预测", "假如", "推断", "设想", "what would happen"]},
]


class RuleClassifier:
    """基于规则的快速查询分类器。

    使用加权评分机制：
    - 模式匹配（高权重 3.0）：句式结构决定级别，如"什么是X"→L1
    - 关键词匹配（中权重 1.5）：领域关键词辅助判断
    - 普通关键词（低权重 1.0）：兜底判断

    这种设计避免了"什么是注意力机制"被误判为 L3 的问题，
    因为"什么是"的模式权重(3.0)高于"机制"的关键词权重(1.0)。
    """

    def classify(self, query: str) -> QueryClassification:
        """对查询进行加权评分分类。

        遍历所有规则，为每个级别累计分数，取最高分的级别。
        """
        query_lower = query.lower().strip()
        scores = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "L4": 0.0}
        matched_reasons = {"L1": [], "L2": [], "L3": [], "L4": []}

        for rule in _LEVEL_RULES:
            level = rule["level"]
            weight = rule["weight"]

            if rule["type"] == "pattern":
                if re.search(rule["pattern"], query):
                    scores[level] += weight
                    matched_reasons[level].append(f"pattern:{rule['pattern']}")

            elif rule["type"] == "keyword":
                for kw in rule["keywords"]:
                    if kw in query_lower:
                        scores[level] += weight
                        matched_reasons[level].append(f"keyword:{kw}")
                        break  # 每条规则只匹配一次

        # 选最高分的级别
        best_level = max(scores, key=scores.get)
        best_score = scores[best_level]

        if best_score == 0:
            return QueryClassification(
                level="L1",
                confidence=0.5,
                query_type="factual",
                reasoning="No rule matched, default to L1",
            )

        # 计算置信度（归一化到 0.5-0.95）
        total = sum(scores.values())
        confidence = min(0.5 + 0.45 * (best_score / total), 0.95) if total > 0 else 0.5

        return QueryClassification(
            level=best_level,
            confidence=round(confidence, 2),
            query_type=self._infer_type(best_level),
            reasoning=f"Scored {best_score:.1f}: {', '.join(matched_reasons[best_level][:3])}",
        )

    def _infer_type(self, level: str) -> str:
        type_map = {"L1": "factual", "L2": "comparative", "L3": "causal", "L4": "hypothetical"}
        return type_map.get(level, "factual")


# ===========================================================================
# LLM 分类器
# ===========================================================================

CLASSIFICATION_PROMPT = """你是一个查询复杂度分类器。请将用户问题分为以下四个级别之一：

L1 (显性事实): 答案直接出现在文档中的简单事实查询。
  特征：定义、概念解释、列举、简单事实
  示例："什么是光合作用？" "列举Python的数据类型"

L2 (隐性事实): 需要跨段落或跨文档推理的事实查询。
  特征：比较、对比、区别、综合多个信息点
  示例："比较监督学习和无监督学习的区别" "A药和B药的副作用有何不同"

L3 (可解释原理): 需要领域知识进行因果推理的查询。
  特征：为什么、原因、机理、深层解释
  示例："为什么该患者服用A药后出现B症状？" "解释梯度消失的原因"

L4 (隐藏原理): 需要深层推理、假设或预测的查询。
  特征：假设性、预测性、推断性、反事实推理
  示例："如果将学习率增大10倍会怎样？" "假设该药物作用于C受体，可能的机制是什么？"

用户问题：{query}

请以 JSON 格式输出：
{{"level": "L1/L2/L3/L4", "confidence": 0.0-1.0, "query_type": "factual/comparative/causal/hypothetical", "reasoning": "分类理由"}}
"""


class LLMClassifier:
    """基于 LLM 的精确查询分类器。

    使用 LLM 的结构化输出进行四级分类，准确率高但延迟较高。
    """

    def __init__(self, llm_func=None):
        """
        Args:
            llm_func: LLM 调用函数，签名为 (prompt: str) -> str
        """
        self._llm_func = llm_func

    async def classify(self, query: str) -> QueryClassification:
        """使用 LLM 对查询进行分类。"""
        if self._llm_func is None:
            logger.warning("LLM function not configured, falling back to rule-based")
            return RuleClassifier().classify(query)

        prompt = CLASSIFICATION_PROMPT.format(query=query)

        try:
            import json
            response = await self._llm_func(prompt)
            # 尝试解析 JSON 响应
            result = json.loads(response)
            return QueryClassification(
                level=result.get("level", "L1"),
                confidence=float(result.get("confidence", 0.5)),
                query_type=result.get("query_type", "factual"),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return RuleClassifier().classify(query)


# ===========================================================================
# 混合分类器（默认）
# ===========================================================================

class HybridClassifier:
    """混合查询分类器。

    流程：
    1. 规则快速预分类
    2. 如果置信度低于阈值，使用 LLM 精确确认
    3. 综合两个结果给出最终分类

    这是系统默认使用的分类器。
    """

    def __init__(
        self,
        settings: QueryClassifierSettings,
        llm_func=None,
    ):
        self._settings = settings
        self._rule_classifier = RuleClassifier()
        self._llm_classifier = LLMClassifier(llm_func=llm_func)

    async def classify(self, query: str) -> QueryClassification:
        """混合分类：规则 + LLM。"""
        # Step 1: 规则预分类
        rule_result = self._rule_classifier.classify(query)

        # 如果模式为纯规则，直接返回
        if self._settings.mode == "rule":
            return rule_result

        # Step 2: 检查置信度
        if rule_result.confidence >= self._settings.llm_threshold:
            logger.debug(
                f"Rule classifier confident enough "
                f"({rule_result.confidence:.2f} >= {self._settings.llm_threshold}), "
                f"level={rule_result.level}"
            )
            return rule_result

        # Step 3: LLM 精确分类
        if self._settings.mode in ("llm", "hybrid"):
            logger.debug(
                f"Rule confidence low ({rule_result.confidence:.2f}), "
                f"invoking LLM classifier"
            )
            llm_result = await self._llm_classifier.classify(query)

            # 以 LLM 结果为主，规则结果为参考
            if llm_result.confidence > rule_result.confidence:
                return llm_result

        return rule_result


def create_classifier(
    settings: QueryClassifierSettings,
    llm_func=None,
) -> HybridClassifier:
    """工厂函数：根据配置创建分类器。"""
    return HybridClassifier(settings=settings, llm_func=llm_func)
