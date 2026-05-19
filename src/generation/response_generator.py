"""分级响应生成器。

根据查询级别（L1-L4）选择不同的生成策略：
- L1: 直接回答（最简单）
- L2: 结构化分析
- L3: 专家角色 Chain-of-Thought 推理
- L4: Self-RAG 迭代批判

这是 RefraRAG 的核心创新模块。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"


def _load_prompt(name: str) -> str:
    """从 prompts 目录加载模板。"""
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    # 回退到内置默认
    return _DEFAULT_PROMPTS.get(name, "")


_DEFAULT_PROMPTS = {
    "generate_l1.txt": (
        "你是一个专业的领域问答助手。\n"
        "请严格根据以下参考资料回答用户问题。\n"
        "规则：仅使用参考资料中的信息回答，不要编造。"
        "如果信息不足请说明。引用时标注来源编号 [1][2]。\n\n"
        "参考资料：\n{context}\n\n用户问题：{question}\n\n请直接给出回答："
    ),
    "generate_l2.txt": (
        "你是一个专业的领域分析助手。\n"
        "请根据参考资料对用户问题进行结构化分析。\n"
        "参考资料：\n{context}\n\n用户问题：{question}\n\n"
        "## 分析\n（分维度详细分析）\n\n## 综合结论"
    ),
    "generate_l3_cot.txt": (
        "你是一位资深的领域专家。\n"
        "请使用 Chain-of-Thought 方法逐步推理。\n"
        "参考资料：\n{context}\n\n用户问题：{question}\n\n"
        "## 问题理解\n## 逐步推理\nStep 1: ...\n## 结论"
    ),
    "generate_l4_selfrag.txt": (
        "你是一位资深领域专家，擅长假设性推理。\n"
        "请进行 Self-RAG 迭代推理。\n"
        "参考资料：\n{context}\n\n用户问题：{question}\n\n"
        "## 初始回答\n## 自我批判\n## 精炼回答"
    ),
}


class ResponseGenerator:
    """分级响应生成器。

    根据查询级别自动选择 Prompt 模板和生成策略。
    L4 级别使用 Self-RAG 迭代引擎（真正的 检索→生成→批判→迭代 循环）。

    Example:
        >>> gen = ResponseGenerator(llm_service, hybrid_search)
        >>> answer = await gen.generate("什么是机器学习？", context, level="L1")
    """

    def __init__(self, llm_service=None, hybrid_search=None):
        self._llm = llm_service
        self._search = hybrid_search
        self._prompts = {
            "L1": _load_prompt("generate_l1.txt"),
            "L2": _load_prompt("generate_l2.txt"),
            "L3": _load_prompt("generate_l3_cot.txt"),
        }
        # Self-RAG 引擎（L4 专用）
        self._self_rag = None

    async def generate(
        self,
        question: str,
        context: str,
        level: str = "L1",
        query=None,
    ) -> str:
        """根据查询级别生成回答。

        Args:
            question: 用户问题
            context: 检索到的上下文
            level: 查询级别 (L1/L2/L3/L4)
            query: ProcessedQuery 对象（L4 Self-RAG 需要）

        Returns:
            生成的回答文本
        """
        if self._llm is None:
            return self._fallback_answer(question, context, level)

        # L4 使用 Self-RAG 迭代引擎
        if level == "L4":
            return await self._generate_with_self_rag(question, context, query)

        # L1-L3 使用对应 Prompt 直接生成
        prompt_template = self._prompts.get(level, self._prompts["L1"])

        max_context_len = 6000
        if len(context) > max_context_len:
            context = context[:max_context_len] + "\n\n...(上下文已截断)"

        prompt = prompt_template.format(context=context, question=question)

        try:
            answer = await self._llm.ainvoke(prompt)
            logger.info(f"[{level}] Generated {len(answer)} chars for '{question[:40]}...'")
            return answer.strip()
        except Exception as e:
            logger.error(f"[{level}] Generation failed: {e}")
            return self._fallback_answer(question, context, level)

    async def _generate_with_self_rag(
        self,
        question: str,
        context: str,
        query=None,
    ) -> str:
        """使用 Self-RAG 迭代引擎生成 L4 回答。"""
        from src.generation.self_rag import SelfRAG
        from src.core.settings import load_settings

        settings = load_settings()
        max_iter = settings.generation.max_self_rag_iterations

        if self._self_rag is None:
            self._self_rag = SelfRAG(
                llm_service=self._llm,
                hybrid_search=self._search,
                max_iterations=max_iter,
                pass_threshold=7.0,
            )

        logger.info(f"[L4] Starting Self-RAG (max {max_iter} iterations)")

        result = await self._self_rag.run(
            question=question,
            query=query,
            initial_context=context,
            initial_answer="",  # 让 Self-RAG 从头生成
        )

        iterations = result.get("total_iterations", 1)
        final_critique = result.get("final_critique")
        overall_score = final_critique.overall if final_critique else 0.0

        logger.info(
            f"[L4] Self-RAG completed: {iterations} iterations, "
            f"final score: {overall_score:.1f}"
        )

        return result.get("final_answer", "")

    @staticmethod
    def _fallback_answer(question: str, context: str, level: str) -> str:
        """无 LLM 时的降级回答。"""
        if not context:
            return f"[{level}] 未找到与问题相关的参考资料，无法回答。"
        snippet = context[:500] + ("..." if len(context) > 500 else "")
        return (
            f"[{level}] 查询: {question}\n\n"
            f"基于检索到的参考资料，以下是相关内容：\n\n{snippet}\n\n"
            f"（注：LLM 未配置，以上为检索结果摘要，非生成答案）"
        )
