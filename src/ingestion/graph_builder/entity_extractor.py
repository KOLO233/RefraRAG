"""实体和关系抽取器。

使用 LLM 从文本中抽取实体和关系，构建知识图谱。
Prompt 设计参考 LightRAG 的 entity_extraction_system_prompt。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple

from src.core.types import Entity, Relation

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """你是一个知识图谱专家，负责从文本中抽取实体和关系。

---指令---
1. **实体抽取**：
   - 识别文本中明确的、有意义的实体
   - 为每个实体指定类型：{entity_types}
   - 提供简洁的实体描述

2. **关系抽取**：
   - 识别已抽取实体之间的直接关系
   - 为每对关系指定关系类型（如：属于、导致、包含、使用、对比、依赖）
   - 提供简洁的关系描述

3. **输出格式**（严格遵守 JSON）：
```json
{{
  "entities": [
    {{"name": "实体名", "type": "实体类型", "description": "简短描述"}}
  ],
  "relations": [
    {{"source": "源实体", "target": "目标实体", "type": "关系类型", "description": "关系描述"}}
  ]
}}
```

4. **规则**：
   - 实体名称保持一致性（同一实体用相同名称）
   - 仅输出 JSON，不要输出其他内容
   - 实体描述和关系描述基于文本内容，不要编造

---文本---
{text}
"""


class EntityExtractor:
    """LLM 实体/关系抽取器。

    Example:
        >>> extractor = EntityExtractor(llm_service, entity_types=["疾病", "药物", "症状"])
        >>> entities, relations = await extractor.extract("阿司匹林可以缓解头痛...")
    """

    def __init__(self, llm_service=None, entity_types: List[str] = None):
        self._llm = llm_service
        self._entity_types = entity_types or [
            "概念", "技术", "方法", "人物", "组织", "工具", "理论"
        ]

    async def extract(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """从文本中抽取实体和关系。

        Args:
            text: 输入文本

        Returns:
            (entities, relations) 二元组
        """
        if not text.strip():
            return [], []

        if self._llm is None:
            logger.warning("No LLM configured, using rule-based extraction")
            return self._rule_based_extract(text)

        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(self._entity_types),
            text=text[:3000],  # 截断避免超 token
        )

        try:
            response = await self._llm.ainvoke(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._rule_based_extract(text)

    def _parse_response(self, response: str) -> Tuple[List[Entity], List[Relation]]:
        """解析 LLM 的 JSON 响应。"""
        # 提取 JSON 块
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            logger.warning("No JSON found in LLM response")
            return [], []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            return [], []

        entities = []
        for item in data.get("entities", []):
            entities.append(Entity(
                name=item.get("name", ""),
                entity_type=item.get("type", "概念"),
                description=item.get("description", ""),
            ))

        relations = []
        for item in data.get("relations", []):
            relations.append(Relation(
                source=item.get("source", ""),
                target=item.get("target", ""),
                relation_type=item.get("type", "related"),
                description=item.get("description", ""),
            ))

        return entities, relations

    def _rule_based_extract(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """规则基线抽取（无 LLM 时的降级方案）。

        简单的基于关键词和模式的实体抽取。
        """
        entities = []
        seen = set()

        # 中文术语模式
        patterns = [
            (r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)', "技术"),  # 英文术语
            (r'([一-鿿]{2,6}(?:学习|网络|算法|模型|函数|机制|方法|系统|框架))', "技术"),
            (r'([一-鿿]{2,4}(?:学习|处理|视觉|智能|推理|检索))', "概念"),
        ]

        for pattern, etype in patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if name not in seen and len(name) >= 2:
                    seen.add(name)
                    entities.append(Entity(
                        name=name,
                        entity_type=etype,
                        description="",
                    ))

        # 简单关系：相邻实体之间的 "属于" 关系
        relations = []
        for i in range(len(entities) - 1):
            e1 = entities[i]
            e2 = entities[i + 1]
            # 如果两个实体在文本中距离较近，认为有关系
            if e1.name in text and e2.name in text:
                idx1 = text.index(e1.name)
                idx2 = text.index(e2.name)
                if abs(idx1 - idx2) < 200:
                    relations.append(Relation(
                        source=e1.name,
                        target=e2.name,
                        relation_type="共现",
                        description=f"{e1.name}和{e2.name}在同一上下文中出现",
                    ))

        return entities, relations
