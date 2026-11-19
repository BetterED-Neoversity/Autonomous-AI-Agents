"""Міст між LangChain-моделлю і бібліотеками deepeval / deepteam.

Обидві бібліотеки без явного model=… ідуть в OpenAI і падають на відсутньому
OPENAI_API_KEY. LangChainJudge підключає туди будь-якого провайдера з
provider.py — той самий, яким працює сам агент.

Потрібен s6b_deepeval.py (метрики-судді) і s5b_deepteam.py (генерація атак
і вердикти). Живе окремим файлом, щоб red-teaming не залежав від скрипта
про evals.
"""
from __future__ import annotations

import os
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM

from provider import make_llm, text_of


class LangChainJudge(DeepEvalBaseLLM):
    """Модель-суддя: чотири методи, яких вимагає DeepEvalBaseLLM.

    deepeval просить структуровану відповідь (schema=pydantic-клас), LangChain
    вміє це через with_structured_output — тому вся обгортка вміщається в екран.

    Anyʼи в анотаціях — не лінь: базовий клас deepeval обіцяє одні типи,
    а фактично очікує інші. Подробиці — у code_walkthrough/.
    """

    def __init__(self, name: str | None = None):
        self.name = name or os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or "judge"
        self.model = make_llm()

    def load_model(self) -> Any:
        return self.model

    def get_model_name(self) -> str:
        return self.name

    def generate(self, prompt: str, schema=None) -> Any:
        if schema is not None:
            return self.model.with_structured_output(schema).invoke(prompt)
        return text_of(self.model.invoke(prompt))

    async def a_generate(self, prompt: str, schema=None) -> Any:
        if schema is not None:
            return await self.model.with_structured_output(schema).ainvoke(prompt)
        return text_of(await self.model.ainvoke(prompt))
