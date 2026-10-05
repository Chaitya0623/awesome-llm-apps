"""Offline stand-in for Gemini inside the ADK graph: answers by which specialist is asking."""

from __future__ import annotations

import json
import re
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def quote_all(instruction: str, low=100, mid=200, high=300, collectible_ids=()) -> str:
    ids = re.findall(r"- id=(\S+)", instruction)
    return json.dumps([
        {"id": i, "low": low, "mid": mid, "high": high, "basis": "used, local listings",
         "collectible": i in collectible_ids}
        for i in ids
    ])


class FakeGemini(BaseLlm):
    model: str = "gemini-fake"
    request: dict = {}
    calls: list = []
    sources: list = ["https://shop.example/listing"]
    collectible_ids: tuple = ()

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        instruction = str(llm_request.config.system_instruction or "")
        if "intake specialist" in instruction:
            role, text = "normalizer", json.dumps(self.request)
        elif "book valuation specialist" in instruction:
            role, text = "books", quote_all(instruction, collectible_ids=self.collectible_ids)
        elif "contents valuation specialist" in instruction:
            role, text = "items", quote_all(instruction, 1000, 1500, 2500)
        else:
            role, text = "other", ""
        self.calls.append(role)
        grounding = types.GroundingMetadata(grounding_chunks=[
            types.GroundingChunk(web=types.GroundingChunkWeb(uri=uri)) for uri in self.sources
        ]) if role in {"books", "items"} else None
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            grounding_metadata=grounding,
        )
