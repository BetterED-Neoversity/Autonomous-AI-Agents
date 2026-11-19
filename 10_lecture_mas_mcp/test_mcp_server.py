"""Тести MCP-сервера і guardrails. Це той мінімум, який вимагає ДЗ-3.

Дві групи, і поділ між ними принциповий:

    * бізнес-логіка (*_payload) тестується як звичайний Python — швидко,
      без підпроцесів і без LLM. Саме заради цього вона винесена з @mcp.tool();
    * один async-тест піднімає справжній сервер через stdio і перевіряє
      транспортний рівень: initialize, list_tools, call_tool.

Guardrails тестуються окремо (test_guardrails.py) — вони не частина сервера.

Запуск:  pytest -v            (LLM не потрібен)
"""
from __future__ import annotations

import json
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_server import (
    get_order_status_payload,
    get_payment_payload,
    process_refund_payload,
    search_faq_payload,
)

# ── Бізнес-логіка ─────────────────────────────────────────────────────────────

def test_order_existing():
    data = get_order_status_payload("ORD-001")
    assert data["status"] == "delivered"
    assert data["total"] == 1250.00


def test_order_hides_pii():
    """PII не має виходити за межі сервера навіть у службових полях."""
    assert "customer_email" not in get_order_status_payload("ORD-001")


def test_order_not_found():
    assert "error" in get_order_status_payload("ORD-999")


def test_payment_existing():
    assert get_payment_payload("PAY-001")["status"] == "completed"


def test_payment_not_found():
    assert "error" in get_payment_payload("PAY-999")


def test_refund_success():
    data = process_refund_payload("ORD-001", "Дефект товару")
    assert data["status"] == "processed"
    assert data["amount"] == 1250.00


def test_refund_rejected_for_wrong_status():
    """Бізнес-правило живе у коді, тому LLM не може його обійти."""
    data = process_refund_payload("ORD-003", "Передумав")
    assert "error" in data and "processing" in data["error"]


def test_refund_requires_reason():
    assert "error" in process_refund_payload("ORD-001", "bad")


def test_faq_found():
    data = search_faq_payload("delivery")
    assert any("доставка" in item["answer"].lower() for item in data)


def test_faq_not_found():
    assert search_faq_payload("xyznonexistent")[0]["topic"] == "no_results"


# ── Транспортний рівень MCP ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_server_speaks_mcp():
    """Сервер піднімається, оголошує 4 tools і коректно виконує виклик."""
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        names = {t.name for t in (await session.list_tools()).tools}
        assert names == {"check_order_status", "check_payment", "process_refund", "search_faq"}

        result = await session.call_tool("check_order_status", {"order_id": "ORD-002"})
        payload = json.loads(result.content[0].text)
        assert payload["status"] == "shipped"
        assert "customer_email" not in payload
