# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import httpx
import pytest

from nat.plugins.intake.observability.exporter.intake_exporter import IntakeExporter


def _client_recording_to(received: list[httpx.Request], *, status: int = 201, body: dict | None = None) -> httpx.AsyncClient:
    """Build an AsyncClient backed by httpx.MockTransport that records every request."""

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(status, json=body or {"name": "entry-id"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _make_exporter(client: httpx.AsyncClient, **overrides) -> IntakeExporter:
    defaults = dict(
        endpoint="https://nmp.example.com",
        workspace="default",
        app="default/coding-agent",
        task="code-generation",
        client=client,
    )
    defaults.update(overrides)
    return IntakeExporter(**defaults)


def test_url_constructed_from_endpoint_and_workspace() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(204)))
    exporter = _make_exporter(client, endpoint="https://nmp.example.com/", workspace="ws-1")
    assert exporter._url == "https://nmp.example.com/apis/intake/v2/workspaces/ws-1/entries"


@pytest.mark.asyncio
async def test_export_processed_posts_each_entry_to_intake() -> None:
    received: list[httpx.Request] = []
    client = _client_recording_to(received)
    exporter = _make_exporter(client)

    await exporter.export_processed([
        {"external_id": "a", "data": {}, "context": {}},
        {"external_id": "b", "data": {}, "context": {}},
    ])

    assert [r.url.path for r in received] == [
        "/apis/intake/v2/workspaces/default/entries",
        "/apis/intake/v2/workspaces/default/entries",
    ]
    assert [r.method for r in received] == ["POST", "POST"]


@pytest.mark.asyncio
async def test_empty_batch_is_no_op() -> None:
    received: list[httpx.Request] = []
    client = _client_recording_to(received)
    exporter = _make_exporter(client)

    await exporter.export_processed([])
    await exporter.export_processed([{}])  # falsy entries are filtered

    assert received == []


@pytest.mark.asyncio
async def test_single_dict_input_is_wrapped_into_a_batch() -> None:
    received: list[httpx.Request] = []
    client = _client_recording_to(received)
    exporter = _make_exporter(client)

    await exporter.export_processed({"external_id": "solo", "data": {}, "context": {}})

    assert len(received) == 1


@pytest.mark.asyncio
async def test_422_logs_and_drops_without_retry(caplog: pytest.LogCaptureFixture) -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(422, text="schema violation: missing 'role'")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    exporter = _make_exporter(client)

    with caplog.at_level(logging.ERROR):
        await exporter.export_processed([{"external_id": "x", "data": {}, "context": {}}])

    assert len(received) == 1
    assert any("422" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_5xx_failure_is_logged_but_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    exporter = _make_exporter(client)

    with caplog.at_level(logging.ERROR):
        await exporter.export_processed([{"external_id": "x", "data": {}, "context": {}}])

    assert any("503" in record.message or "intake POST failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_cleanup_closes_http_client() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(204)))
    exporter = _make_exporter(client)

    await exporter._cleanup()

    assert client.is_closed
