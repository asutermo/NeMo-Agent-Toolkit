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
"""Intake telemetry exporter.

Assembles a four-stage processor chain (Span → EntryInput → dict → batched →
filtered) and POSTs each entry to the NeMo Intake v2 producer endpoint.
"""

import logging

import httpx

from nat.builder.context import ContextState
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.span import Span
from nat.observability.exporter.span_exporter import SpanExporter
from nat.observability.processor.batching_processor import BatchingProcessor
from nat.observability.processor.falsy_batch_filter_processor import DictBatchFilterProcessor
from nat.utils.type_utils import override

from nat.plugins.intake.observability.processor import IntakeEntryToDictProcessor
from nat.plugins.intake.observability.processor import SpanToIntakeEntryProcessor

logger = logging.getLogger(__name__)

_INTAKE_PATH_TEMPLATE = "/apis/intake/v2/workspaces/{workspace}/entries"


class _DictBatchingProcessor(BatchingProcessor[dict]):
    """Batching processor specialized for dict payloads (matches DFW pattern)."""


class IntakeExporter(SpanExporter[Span, dict]):
    """SpanExporter that publishes normalized LLM interactions to NeMo Intake.

    The HTTP client is owned by this exporter and closed in :meth:`_cleanup`.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        workspace: str,
        app: str,
        task: str,
        client: httpx.AsyncClient,
        default_model: str = "unknown-model",
        project: str | None = None,
        event_types: frozenset[str] | None = None,
        context_state: ContextState | None = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_queue_size: int = 1000,
        drop_on_overflow: bool = False,
        shutdown_timeout: float = 10.0,
    ) -> None:
        super().__init__(context_state=context_state)
        self._endpoint = endpoint.rstrip("/")
        self._workspace = workspace
        self._client = client
        self._url = f"{self._endpoint}{_INTAKE_PATH_TEMPLATE.format(workspace=workspace)}"
        self._posted_count = 0

        span_to_entry = SpanToIntakeEntryProcessor(
            app=app,
            task=task,
            default_model=default_model,
            project=project,
            **({"event_types": event_types} if event_types is not None else {}),
        )
        logger.info(
            "intake exporter ready: url=%s app=%s task=%s filter=%s batch=%d flush=%.1fs",
            self._url, app, task,
            sorted(span_to_entry._event_types),
            batch_size, flush_interval,
        )
        self._span_to_entry = span_to_entry
        self.add_processor(span_to_entry)
        self.add_processor(IntakeEntryToDictProcessor())
        self.add_processor(
            _DictBatchingProcessor(
                batch_size=batch_size,
                flush_interval=flush_interval,
                max_queue_size=max_queue_size,
                drop_on_overflow=drop_on_overflow,
                shutdown_timeout=shutdown_timeout,
            ))
        self.add_processor(DictBatchFilterProcessor())
        self._events_seen = 0

    @override
    def export(self, event: IntermediateStep) -> None:
        """Trace every event that reaches the exporter.

        Span creation only happens inside :meth:`SpanExporter.export` on the END
        event, so events that never arrive here explain a silent ``matched=0``.
        """
        if isinstance(event, IntermediateStep):
            self._events_seen += 1
            event_type = event.payload.event_type.value if event.payload and event.payload.event_type else "?"
            state = event.event_state.value if event.event_state else "?"
            # CHUNK / LLM_NEW_TOKEN fire once per streamed token — thousands
            # per agent run. Log those at DEBUG so `-v debug` can still dump
            # them, but keep START/END at INFO for span-boundary signal.
            log = logger.debug if state == "CHUNK" else logger.info
            log(
                "intake: event seen state=%s event_type=%s (total_events=%d)",
                state,
                event_type,
                self._events_seen,
            )
        super().export(event)

    @override
    async def export_processed(self, item: dict | list[dict]) -> None:
        batch = item if isinstance(item, list) else [item]
        if not batch:
            return
        for entry in batch:
            if not entry:
                continue
            try:
                response = await self._client.post(self._url, json=entry)
                response.raise_for_status()
                self._posted_count += 1
                logger.info(
                    "intake POST ok: status=%d external_id=%s total=%d",
                    response.status_code, entry.get("external_id", "?"), self._posted_count,
                )
            except httpx.HTTPStatusError as exc:
                # 422 = schema violation; retrying won't help, so log + drop.
                status = exc.response.status_code if exc.response is not None else "?"
                body = exc.response.text[:500] if exc.response is not None else ""
                if status == 422:
                    logger.error("intake rejected entry (422): %s", body)
                else:
                    logger.exception("intake POST failed (status=%s): %s", status, body)
            except Exception:
                logger.exception("intake POST failed (url=%s)", self._url)

    async def _cleanup(self) -> None:
        logger.info("intake exporter shutting down after %d POSTs to %s", self._posted_count, self._url)
        try:
            await self._client.aclose()
        finally:
            await super()._cleanup()
