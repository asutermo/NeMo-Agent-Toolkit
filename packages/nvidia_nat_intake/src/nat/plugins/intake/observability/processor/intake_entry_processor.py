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
"""Span → ``EntryInput`` and ``EntryInput`` → ``dict`` processors.

The Intake producer contract is one entry per LLM interaction. Spans for tool
or workflow lifecycle events are dropped by default (they pollute training
datasets with non-LLM rows). Workflow events can be enabled via the exporter
config when an agent type doesn't emit ``LLM_END`` (e.g. early ``react_agent``).
"""

import json
import logging
from typing import Any

from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.span import Span
from nat.observability.processor.processor import Processor
from nat.utils.type_utils import override

from nat.plugins.intake.observability.schema.entry import EntryContext
from nat.plugins.intake.observability.schema.entry import EntryData
from nat.plugins.intake.observability.schema.entry import EntryInput
from nat.plugins.intake.observability.schema.entry import FlexibleChoice
from nat.plugins.intake.observability.schema.entry import FlexibleEntryRequest
from nat.plugins.intake.observability.schema.entry import FlexibleEntryResponse
from nat.plugins.intake.observability.schema.entry import FlexibleMessage

logger = logging.getLogger(__name__)


def _decode_value(raw: Any) -> Any:
    """Span attributes carry JSON-as-string when mime_type is application/json.

    Try parsing; fall through to the raw value if it isn't JSON.
    """
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _coerce_messages(raw: Any) -> list[FlexibleMessage]:
    """Coerce an arbitrary input payload into a non-empty messages list."""
    decoded = _decode_value(raw)
    if isinstance(decoded, list) and decoded and all(isinstance(m, dict) and "role" in m for m in decoded):
        return [FlexibleMessage(**m) for m in decoded]
    if isinstance(decoded, dict) and isinstance(decoded.get("messages"), list):
        return _coerce_messages(decoded["messages"])
    content = decoded if isinstance(decoded, str) else json.dumps(decoded) if decoded is not None else ""
    return [FlexibleMessage(role="user", content=content)]


def _coerce_choices(raw: Any) -> list[FlexibleChoice]:
    """Coerce an arbitrary output payload into a non-empty choices list."""
    decoded = _decode_value(raw)
    if isinstance(decoded, dict) and isinstance(decoded.get("choices"), list) and decoded["choices"]:
        return [FlexibleChoice(**c) for c in decoded["choices"]]
    if isinstance(decoded, list) and decoded and all(isinstance(c, dict) and "message" in c for c in decoded):
        return [FlexibleChoice(**c) for c in decoded]
    if isinstance(decoded, dict) and "role" in decoded:
        return [FlexibleChoice(index=0, message=FlexibleMessage(**decoded), finish_reason="stop")]
    content = decoded if isinstance(decoded, str) else json.dumps(decoded) if decoded is not None else ""
    return [FlexibleChoice(index=0, message=FlexibleMessage(role="assistant", content=content), finish_reason="stop")]


_DEFAULT_EVENT_TYPES: frozenset[str] = frozenset({IntermediateStepType.LLM_END.value})


class SpanToIntakeEntryProcessor(Processor[Span, EntryInput | None]):
    """Convert a NAT ``Span`` to an Intake ``EntryInput``.

    Returns ``None`` for spans whose ``nat.event_type`` is not in
    ``event_types`` — the downstream falsy-batch-filter drops them so they
    never leave the pipeline.
    """

    def __init__(
        self,
        *,
        app: str,
        task: str,
        default_model: str,
        project: str | None = None,
        event_types: frozenset[str] = _DEFAULT_EVENT_TYPES,
    ) -> None:
        self._app = app
        self._task = task
        self._default_model = default_model
        self._project = project
        self._event_types = event_types

    @override
    async def process(self, item: Span) -> EntryInput | None:
        attrs = item.attributes
        event_type = attrs.get("nat.event_type")
        if event_type not in self._event_types:
            return None

        # Prefer the *_obj attribute when present — it's the structured form.
        raw_input = attrs.get("input.value_obj") or attrs.get("input.value")
        raw_output = attrs.get("output.value_obj") or attrs.get("output.value")

        external_id = self._build_external_id(item)
        model_name = item.name or self._default_model

        try:
            entry = EntryInput(
                external_id=external_id,
                project=self._project,
                data=EntryData(
                    request=FlexibleEntryRequest(model=model_name, messages=_coerce_messages(raw_input)),
                    response=FlexibleEntryResponse(choices=_coerce_choices(raw_output)),
                ),
                context=EntryContext(
                    app=self._app,
                    task=self._task,
                    thread_id=attrs.get("session.id"),
                    trace_id=self._format_trace_id(item),
                ),
            )
        except Exception:  # pragma: no cover — schema mismatches surface in logs, drop the entry
            logger.exception("intake: failed to build EntryInput for span %s", item.name)
            return None

        return entry

    @staticmethod
    def _build_external_id(span: Span) -> str:
        """Idempotent ID: prefer the OTel span_id; fall back to span name + start_time."""
        if span.context is not None and span.context.span_id:
            return f"{span.context.span_id:032x}"
        return f"{span.name}:{span.start_time}"

    @staticmethod
    def _format_trace_id(span: Span) -> str | None:
        if span.context is None or not span.context.trace_id:
            return None
        return f"{span.context.trace_id:032x}"


class IntakeEntryToDictProcessor(Processor[EntryInput | None, dict]):
    """Serialize an ``EntryInput`` to a JSON-safe ``dict``.

    Returns ``{}`` for ``None`` inputs so the downstream
    ``DictBatchFilterProcessor`` can drop them.
    """

    @override
    async def process(self, item: EntryInput | None) -> dict:
        if item is None:
            return {}
        return json.loads(item.model_dump_json(by_alias=True, exclude_none=True))
