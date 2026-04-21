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

import json

import pytest

from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.plugins.intake.observability.processor.intake_entry_processor import IntakeEntryToDictProcessor
from nat.plugins.intake.observability.processor.intake_entry_processor import SpanToIntakeEntryProcessor
from nat.plugins.intake.observability.schema.entry import EntryInput


def _llm_span(*, input_value: object, output_value: object, name: str = "llm-call") -> Span:
    """Build a Span carrying the attributes that NAT's SpanExporter would set on LLM_END."""
    return Span(
        name=name,
        context=SpanContext(),
        attributes={
            "nat.event_type": IntermediateStepType.LLM_END.value,
            "input.value": input_value if isinstance(input_value, str) else json.dumps(input_value),
            "output.value": output_value if isinstance(output_value, str) else json.dumps(output_value),
        },
    )


@pytest.fixture
def processor() -> SpanToIntakeEntryProcessor:
    return SpanToIntakeEntryProcessor(
        app="default/coding-agent",
        task="code-generation",
        default_model="nemotron-super-49b",
    )


@pytest.mark.asyncio
async def test_llm_end_with_openai_shaped_payloads(processor: SpanToIntakeEntryProcessor) -> None:
    span = _llm_span(
        input_value={"messages": [{"role": "user", "content": "hi"}]},
        output_value={"choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}]},
    )

    entry = await processor.process(span)

    assert isinstance(entry, EntryInput)
    assert entry.data.request.messages[0].role == "user"
    assert entry.data.request.messages[0].content == "hi"
    assert entry.data.response.choices[0].message.role == "assistant"
    assert entry.data.response.choices[0].message.content == "hello"
    assert entry.context.app == "default/coding-agent"
    assert entry.context.task == "code-generation"


@pytest.mark.asyncio
async def test_llm_end_with_unstructured_strings_wraps_into_messages(processor: SpanToIntakeEntryProcessor) -> None:
    span = _llm_span(input_value="just a string", output_value="just a reply")

    entry = await processor.process(span)

    assert entry is not None
    assert len(entry.data.request.messages) == 1
    assert entry.data.request.messages[0].role == "user"
    assert entry.data.request.messages[0].content == "just a string"
    assert entry.data.response.choices[0].message.content == "just a reply"


@pytest.mark.asyncio
async def test_workflow_end_is_dropped_by_default(processor: SpanToIntakeEntryProcessor) -> None:
    span = Span(
        name="agent-turn",
        context=SpanContext(),
        attributes={
            "nat.event_type": IntermediateStepType.WORKFLOW_END.value,
            "input.value": "hi",
            "output.value": "ok",
        },
    )

    assert await processor.process(span) is None


@pytest.mark.asyncio
async def test_workflow_end_published_when_event_types_includes_it() -> None:
    processor = SpanToIntakeEntryProcessor(
        app="default/agent",
        task="general",
        default_model="m",
        event_types=frozenset({IntermediateStepType.WORKFLOW_END.value}),
    )
    span = Span(
        name="agent-turn",
        context=SpanContext(),
        attributes={
            "nat.event_type": IntermediateStepType.WORKFLOW_END.value,
            "input.value": "hi",
            "output.value": "ok",
        },
    )

    entry = await processor.process(span)
    assert entry is not None
    assert entry.data.request.messages[0].content == "hi"


@pytest.mark.asyncio
async def test_external_id_uses_span_id_when_present(processor: SpanToIntakeEntryProcessor) -> None:
    span = _llm_span(input_value="hi", output_value="ok")
    entry = await processor.process(span)
    assert entry is not None
    assert entry.external_id == f"{span.context.span_id:032x}"


@pytest.mark.asyncio
async def test_value_obj_attribute_takes_precedence_over_value() -> None:
    processor = SpanToIntakeEntryProcessor(app="a/b", task="t", default_model="m")
    span = Span(
        name="llm",
        context=SpanContext(),
        attributes={
            "nat.event_type": IntermediateStepType.LLM_END.value,
            "input.value": "stale",
            "input.value_obj": json.dumps({"messages": [{"role": "user", "content": "fresh"}]}),
            "output.value_obj": json.dumps(
                {"choices": [{"index": 0, "message": {"role": "assistant", "content": "fresh-reply"}}]}),
        },
    )

    entry = await processor.process(span)
    assert entry is not None
    assert entry.data.request.messages[0].content == "fresh"
    assert entry.data.response.choices[0].message.content == "fresh-reply"


@pytest.mark.asyncio
async def test_dict_processor_returns_empty_for_none() -> None:
    processor = IntakeEntryToDictProcessor()
    assert await processor.process(None) == {}


@pytest.mark.asyncio
async def test_dict_processor_round_trips_entry(processor: SpanToIntakeEntryProcessor) -> None:
    dict_processor = IntakeEntryToDictProcessor()
    span = _llm_span(input_value="hi", output_value="ok")
    entry = await processor.process(span)
    assert entry is not None

    payload = await dict_processor.process(entry)

    assert payload["data"]["request"]["model"] == "llm-call"
    assert payload["data"]["request"]["messages"][0]["role"] == "user"
    assert payload["context"]["app"] == "default/coding-agent"
    # exclude_none keeps unset optional fields out of the wire body
    assert "project" not in payload
    assert "user_id" not in payload["context"]
