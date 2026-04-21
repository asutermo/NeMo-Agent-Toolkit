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

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.common import OptionalSecretStr
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin

logger = logging.getLogger(__name__)


class IntakeTelemetryExporter(BatchConfigMixin, TelemetryExporterBaseConfig, name="intake"):
    """Telemetry exporter that publishes normalized LLM interactions to NeMo Intake."""

    endpoint: str = Field(
        description="Base URL of the NeMo platform hosting Intake (without the "
        "'/apis/intake/v2' path), e.g. 'https://nmp.example.com'.")
    workspace: str = Field(
        default="default",
        description="Intake workspace; becomes the URL path segment.")
    app: str = Field(description="App reference in 'workspace/name' form. Auto-created on first entry.")
    task: str = Field(description="Task name within the app. Auto-created on first entry.")
    project: str | None = Field(
        default=None,
        description="Optional project name for org-scoped filtering in Intake exports.")
    default_model: str = Field(
        default="unknown-model",
        description="Fallback model name when the span doesn't carry one.")
    event_types: list[str] = Field(
        default_factory=lambda: [IntermediateStepType.LLM_END.value],
        description="Span ``nat.event_type`` values to publish. Add 'WORKFLOW_END' for "
        "agent types that don't emit LLM_END events.")
    api_key: OptionalSecretStr = Field(
        default=None,
        description="Bearer token for the Intake API. When unset, requests go out unauthenticated "
        "(suitable for platform-injected service-to-service auth).")
    timeout: float = Field(default=10.0, description="HTTP request timeout in seconds.")


@register_telemetry_exporter(config_type=IntakeTelemetryExporter)
async def intake_telemetry_exporter(config: IntakeTelemetryExporter, builder: Builder):
    """Build an Intake telemetry exporter."""
    del builder
    # pylint: disable=import-outside-toplevel
    import httpx

    from nat.plugins.intake.observability.exporter.intake_exporter import IntakeExporter

    headers: dict[str, str] = {}
    if config.api_key is not None:
        secret = config.api_key.get_secret_value()
        if secret:
            headers["Authorization"] = f"Bearer {secret}"

    client = httpx.AsyncClient(timeout=config.timeout, headers=headers)
    yield IntakeExporter(
        endpoint=config.endpoint,
        workspace=config.workspace,
        app=config.app,
        task=config.task,
        client=client,
        default_model=config.default_model,
        project=config.project,
        event_types=frozenset(config.event_types),
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )
