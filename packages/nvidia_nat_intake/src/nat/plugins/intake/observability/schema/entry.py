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
"""Vendored Pydantic models for the NeMo Intake v2 producer schema.

Mirrors ``POST /apis/intake/v2/workspaces/{workspace}/entries``. Kept local to
avoid a runtime dependency on the Intake service code; schema drift is accepted
and resolved by re-vendoring on Intake releases.
"""

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class FlexibleMessage(BaseModel):
    """One element of an OpenAI-shaped ``messages`` list.

    Only ``role`` is required by Intake; other fields pass through as-is.
    """

    model_config = ConfigDict(extra="allow")

    role: str = Field(description="Message role (e.g. 'user', 'assistant', 'system', 'tool').")
    content: Any | None = Field(default=None, description="Message content; string or structured payload.")


class FlexibleChoice(BaseModel):
    """One element of an OpenAI-shaped ``choices`` list."""

    model_config = ConfigDict(extra="allow")

    index: int = Field(default=0)
    message: FlexibleMessage = Field(description="The assistant message for this choice.")
    finish_reason: str | None = Field(default="stop")


class FlexibleEntryRequest(BaseModel):
    """Request half of an Intake entry."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(description="Model name; required by Intake.")
    messages: list[FlexibleMessage] = Field(
        description="At least one message; required by Intake.",
        min_length=1,
    )


class FlexibleEntryResponse(BaseModel):
    """Response half of an Intake entry."""

    model_config = ConfigDict(extra="allow")

    choices: list[FlexibleChoice] = Field(
        description="At least one choice; required by Intake.",
        min_length=1,
    )


class EntryData(BaseModel):
    """Required ``data`` block on an Intake entry."""

    request: FlexibleEntryRequest
    response: FlexibleEntryResponse


class EntryContext(BaseModel):
    """Required ``context`` block on an Intake entry.

    ``app`` MUST be in ``workspace/name`` form.
    """

    model_config = ConfigDict(extra="allow")

    app: str = Field(description="App reference in 'workspace/name' form.")
    task: str = Field(description="Task name within the app.")
    thread_id: str | None = Field(default=None, description="Groups multi-turn entries.")
    user_id: str | None = Field(default=None)
    trace_id: str | None = Field(default=None, description="W3C traceparent for cross-system joins.")
    session_id: str | None = Field(default=None)


class EntryInput(BaseModel):
    """Top-level body for ``POST /apis/intake/v2/workspaces/{ws}/entries``."""

    model_config = ConfigDict(extra="allow")

    external_id: str = Field(description="Idempotent client-supplied ID.")
    project: str | None = Field(default=None, description="Optional org-scoping.")
    data: EntryData
    context: EntryContext
