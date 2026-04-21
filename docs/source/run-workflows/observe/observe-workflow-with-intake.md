<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Observing a Workflow with NeMo Intake

This guide enables observability in a NeMo Agent Toolkit workflow that publishes normalized LLM interactions to **NeMo Intake**, the public data-producer surface for the NeMo platform. Each emitted entry is a normalized `(request, response, context)` tuple that downstream NeMo consumers — Datastore, Evaluator, Customizer, HITL — can format for training and evaluation.

By the end of this guide you will have:
- Configured the `intake` tracing exporter in your workflow.
- Published one Intake entry per agent LLM turn.
- Understood how to scope entries with `app`, `task`, `project`, and `thread_id`.

## Step 1: Prerequisites

- A reachable NeMo Intake endpoint (either a standalone deployment or the Intake service on an NMP cluster).
- A bearer token for the Intake API, unless the platform injects service-to-service credentials for you.

## Step 2: Install the Intake Plugin

::::{tab-set}
:sync-group: install-tool

:::{tab-item} source
:selected:
:sync: source

```bash
uv pip install -e ".[intake]"
```

:::

:::{tab-item} package
:sync: package

```bash
uv pip install "nvidia-nat[intake]"
```

:::

::::

## Step 3: Modify Workflow Configuration

Add the `intake` exporter to the `general.telemetry.tracing` block:

```yaml
general:
  telemetry:
    tracing:
      intake:
        _type: intake
        endpoint: ${INTAKE_URL}              # e.g. https://nmp.example.com
        workspace: default
        app: default/coding-agent            # "workspace/name" form, required
        task: code-generation
        api_key: ${INTAKE_API_KEY}           # optional; omit for platform-injected auth
        project: coding                      # optional; for org-scoped filters
        batch_size: 100
        flush_interval: 5.0
```

Entries land at `POST {endpoint}/apis/intake/v2/workspaces/{workspace}/entries`. The `app` and `task` are auto-created on first reference.

## Configuration Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `endpoint` | Base URL of the NeMo platform hosting Intake (without the `/apis/intake/v2` path). | Yes | — |
| `workspace` | Intake workspace; becomes the URL path segment. | No | `default` |
| `app` | App reference in `workspace/name` form. Auto-created on first entry. | Yes | — |
| `task` | Task name within the app. Auto-created on first entry. | Yes | — |
| `project` | Project name for org-scoped filtering in Intake exports. | No | `None` |
| `default_model` | Fallback model name when the span doesn't carry one. | No | `unknown-model` |
| `event_types` | Span `nat.event_type` values to publish. NAT stamps `nat.event_type` at START and retains it through export (the span has both input and output by then), so the default matches `LLM_START`. Widen to include `WORKFLOW_START` for agents that don't produce an LLM span. | No | `["LLM_START"]` |
| `api_key` | Bearer token for the Intake API. Unset for platform-injected auth. | No | `None` |
| `timeout` | HTTP request timeout in seconds. | No | `10.0` |
| `batch_size` | Entries per HTTP flush. | No | `100` |
| `flush_interval` | Seconds between flushes. | No | `5.0` |
| `max_queue_size` | Max queued entries before backpressure. | No | `1000` |
| `drop_on_overflow` | Drop entries when the queue is full (vs. block). | No | `False` |
| `shutdown_timeout` | Seconds to wait for final flush on shutdown. | No | `10.0` |

## Step 4: Run Your Workflow

```bash
nat run --config_file config-intake.yml --input "Your workflow input here"
```

As the workflow runs, one Intake entry is POSTed per LLM span (each exported span carries `nat.event_type=LLM_START` plus the merged output). Tool and workflow lifecycle spans are dropped so the dataset stays formatted for training and evaluation.

## How It Works

The `intake` exporter assembles a four-stage processor chain:

1. **`SpanToIntakeEntryProcessor`** — filters spans by `event_types` and converts the `input.value`/`output.value` attributes into a Pydantic `EntryInput`.
2. **`IntakeEntryToDictProcessor`** — serializes to a JSON-safe dict (unset optional fields are omitted).
3. **`BatchingProcessor[dict]`** — batches by `batch_size` / `flush_interval` with backpressure governed by `max_queue_size` + `drop_on_overflow`.
4. **`DictBatchFilterProcessor`** — drops empty entries that survived filtering.

The emitted body matches the Intake v2 producer schema:

```json
{
  "external_id": "<run_id>:<seq>",
  "project": "<optional>",
  "data": {
    "request":  {"model": "...", "messages": [{"role": "user", "content": "..."}]},
    "response": {"choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}}]}
  },
  "context": {
    "app": "default/coding-agent",
    "task": "code-generation",
    "thread_id": "<session.id>",
    "trace_id": "<OTel trace_id>"
  }
}
```

## Working With Non-LLM Agent Types

Some agent workflows don't route through an LLM span (for example some custom tool-only agents). Widen the filter to capture those workflow turns too:

```yaml
intake:
  _type: intake
  event_types: ["LLM_START", "WORKFLOW_START"]
  # ...
```

## Resources

- [NeMo Intake service repository](https://github.com/NVIDIA/NeMo-Intake)
- [NeMo Agent Toolkit Observability Guide](./observe.md)
