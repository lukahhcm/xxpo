"""
OpenAI/vLLM-native LLM utilities for text agents.

This module intentionally avoids LiteLLM and calls OpenAI-compatible
Chat Completions endpoints directly (e.g., local vLLM /v1).
"""

import json
import os
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from openai import OpenAI

from tau2.config import DEFAULT_MAX_RETRIES
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ParticipantMessageBase,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# Context variable to store the directory where LLM debug logs should be written
llm_log_dir: ContextVar[Optional[Path]] = ContextVar("llm_log_dir", default=None)

# Context variable to store the LLM logging mode ("all" or "latest")
llm_log_mode: ContextVar[str] = ContextVar("llm_log_mode", default="latest")


def set_llm_log_dir(log_dir: Optional[Path | str]) -> None:
    """
    Set the directory where LLM debug logs should be written.

    Args:
        log_dir: Path to the directory where logs should be saved, or None to disable file logging
    """
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    llm_log_dir.set(log_dir)


def set_llm_log_mode(mode: str) -> None:
    """
    Set the LLM debug logging mode.

    Args:
        mode: Logging mode - "all" to save every LLM call, "latest" to keep only the most recent call of each type
    """
    if mode not in ("all", "latest"):
        raise ValueError(f"Invalid LLM log mode: {mode}. Must be 'all' or 'latest'")
    llm_log_mode.set(mode)


def _normalize_model_name(model: str) -> str:
    """
    Normalize provider-prefixed model names.

    For backwards compatibility with existing tau2 CLI usage, `openai/<id>`
    is mapped to `<id>` before sending to the OpenAI-compatible endpoint.
    """
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def _format_messages_for_logging(messages: list[dict]) -> list[dict]:
    """
    Format messages for debug logging by splitting content on newlines.
    """
    formatted = []
    for msg in messages:
        msg_copy = msg.copy()
        if "content" in msg_copy and isinstance(msg_copy["content"], str):
            content_lines = msg_copy["content"].split("\n")
            if len(content_lines) > 1:
                msg_copy["content"] = content_lines
        formatted.append(msg_copy)
    return formatted


def _write_llm_log(
    request_data: dict, response_data: dict, call_name: Optional[str] = None
) -> None:
    """
    Write LLM call log to file if a log directory is set.
    """
    log_dir = llm_log_dir.get()
    if log_dir is None:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    current_log_mode = llm_log_mode.get()

    if current_log_mode == "latest" and call_name:
        pattern = f"*_{call_name}_*.json"
        existing_files = list(log_dir.glob(pattern))
        for existing_file in existing_files:
            try:
                existing_file.unlink()
            except FileNotFoundError:
                pass

    call_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    if call_name:
        log_file = log_dir / f"{timestamp}_{call_name}_{call_id}.json"
    else:
        log_file = log_dir / f"{timestamp}_{call_id}.json"

    call_data = {
        "call_id": call_id,
        "call_name": call_name,
        "timestamp": datetime.now().isoformat(),
        "request": request_data,
        "response": response_data,
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(call_data, f, indent=2)


def validate_message(message: Message) -> None:
    """
    Validate a Tau2 message before LLM call.
    """

    def has_text_content(m: Message) -> bool:
        return m.content is not None and bool(m.content.strip())

    def has_content_or_tool_calls(m: ParticipantMessageBase) -> bool:
        return m.has_content() or m.is_tool_call()

    if isinstance(message, SystemMessage):
        assert has_text_content(message), (
            f"System message must have content. got {message}"
        )
    if isinstance(message, ParticipantMessageBase):
        assert has_content_or_tool_calls(message), (
            f"Message must have content or tool calls. got {message}"
        )


def validate_message_history(messages: list[Message]) -> None:
    for message in messages:
        validate_message(message)


def to_openai_messages(messages: list[Message]) -> list[dict]:
    """
    Convert Tau2 messages to OpenAI Chat Completions message format.
    """
    openai_messages: list[dict] = []
    for message in messages:
        if isinstance(message, UserMessage):
            openai_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            openai_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            openai_messages.append({"role": "system", "content": message.content})
    return openai_messages


def _response_usage_to_dict(response: Any) -> Optional[dict]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
    }


def _default_function_parameters_schema() -> dict:
    return {"type": "object", "properties": {}, "required": []}


def _normalize_function_parameters_schema(parameters: Any, tool_name: str) -> dict:
    """
    Normalize tool function parameters to a strict JSON object schema.

    Some OpenAI-compatible backends are strict about this field and may reject
    empty strings or non-object values.
    """
    if parameters is None or parameters == "":
        logger.warning(
            "Tool '{}' has empty parameters schema; replacing with empty object schema.",
            tool_name,
        )
        return _default_function_parameters_schema()

    if isinstance(parameters, str):
        stripped = parameters.strip()
        if not stripped:
            logger.warning(
                "Tool '{}' has blank parameters schema string; replacing with empty object schema.",
                tool_name,
            )
            return _default_function_parameters_schema()
        try:
            parameters = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning(
                "Tool '{}' has invalid JSON parameters schema string; replacing with empty object schema.",
                tool_name,
            )
            return _default_function_parameters_schema()

    if not isinstance(parameters, dict):
        logger.warning(
            "Tool '{}' has non-dict parameters schema (type={}); replacing with empty object schema.",
            tool_name,
            type(parameters).__name__,
        )
        return _default_function_parameters_schema()

    normalized = dict(parameters)
    normalized["type"] = "object"

    props = normalized.get("properties")
    if not isinstance(props, dict):
        normalized["properties"] = {}

    required = normalized.get("required")
    if required is None:
        normalized["required"] = []
    elif isinstance(required, list):
        normalized["required"] = [x for x in required if isinstance(x, str)]
    elif isinstance(required, str):
        normalized["required"] = [required] if required else []
    else:
        normalized["required"] = []

    return normalized


def _sanitize_tools_schema(tools_schema: Optional[list[dict]]) -> Optional[list[dict]]:
    """
    Sanitize tool schema list for stricter OpenAI-compatible backends.
    """
    if not tools_schema:
        return None

    sanitized: list[dict] = []
    for i, tool_schema in enumerate(tools_schema):
        if not isinstance(tool_schema, dict):
            logger.warning(
                "Skipping non-dict tool schema at index {} (type={}).",
                i,
                type(tool_schema).__name__,
            )
            continue

        fn = tool_schema.get("function")
        if not isinstance(fn, dict):
            logger.warning(
                "Skipping tool schema at index {} with missing function object.", i
            )
            continue

        name = fn.get("name")
        if not isinstance(name, str) or not name:
            logger.warning(
                "Skipping tool schema at index {} with invalid function name.", i
            )
            continue

        description = fn.get("description")
        if not isinstance(description, str) or not description:
            description = name

        parameters = _normalize_function_parameters_schema(
            fn.get("parameters"), tool_name=name
        )

        sanitized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    return sanitized or None


def _minimal_tools_schema(tools_schema: Optional[list[dict]]) -> Optional[list[dict]]:
    """
    Build a minimal fallback tool schema for stricter backends.
    """
    if not tools_schema:
        return None

    minimal: list[dict] = []
    for tool_schema in tools_schema:
        fn = tool_schema.get("function", {})
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        desc = fn.get("description")
        if not isinstance(desc, str) or not desc:
            desc = name
        minimal.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": _default_function_parameters_schema(),
                },
            }
        )
    return minimal or None


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    call_name: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response via OpenAI-compatible Chat Completions API.

    This path does not use LiteLLM.
    """
    validate_message_history(messages)

    request_kwargs = dict(kwargs)
    if request_kwargs.get("num_retries") is None:
        request_kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    openai_messages = to_openai_messages(messages)
    tools_schema = [tool.openai_schema for tool in tools] if tools else None
    tools_schema = _sanitize_tools_schema(tools_schema)

    if tools_schema and tool_choice is None:
        tool_choice = "auto"

    formatted_messages = _format_messages_for_logging(openai_messages)
    request_data = {
        "model": model,
        "messages": formatted_messages,
        "tools": tools_schema,
        "tool_choice": tool_choice,
        "kwargs": {
            k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
            for k, v in request_kwargs.items()
        },
    }
    request_timestamp = datetime.now().isoformat()

    api_base = request_kwargs.pop("api_base", None)
    base_url = request_kwargs.pop("base_url", None) or api_base
    api_key = request_kwargs.pop("api_key", None) or os.getenv("OPENAI_API_KEY") or "EMPTY"
    num_retries = request_kwargs.pop("num_retries", DEFAULT_MAX_RETRIES)
    timeout = request_kwargs.pop("timeout", None)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=num_retries,
        timeout=timeout,
    )

    request_payload: dict[str, Any] = {
        "model": _normalize_model_name(model),
        "messages": openai_messages,
    }
    if tools_schema is not None:
        request_payload["tools"] = tools_schema
    if tool_choice is not None:
        request_payload["tool_choice"] = tool_choice
    request_payload.update(request_kwargs)

    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(**request_payload)
    except Exception as e:
        err_text = str(e)
        should_retry_with_minimal_tools = (
            request_payload.get("tools") is not None
            and ("json_invalid" in err_text or "Invalid JSON" in err_text)
        )

        if should_retry_with_minimal_tools:
            minimal_tools = _minimal_tools_schema(request_payload.get("tools"))
            if minimal_tools:
                logger.warning(
                    "Backend rejected tool schema as invalid JSON; retrying with minimal tool schemas."
                )
                request_payload["tools"] = minimal_tools
                request_data["tools"] = minimal_tools
                try:
                    response = client.chat.completions.create(**request_payload)
                except Exception as e2:
                    logger.error(e2)
                    raise e2
            else:
                logger.error(e)
                raise e
        else:
            logger.error(e)
            raise e

    generation_time_seconds = time.perf_counter() - start_time

    usage = _response_usage_to_dict(response)
    cost = 0.0  # No built-in pricing layer in direct OpenAI/vLLM mode.

    response_choice = response.choices[0]
    finish_reason = response_choice.finish_reason
    if finish_reason == "length":
        logger.warning("Output might be incomplete due to token limit!")

    assistant = response_choice.message
    content = assistant.content
    raw_tool_calls = assistant.tool_calls or []

    tool_calls: list[ToolCall] = []
    for tool_call in raw_tool_calls:
        arguments_raw = tool_call.function.arguments
        if arguments_raw is None or arguments_raw == "":
            arguments_raw = "{}"
        elif not isinstance(arguments_raw, str):
            arguments_raw = json.dumps(arguments_raw)

        try:
            arguments = json.loads(arguments_raw)
        except json.JSONDecodeError:
            logger.warning(
                "Tool call arguments are not valid JSON; preserving raw string. tool={} args={}",
                tool_call.function.name,
                arguments_raw,
            )
            arguments = {"_raw_arguments": arguments_raw}

        tool_calls.append(
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=arguments,
            )
        )

    assistant_message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
        cost=cost,
        usage=usage,
        raw_data=response.model_dump(),
        generation_time_seconds=generation_time_seconds,
    )

    response_data = {
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
        "cost": cost,
        "usage": usage,
        "generation_time_seconds": generation_time_seconds,
    }
    request_data["timestamp"] = request_timestamp
    _write_llm_log(request_data, response_data, call_name=call_name)

    return assistant_message
