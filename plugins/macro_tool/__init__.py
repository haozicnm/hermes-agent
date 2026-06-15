"""
MacroTool Plugin — Reflection-before-action pattern for Hermes Agent.

Wraps all tool schemas into a single "AgentOutput" tool that requires
the LLM to output reflection (evaluation, memory, next_goal) before
choosing an action. Inspired by alibaba/page-agent's MacroTool design.

Configuration (config.yaml):
  plugins:
    entries:
      macro_tool:
        enabled: true
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── State ──────────────────────────────────────────────────────────
_original_get_tool_definitions = None
_original_handle_function_call = None
_original_tool_names: set = set()
_enabled = True
_pending_reflection: Dict[str, Dict[str, str]] = {}  # task_id → reflection dict


# ── Schema Builder ─────────────────────────────────────────────────

def _build_macro_tool_schema(original_tools: List[Dict]) -> Dict:
    """Wrap all original tools into a single MacroTool (AgentOutput) schema.

    The action field is a JSON object where exactly one property must be
    set — the tool name — and its value is that tool's parameters.
    """
    action_properties: Dict[str, Any] = {}
    action_required: List[str] = []

    for tool in original_tools:
        func = tool.get("function", tool)
        name = func["name"]
        params = func.get("parameters", {"type": "object", "properties": {}})

        action_properties[name] = {
            "type": "object",
            "description": func.get("description", f"Call the {name} tool"),
            "properties": params.get("properties", {}),
            "additionalProperties": False,
        }
        # Include required fields from original schema
        if params.get("required"):
            action_properties[name]["required"] = params["required"]

    schema = {
        "type": "function",
        "function": {
            "name": "AgentOutput",
            "description": (
                "You MUST call this tool for EVERY action. "
                "First reflect on your previous step (evaluation_previous_goal), "
                "note key information to remember (memory), "
                "state your specific next goal (next_goal), "
                "then choose ONE action to execute."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "evaluation_previous_goal": {
                        "type": "string",
                        "description": (
                            "Honest assessment of the last action: "
                            "what worked, what failed, what was unexpected. "
                            "Use ✅ for success, ❌ for failure."
                        ),
                    },
                    "memory": {
                        "type": "string",
                        "description": (
                            "Key facts, file paths, findings, or context "
                            "to carry forward. Keep it concise but specific."
                        ),
                    },
                    "next_goal": {
                        "type": "string",
                        "description": (
                            "Specific, actionable goal for this step. "
                            "Be precise: 'Read lines 200-300 of config.rs' "
                            "not 'continue working'."
                        ),
                    },
                    "action": {
                        "type": "object",
                        "description": (
                            "Choose ONE tool to call. The property name is "
                            "the tool name, the value is the tool's parameters."
                        ),
                        "properties": action_properties,
                        "additionalProperties": False,
                        "minProperties": 1,
                        "maxProperties": 1,
                    },
                },
                "required": ["action"],
            },
        },
    }

    return schema


# ── Tool Wrapping ──────────────────────────────────────────────────

def _wrapped_get_tool_definitions(
    enabled_toolsets=None,
    disabled_toolsets=None,
    quiet_mode=False,
    skip_tool_search_assembly=False,
    **kwargs,
) -> List[Dict]:
    """Replacement for get_tool_definitions that wraps all tools into MacroTool."""
    global _original_get_tool_definitions, _original_tool_names, _enabled

    _orig = _original_get_tool_definitions
    if not _enabled or _orig is None:
        return _orig(  # type: ignore[misc]
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=quiet_mode,
            skip_tool_search_assembly=skip_tool_search_assembly,
            **kwargs,
        )

    # Internal bridge calls need the real catalog — don't wrap
    if skip_tool_search_assembly:
        return _orig(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=quiet_mode,
            skip_tool_search_assembly=True,
            **kwargs,
        )

    original_tools = _orig(
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        quiet_mode=True,  # silence duplicate logging
        skip_tool_search_assembly=skip_tool_search_assembly,
        **kwargs,
    )

    if not original_tools:
        return []

    _original_tool_names = {
        t.get("function", t)["name"] for t in original_tools
    }

    macro_tool = _build_macro_tool_schema(original_tools)

    if not quiet_mode:
        print(f"🧠 MacroTool: wrapped {len(original_tools)} tools → 1 AgentOutput")

    return [macro_tool]


# ── Tool Dispatch ──────────────────────────────────────────────────

def _dispatch_agent_output(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: str = "",
    **kwargs,
) -> str:
    """Handle AgentOutput calls: extract reflection, dispatch to real tool."""
    global _original_handle_function_call, _original_tool_names, _pending_reflection

    action = function_args.get("action", {})
    if not action:
        return json.dumps({"error": "No action specified in AgentOutput"})

    # Extract the single action (tool_name → tool_args)
    tool_names = list(action.keys())
    if len(tool_names) != 1:
        return json.dumps({"error": f"Expected exactly 1 action, got {len(tool_names)}: {tool_names}"})

    real_tool_name = tool_names[0]
    real_tool_args = action[real_tool_name] or {}

    # Store reflection for logging/display
    reflection: Dict[str, str] = {
        "evaluation_previous_goal": str(function_args.get("evaluation_previous_goal", "")),
        "memory": str(function_args.get("memory", "")),
        "next_goal": str(function_args.get("next_goal", "")),
    }
    if task_id:
        _pending_reflection[task_id] = reflection

    # Log the reflection
    eval_text = reflection["evaluation_previous_goal"]
    mem_text = reflection["memory"]
    goal_text = reflection["next_goal"]
    parts = []
    if eval_text:
        parts.append(f"✅ {eval_text}")
    if mem_text:
        parts.append(f"💾 {mem_text}")
    if goal_text:
        parts.append(f"🎯 {goal_text}")
    if parts:
        logger.info("MacroTool: %s → %s", " | ".join(parts), real_tool_name)

    # Dispatch to the real tool
    try:
        result = _original_handle_function_call(
            real_tool_name,
            real_tool_args,
            task_id=task_id,
            **kwargs,
        )

        # Append reflection prefix to result so the LLM sees it in history
        reflection_prefix = " | ".join(parts) if parts else ""
        if reflection_prefix and isinstance(result, str):
            result = f"[{reflection_prefix}]\n{result}"

        return result
    except Exception as e:
        logger.error("MacroTool dispatch error: %s → %s: %s", real_tool_name, e)
        return json.dumps({"error": f"Tool '{real_tool_name}' failed: {str(e)}"})


def _wrapped_handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: str = "",
    **kwargs,
) -> str:
    """Replacement for handle_function_call that intercepts AgentOutput."""
    global _enabled, _original_handle_function_call

    _orig = _original_handle_function_call
    if _enabled and function_name == "AgentOutput":
        return _dispatch_agent_output(
            function_name,
            function_args,
            task_id=task_id,
            **kwargs,
        )

    if _orig is None:
        return json.dumps({"error": "handle_function_call not initialized"})

    return _orig(
        function_name,
        function_args,
        task_id=task_id,
        **kwargs,
    )


# ── Plugin Entry ───────────────────────────────────────────────────

def register(ctx) -> None:
    """Register the MacroTool plugin.

    Monkey-patches get_tool_definitions and handle_function_call to
    implement the reflection-before-action pattern.
    """
    global _original_get_tool_definitions, _original_handle_function_call, _enabled

    try:
        from model_tools import (
            get_tool_definitions as _get_tool_definitions,
            handle_function_call as _handle_function_call,
        )

        _original_get_tool_definitions = _get_tool_definitions
        _original_handle_function_call = _handle_function_call

        # Replace with wrapped versions
        import model_tools
        model_tools.get_tool_definitions = _wrapped_get_tool_definitions
        model_tools.handle_function_call = _wrapped_handle_function_call

        # Also patch the reference in run_agent
        import run_agent
        if hasattr(run_agent, "get_tool_definitions"):
            run_agent.get_tool_definitions = _wrapped_get_tool_definitions
        if hasattr(run_agent, "handle_function_call"):
            run_agent.handle_function_call = _wrapped_handle_function_call

        logger.info("MacroTool plugin registered — wrapping tools into AgentOutput")

    except Exception as e:
        logger.error("MacroTool plugin failed to register: %s", e)
        _enabled = False


def unregister() -> None:
    """Restore original functions (for plugin unload)."""
    global _original_get_tool_definitions, _original_handle_function_call, _enabled

    _enabled = False

    if _original_get_tool_definitions:
        try:
            import model_tools
            model_tools.get_tool_definitions = _original_get_tool_definitions
        except Exception:
            pass

    if _original_handle_function_call:
        try:
            import model_tools
            model_tools.handle_function_call = _original_handle_function_call
        except Exception:
            pass
