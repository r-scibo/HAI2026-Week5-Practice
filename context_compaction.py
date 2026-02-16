import json
import tiktoken

MODEL = "gpt-4o-mini"
MAX_CONTEXT_TOKENS = 128_000
COMPACTION_THRESHOLD = 100_000
KEEP_RECENT = 2  # messages to preserve at the end


def _get_field(msg, field, default=""):
    """Safely get a field from a dict or an OpenAI message object."""
    if isinstance(msg, dict):
        return msg.get(field, default)
    return getattr(msg, field, default)


def count_message_tokens(messages, encoding=None):
    """Count tokens in message content only (excludes tool definitions)."""
    if encoding is None:
        encoding = tiktoken.encoding_for_model(MODEL)
    total = 0
    for msg in messages:
        content = _get_field(msg, "content") or ""
        if isinstance(content, str):
            total += len(encoding.encode(content))
        # Count tool call arguments (assistant messages with tool_calls)
        tool_calls = _get_field(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                func = getattr(tc, "function", None) or tc.get("function", {})
                args = getattr(func, "arguments", None) or func.get("arguments", "")
                if args:
                    total += len(encoding.encode(args))
        total += 4  # overhead per message (role, separators)
    return total


def count_tool_tokens(tools, encoding=None):
    """Count tokens used by tool definitions (fixed cost)."""
    if not tools:
        return 0
    if encoding is None:
        encoding = tiktoken.encoding_for_model(MODEL)
    return len(encoding.encode(json.dumps(tools)))


def compact_if_needed(client, messages, tools=None):
    """Check message token count and compact via summarization if over threshold.
    Only message tokens are checked against the threshold (tool definitions are
    a fixed cost we cannot reduce).
    Returns (compacted: bool, message_tokens: int, tool_tokens: int)."""
    encoding = tiktoken.encoding_for_model(MODEL)
    msg_tokens = count_message_tokens(messages, encoding)
    tool_toks = count_tool_tokens(tools, encoding)

    if msg_tokens < COMPACTION_THRESHOLD:
        return False, msg_tokens, tool_toks

    # Need at least 1 message beyond system + user to summarize
    if len(messages) <= 2:
        return False, msg_tokens, tool_toks

    # Find cut point: keep at least KEEP_RECENT messages at the end,
    # but expand backwards to avoid splitting a tool_call/tool group.
    # A "tool" message must always follow its parent "assistant" message with tool_calls.
    cut = max(2, len(messages) - KEEP_RECENT)
    while cut > 2 and _get_field(messages[cut], "role") == "tool":
        cut -= 1  # include the assistant message that owns these tool results

    if cut <= 2:
        middle = messages[2:]
        preserved_end = []
    else:
        middle = messages[2:cut]
        preserved_end = messages[cut:]

    if not middle:
        return False, msg_tokens, tool_toks

    # Build summarization prompt from the middle messages
    summary_prompt = [
        {
            "role": "system",
            "content": (
                "Summarize the following conversation between an AI data analyst and its tools. "
                "Focus on: key findings, data results, errors encountered, and current analysis direction. "
                "Be concise but preserve all important factual information."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                f"[{_get_field(m, 'role', 'unknown')}]: {_get_field(m, 'content') or '(tool call)'}"
                for m in middle
            ),
        },
    ]

    response = client.chat.completions.create(
        model=MODEL, messages=summary_prompt, max_tokens=1000,
    )
    summary = response.choices[0].message.content

    # Replace messages in-place
    preserved_start = messages[:2]
    messages.clear()
    messages.extend(preserved_start)
    messages.append({"role": "assistant", "content": f"[Context Summary] {summary}"})
    messages.extend(preserved_end)

    new_msg_tokens = count_message_tokens(messages, encoding)
    return True, new_msg_tokens, tool_toks
