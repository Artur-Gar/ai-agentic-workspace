from typing import Dict, List

### Helpers
def format_memory(history: List[Dict[str, str]], max_turns: int = 10, max_assistant_chars: int = 600) -> str:
    """Render last N exchanges into a compact, model-friendly string."""
    if not history:
        return "No recent context."
    recent = history[-max_turns:]
    lines = []
    for m in recent:
        user = (m.get("user") or "").strip()
        assistant = (m.get("assistant") or "").strip()
        if len(assistant) > max_assistant_chars:
            assistant = assistant[:max_assistant_chars] + "..."
        lines.append(f"User: {user}\nAssistant: {assistant}")
    return "\n\n".join(lines)


def augment_task_with_memory(task: str, memory_text: str, extra_guidance: str = "") -> str:
    """Compose an augmented instruction that includes memory context without changing agent APIs."""
    guidance = extra_guidance.strip()
    guidance_block = f"\n\n### guidance\n{guidance}" if guidance else ""
    return (
        f"{task.strip()}\n\n"
        f"### relevant recent context\n{memory_text}"
        f"{guidance_block}\n\n"
        f"### instruction\n"
        f"If this request refines or modifies a previous result (e.g., filters, exclusions, formatting changes), "
        f"reuse the prior context and behave incrementally rather than starting from scratch."
    )
