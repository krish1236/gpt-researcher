"""RunForge wrapper for GPT Researcher — deep research analyst."""

from __future__ import annotations

import json
from typing import Any

from agent_runtime import AgentRuntime
from gpt_researcher import GPTResearcher
from gpt_researcher.utils.enum import Tone

runtime = AgentRuntime()


def _user_payload(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize trigger body: platform uses ``user_inputs``; local dev often sends a flat dict."""
    ui = input_payload.get("user_inputs")
    if isinstance(ui, dict):
        return dict(ui)
    skip = frozenset({"file_refs", "user_inputs"})
    return {k: v for k, v in input_payload.items() if k not in skip}


def _effective_inputs(ctx, input_payload: dict[str, Any]) -> dict[str, Any]:
    return {**dict(ctx.inputs), **_user_payload(input_payload)}


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default


def _as_str_list(v: Any) -> list[str] | None:
    if v is None:
        return None
    if isinstance(v, list):
        out = [str(x).strip() for x in v if str(x).strip()]
        return out or None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()] or None
            except json.JSONDecodeError:
                pass
        return [u.strip() for u in s.split(",") if u.strip()] or None
    return None


def _resolve_tone(raw: Any) -> Tone:
    if isinstance(raw, Tone):
        return raw
    if not raw or not isinstance(raw, str):
        return Tone.Objective
    s = raw.strip()
    for t in Tone:
        if t.name.lower() == s.lower():
            return t
    for t in Tone:
        if s in t.value or t.value.startswith(s):
            return t
    return Tone.Objective


@runtime.agent(
    name="deep-research-analyst",
    planned_steps=["research", "write_report"],
)
async def run(ctx, input: dict[str, Any]):
    eff = _effective_inputs(ctx, input)
    question = eff.get("research_question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "research_question is required (set in the run form or trigger JSON).",
        )

    report_type = eff.get("report_type") or "research_report"
    report_source = eff.get("report_source") or "web"
    tone = _resolve_tone(eff.get("tone"))
    source_urls = _as_str_list(eff.get("source_urls"))
    document_urls = _as_str_list(eff.get("document_urls"))
    complement_source_urls = _as_bool(eff.get("complement_source_urls"), False)

    with ctx.safe_step("research"):
        researcher = GPTResearcher(
            query=question.strip(),
            report_type=report_type,
            report_source=report_source,
            tone=tone,
            source_urls=source_urls,
            document_urls=document_urls,
            complement_source_urls=complement_source_urls,
            verbose=_as_bool(eff.get("verbose"), True),
        )
        await researcher.conduct_research()
        ctx.log("Research complete", level="info")

    with ctx.safe_step("write_report"):
        report = await researcher.write_report()
        ctx.artifact("report.md", report, "text/markdown")
        ctx.results.set_stats(
            {
                "report_length": len(report),
                "report_type": str(report_type),
            },
        )

    return {"status": "completed"}


if __name__ == "__main__":
    runtime.serve()
