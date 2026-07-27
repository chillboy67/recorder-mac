from __future__ import annotations

import re
import httpx
from app.config import settings
from app.models import GrammarIssue


async def check_grammar(text: str) -> tuple[list[GrammarIssue], list[str]]:
    warnings: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            response = await client.post(
                settings.languagetool_url,
                data={"text": text, "language": "en-US"},
            )
            response.raise_for_status()
            return _from_languagetool(response.json()), warnings
    except Exception:
        warnings.append("LanguageTool local server is unavailable; using built-in offline grammar heuristics.")
        return _fallback_rules(text), warnings


def _from_languagetool(payload: dict) -> list[GrammarIssue]:
    issues: list[GrammarIssue] = []
    for match in payload.get("matches", []):
        replacements = match.get("replacements") or []
        replacement = replacements[0].get("value") if replacements else None
        issues.append(
            GrammarIssue(
                message=match.get("message", "Possible grammar issue."),
                context=match.get("context", {}).get("text", ""),
                replacement=replacement,
                category=match.get("rule", {}).get("category", {}).get("id", "grammar").lower(),
                severity="medium",
            )
        )
    return issues


def _fallback_rules(text: str) -> list[GrammarIssue]:
    rules: list[tuple[str, str, str | None, str]] = [
        (r"\bI has\b", "Use 'have' with I.", "I have", "subject-verb agreement"),
        (r"\bHe have\b", "Use 'has' with he.", "He has", "third-person singular"),
        (r"\bShe have\b", "Use 'has' with she.", "She has", "third-person singular"),
        (r"\bIt have\b", "Use 'has' with it.", "It has", "third-person singular"),
        (r"\bpeople is\b", "Use 'are' with plural nouns.", "people are", "subject-verb agreement"),
        (r"\ba university\b", "Check article pronunciation; 'a university' is correct because it starts with a /ju:/ sound.", None, "article"),
        (r"\ban useful\b", "Use 'a' before useful.", "a useful", "article"),
        (r"\bmore better\b", "Avoid double comparatives.", "better", "word choice"),
        (r"\bdiscuss about\b", "Use 'discuss' without 'about'.", "discuss", "collocation"),
        (r"\binformations\b", "'Information' is usually uncountable.", "information", "noun form"),
        (r"\badvices\b", "'Advice' is usually uncountable.", "advice", "noun form"),
    ]
    issues: list[GrammarIssue] = []
    for pattern, message, replacement, category in rules:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            issues.append(
                GrammarIssue(
                    message=message,
                    context=text[max(0, match.start() - 35) : match.end() + 35],
                    replacement=replacement,
                    category=category,
                    severity="medium",
                )
            )
    return issues
