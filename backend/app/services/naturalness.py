from __future__ import annotations

from app.services.text_utils import split_sentences, normalize_space


REPLACEMENTS = {
    "a lot of advice": "a great deal of advice",
    "a lot of information": "a great deal of information",
    "very good": "strong / impressive",
    "very important": "essential / crucial",
    "a lot of": "many / a wide range of",
    "kids": "children",
    "stuff": "things / belongings / materials",
    "I think": "I would say / From my perspective",
    "Nowadays": "These days / In recent years",
    "make me happy": "lift my mood",
    "big problem": "serious issue",
}


def suggest_naturalness(text: str) -> list[str]:
    lowered = text.lower()
    suggestions: list[str] = []
    for plain, upgraded in REPLACEMENTS.items():
        if plain.lower() in lowered:
            suggestions.append(f"Replace '{plain}' with a more precise expression such as '{upgraded}'.")
    sentences = split_sentences(text)
    if len(sentences) == 1 and len(text.split()) > 45:
        suggestions.append("Break the answer into shorter sentences to improve fluency and listener processing.")
    if not suggestions:
        suggestions.append("Use one concrete example and one reflective detail to make the answer sound more natural.")
    return suggestions


def build_upgraded_answer(text: str) -> str:
    upgraded = normalize_space(text)
    for plain, replacement in REPLACEMENTS.items():
        first_choice = replacement.split(" / ")[0]
        upgraded = upgraded.replace(plain, first_choice)
        upgraded = upgraded.replace(plain.capitalize(), first_choice.capitalize())
    if upgraded == normalize_space(text):
        upgraded = (
            "I would answer more specifically by giving a clear example, explaining why it matters, "
            f"and then linking it back to the question: {upgraded}"
        )
    return upgraded
