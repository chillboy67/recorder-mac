from __future__ import annotations

from app.models import AnalysisResult


PART1_TEMPLATE = """口语反馈
part1:（新题）
Q1:{q1_feedback}
Q2:{q2_feedback}
Q3:{q3_feedback}
Q4:{q4_feedback}
Q5:{q5_feedback}

复练建议：{practice_plan}
继续加油！
"""

PART2_TEMPLATE = """口语反馈
part2:（新题）
时长{duration}

批改反馈：
{part2_feedback}

语法问题：
{grammar_issues}

表达升级：
{natural_suggestions}

优化示范：
{upgraded_answer}

发音/流利度：
{pronunciation_issues}

复练建议：{practice_plan}

学生转写：
{transcript}

继续加油！
"""

PART3_TEMPLATE = """口语反馈
part3:（同上）
{part3_feedback}

复练建议：{practice_plan}
继续加油！
"""

DEFAULT_TEMPLATES = {
    "part1": PART1_TEMPLATE,
    "part2": PART2_TEMPLATE,
    "part3": PART3_TEMPLATE,
}


def render_report(result: AnalysisResult, template: str | None = None) -> str:
    values = _build_template_values(result)
    selected_template = template or DEFAULT_TEMPLATES.get(result.speaking_part, PART2_TEMPLATE)
    try:
        return selected_template.format(**values)
    except (KeyError, ValueError) as exc:
        fallback = DEFAULT_TEMPLATES.get(result.speaking_part, PART2_TEMPLATE).format(**values)
        return fallback + f"\n模板提示：自定义模板无法渲染，已使用默认模板。原因：`{exc}`\n"


def _build_template_values(result: AnalysisResult) -> dict[str, str]:
    return {
        "overall": _overall(result),
        "pronunciation_band": _band_label(len(result.pronunciation_issues)),
        "grammar_band": _band_label(len(result.grammar_issues)),
        "lexical_band": "表达可以继续升级" if result.natural_suggestions else "表达比较清楚",
        "fluency_band": result.fluency.note if result.fluency else "未评估",
        "transcript": result.transcript or "暂无学生发言文本",
        "grammar_issues": _format_grammar(result),
        "pronunciation_issues": _format_pronunciation(result),
        "natural_suggestions": _format_list(result.natural_suggestions),
        "upgraded_answer": result.upgraded_answer or "暂无优化答案",
        "practice_plan": _practice_plan(result),
        "duration": _format_duration(result),
        "q1_feedback": _part1_answer_feedback(result, 1),
        "q2_feedback": _part1_answer_feedback(result, 2),
        "q3_feedback": _part1_answer_feedback(result, 3),
        "q4_feedback": _part1_answer_feedback(result, 4),
        "q5_feedback": _part1_answer_feedback(result, 5),
        "part2_feedback": _part2_correction_feedback(result),
        "part3_feedback": _part3_expansion_feedback(result),
        "assistant_notes": _assistant_notes(result),
    }


def _overall(result: AnalysisResult) -> str:
    if not result.grammar_issues and not result.pronunciation_issues:
        return "学生回答整体清楚，先保持流利度，再补充更具体的例子。"
    return "以学生发言为主来看，思路基本可以，重点注意语法准确度、表达自然度和个别发音/停顿。"


def _practice_plan(result: AnalysisResult) -> str:
    if result.speaking_part == "part1":
        return "P1先练短答结构：直接回答 + 一个原因或细节，避免只说一句。"
    if result.speaking_part == "part3":
        return "P3先练观点展开：观点 + 原因链 + 例子/影响，注意连接词自然。"
    return "P2先复练纠错句，再按经历、细节、感受和结尾四步重讲一遍。"


def _part1_answer_feedback(result: AnalysisResult, question_number: int) -> str:
    if question_number == 1:
        if not result.grammar_issues:
            return "没有明显语法问题，回答可以再补充一个具体细节。"
        issue = result.grammar_issues[0]
        fix = f"，建议改为{issue.replacement}" if issue.replacement else ""
        return f"学生发言中注意{issue.category}：{issue.message}{fix}"
    if question_number == 2:
        if result.natural_suggestions:
            return _clean_suggestion(result.natural_suggestions[0])
        return "表达方向没有问题，可以补充原因，让答案不只停留在一句话。"
    if question_number == 3:
        return "可以按照原因 + 例子展开，比如补充一次真实经历或具体场景。"
    if question_number == 4:
        return "可以补充对比角度，例如过去和现在、自己和别人、国内和国外。"
    assistant = _assistant_notes(result)
    if assistant != "助教初步纠正：暂无。":
        return assistant
    return _overall(result)


def _part2_correction_feedback(result: AnalysisResult) -> str:
    lines = [_overall(result)]
    if result.grammar_issues:
        for issue in result.grammar_issues[:4]:
            fix = f"，建议改为{issue.replacement}" if issue.replacement else ""
            lines.append(f"语法/用词：{issue.message}{fix}")
    else:
        lines.append("语法时态没有明显问题，继续保持。")
    if result.pronunciation_issues:
        for issue in result.pronunciation_issues[:3]:
            lines.append(f"发音/流利度：注意{issue.word}，{issue.message}")
    else:
        lines.append("发音部分没有检测到明显问题；如果是音频并开启MFA，可以进一步检查单词级发音。")
    if result.upgraded_answer:
        lines.append(f"可替换/补充表达：{result.upgraded_answer}")
    assistant = _assistant_notes(result)
    if assistant != "助教初步纠正：暂无。":
        lines.append(assistant + " 如果与学生原句问题一致，可以纳入复练。")
    return "".join(line if line.endswith("。") else line + "。" for line in lines)


def _part3_expansion_feedback(result: AnalysisResult) -> str:
    lines = [
        "Q1:思路可以从公共场景、学习/工作场景、个人经历三个角度展开，先给观点再给例子。",
        "Q2:可以补充原因链，比如科技变化、个人习惯、社会效率这些方向，让答案更像part3。",
        "Q3:逻辑上建议用firstly / another reason / as a result组织，最后补一句影响或总结。",
    ]
    if result.natural_suggestions:
        lines.append(f"Q4:表达可以升级：{_clean_suggestion(result.natural_suggestions[0])}")
    if result.grammar_issues:
        issue = result.grammar_issues[0]
        fix = f"，例如{issue.replacement}" if issue.replacement else ""
        lines.append(f"Q5:拓展时也要注意{issue.category}{fix}，避免同类错误反复出现。")
    return "\n".join(lines)


def _assistant_notes(result: AnalysisResult) -> str:
    notes = (result.assistant_notes or "").strip()
    if not notes:
        return "助教初步纠正：暂无。"
    return f"助教初步纠正参考：{notes}"


def _clean_suggestion(suggestion: str) -> str:
    return suggestion.replace("Replace", "可以把").replace("with a more precise expression such as", "替换成更地道的表达，例如")


def _format_duration(result: AnalysisResult) -> str:
    if result.fluency and result.fluency.words_per_minute:
        return f"（约{result.fluency.words_per_minute} WPM，长停顿{result.fluency.estimated_pause_count}处）"
    if result.source == "audio":
        return "：已上传音频，需开启Whisper/MFA后可获得更精确时长"
    return "：未提供音频，无法判断"


def _band_label(issue_count: int) -> str:
    if issue_count == 0:
        return "这一段表现较稳定"
    if issue_count <= 2:
        return "整体不错，有少量问题需要注意"
    if issue_count <= 5:
        return "问题较集中，需要针对性练习"
    return "需要重点纠错和复练"


def _format_grammar(result: AnalysisResult) -> str:
    if not result.grammar_issues:
        return "没有明显语法问题。"
    return "\n".join(
        f"- {issue.category}: {issue.message} 原文片段：`{issue.context}`"
        + (f" 建议：`{issue.replacement}`" if issue.replacement else "")
        for issue in result.grammar_issues
    )


def _format_pronunciation(result: AnalysisResult) -> str:
    if not result.pronunciation_issues:
        return "没有检测到明显发音问题。"
    return "\n".join(f"- {issue.word}: {issue.message}" for issue in result.pronunciation_issues)


def _format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "没有额外建议。"
