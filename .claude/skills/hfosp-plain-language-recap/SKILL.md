---
name: hfosp-plain-language-recap
description: Use when about to write a user-facing status, recap, or explanation in this HFOsp project — answering "现在做到哪一步 / 为什么这么设计 / 揭示了什么", or drafting topic main-doc intros, §章节首句, archive doc abstracts, cross-topic emails or slide notes. Symptoms: about to type a PR code (PR-T4-2, PR-6, PR-2.5) or archive field name (Λ_gap, δ_excess=0.05, stable_k=2, forward_reverse_reproduced, axis_reversal_strict, swap_class, decision_k, h2_swap_check, INCONCLUSIVE-locked, lambda_fragile, producer_health, clinical_concordance, mechanism-sanity, cohort-claim primary) inside the main prose of a user-facing paragraph; about to say "PASS / NULL / INCONCLUSIVE / clean PASS / underpowered at n=6"; about to leak bare English code-words (swap, sanity, demoted) inline in a Chinese explanation. Non-project readers (or 6-months-future-you) cannot reconstruct 测了什么 / 怎么测的 / 揭示了什么 from such prose.
---

# HFOsp Plain-Language Recap

## Overview

`CLAUDE.md` §8 ("第一性原理表达：避免代号雪球") declares: every user-facing 解释 / 回顾 / 现状汇报 must lead with first-principles 朴素话; archive code names appear only as parenthetical 补注 or a trailing "（内部归档代号：…）" block. §8 is the *why*. **This skill is the ritual that makes §8 mechanical** — three-section 朴素话 as TodoWrite items before the draft, plus an "外部读者复述" self-check before sending.

**Core failure mode** (observed in 2026-05-28 RED baseline): even with §8 sitting in CLAUDE.md and the agent dutifully placing a `（内部归档代号：…）` block at the end, **半代号 still leaks into the inline prose** — phrases like "rank-displacement 判定它俩 swap 了", "decision_k ≈ 通道数一半时 null 饱和", "PR-2.5 forward_reverse / PR-6 h2_swap_check / rank_displacement swap_class 三层正交", "属于 mechanism-sanity 层不是 cohort-claim primary". A stranger (or 6-months-future-you) reading only the inline prose cannot reproduce 测了什么 / 怎么测的 / 揭示了什么. The 括号补注 block does not rescue this — **the prose itself must be self-contained 朴素话**.

## When to use

Fires whenever you are about to write user-facing prose that explains, recaps, or justifies project work:

- User asked "现在做到哪一步 / 我们到底揭示了什么 / 为什么这么设计 / 这个 figure 在展示什么 / 这个 PR 是干啥的"
- About to write a 引言 paragraph or 章节首句 in `docs/topic{1,2,3,4,5}_*.md` / `docs/paper_overview.md`
- Drafting an archive doc abstract / executive summary / "结论" 段
- Composing an email, slide note, or PPT bullet for cross-topic / cross-collaborator communication
- Writing verbal-report ghostwriter prose for the user to read aloud
- Replying with status when a long-running batch or audit finishes

Do **NOT** use for:

- Archive doc 正文 (TDD checklist, per-subject JSON schema, raw verdict tables, n-by-n cohort numbers) — 这些需要代号精度，朴素化反而失真
- Code comments — 代码引用代号是工程必要
- Tool-call status echoes ("已运行 pytest，10 个全过") and pure file-edit confirmations ("已写入 X")
- Internal-only PR plan content whose audience is "下一轮实现这个 PR 的 Agent 自己"

## Mandatory ritual (TodoWrite *before* drafting)

Create one TodoWrite item per step. Do not skip to drafting until items 1–3 exist:

1. **测了什么** — one sentence using only everyday physical / 日常因果 vocabulary describing what was actually measured. Forbidden in this sentence: PR codes (PR-T4-X, PR-6), math symbols (Λ_gap, δ_excess), archive field names (`forward_reverse_reproduced`, `swap_class`, `decision_k`, `h2_swap_check`). Allowed terms: 通道, 事件, 时序, 排名, 模板, 方向 — but only if their meaning is obvious from context. If the meaning isn't obvious, expand it ("把通道按发放先后排序" instead of "rank vector").

2. **怎么测的** — 1–2 sentences framing the test as **"如果完全随机的话，数应该是 X；实测是 Y"** comparison. Do not dump algorithm or function names; describe the comparison. Multi-layer tests must be named in plain words ("我们做了两层检验：第一层…第二层…"). If there's a magic threshold (cos ≥ 0.85, k ≥ 7, p < 0.05), either explain its physical meaning ("方向向量几乎完全反向才算") or honestly mark it as audit-fixed ("这个阈值是审计期确定的").

3. **揭示了什么** — conclusion phrased as **"在这个尺度上看起来像 X / 不像 X / 没看清"**, NOT as `PASS / NULL / INCONCLUSIVE / mechanism-sanity / cohort-claim primary / clean PASS / underpowered at n=6`. If sensitivity (small n, design saturation, structural outliers) is the limit, say so explicitly: "我们的检验在 X 这种情况下分辨不开"; "n=6 + 有结构性离群被试，置信区间太宽，结论是没看清，不是没有".

4. **外部读者复述 self-check** — before sending, cover the trailing 括号代号 block with your hand and read **only the inline prose**. Ask: can a stranger to this project reproduce 测了什么 / 怎么测的 / 揭示了什么? If no — identify which 半代号 leaked inline (use the Red-flags table below) and rewrite that sentence in 朴素话.

5. **代号补注 trailing block** — only *after* items 1–3 are 朴素-only, append a parenthetical "（内部归档代号：PR-T4-2, axis_reversal_strict / _shaped, swap_sweep.decision_k, H_orth, …）" tail for indexability. This block carries no explanation load — removing it must not damage the reader's ability to understand the paragraph.

## Red flags — STOP and rewrite

| Inline phrase | Why it leaks | Fix |
|---|---|---|
| "rank-displacement 判定它俩 swap 了" | "rank-displacement" 是模块代号; 外部读者不知道这是 "把通道排序前后对比" 的方法 | 改用朴素描述: "我们把每个事件里通道的发放先后顺序拿出来，跟随机排列比，确实跨事件能稳定看到角色互换" |
| "decision_k ≈ 通道数一半时 null 饱和" | `decision_k` + "null 饱和" 都是代号 | 改: "当我们要求'角色互换'的通道数恰好是总通道数的一半时，连随机重排都能达到同样的互换程度，真信号被淹没了" |
| "属于 mechanism-sanity 层不是 cohort-claim primary" | "mechanism-sanity" / "cohort-claim primary" 是 framework jargon | 改: "这只能算机制上的兜底确认 (n 小 + 设计就是 sanity 层)，不能写成队列级别的主结论" |
| "PR-2.5 的 forward_reverse, PR-6 的 h2_swap_check, rank_displacement 的 swap_class 三层正交" | 纯代号墙 | 改: "我们在三个层次都问过 swap: 模板整体上是不是一对反向 (模板层), 模板的源/汇端点是不是互换 (端点层), 哪些通道实际参与了互换 (通道层) — 三个层次回答的问题不一样，不能用一个 'H2 通过' 揉成一句" |
| "0.85 cos 阈值" 空降 | 魔数 | 加上物理含义 ("两个方向向量几乎完全相反才算对得上") 或来源 ("审计期定的阈值") |
| "0/9 strict, 5/9 shaped" 没上下文 | 数字不带含义 | 先说分子分母: "9 个有候选模板对的被试中，严格层 0 个跨过线、形态层 5 个对得上" |
| "PASS / NULL / INCONCLUSIVE / clean PASS / underpowered" 出现在 inline 主线 | §8 明确反例形态 | 一律改成 "看起来像 / 不像 / 没看清"，代号收到括号补注 |
| "P3 lag1_same_excess null-relative 干净 PASS" | §8 原文给出的反例直接出现 | 改成 §8 正例形态 ("我们看相邻两次事件挑的是不是同一个模板 …") |
| "λ_fragile / producer_health / clinical_concordance 良好" | 项目内部 framework 词 | 朴素描述这些指标在测什么 (如 "在多次重抽样里，这个频段的模板挑选还稳不稳"), 代号入括号 |
| inline 出现裸 "sanity 层" / "primary 层" / "demoted" | tier 标签词是 framework jargon，§6.3 pronoun discipline 反例 | 改 "设计上这一档就是兜底确认，不是用来下队列级主结论的"，或 "这一档被降级，不再算主结论" |
| inline 出现裸英文 "swap" / "endpoint" / "rank" 不带中文解释 | 半代号 — 朴素 prose 不应混英文 framework 词 | 第一次出现时朴素化（"角色互换" / "起止端通道" / "先后顺序排名"），后续用中文同义词，英文原词收到括号补注 |

## Rationalizations the baseline agent used (RED test 2026-05-28)

Captured from baseline pure-prompt test ("用户问 topic4 H2b 现在是个什么情况", CLAUDE.md §8 在 context 内但 skill 未触发):

- *"我已经把代号都收到末尾括号了，应该够了吧？"* → 不够。inline prose 仍然出现了 "rank-displacement 判定 swap"、"decision_k ≈ 通道数一半 null 饱和"、"PR-2.5 forward_reverse / PR-6 h2_swap_check / rank_displacement swap_class". 括号补注是 indexability, 不是解释。
- *"我把 mechanism-sanity 翻译成 '机制兜底、非主结论'，已经朴素了"* → "机制兜底" 仍是中性 jargon, 没说为什么这次只能停在机制兜底 (n=9 偏小 + 设计上就是 sanity tier + decision_k 饱和), 没说理由的"朴素"还是行话翻译。朴素 = 把*理由*讲出来。
- *"0.85 / decision_k / Λ_gap 都是审计期就锁的数字，没法重新解释了"* → 至少要给物理含义或来源；空降魔数对外部读者就是噪音。即使无法重新解释，也要明说"审计期确定，这一轮不动".
- *"用户问的是 topic4 H2b 状态，他懂代号，没必要朴素化"* → §8 适用范围明确包含 "用户问'现在做到哪一步'" 这个场景；用户即使懂代号，三段式朴素话仍是必须形态 (是为了 6-months-future-you 也能复述).
- *"我已经分了三段，骨架在了"* → 骨架在 ≠ 内容朴素。逐字检查每段 inline 是否漏代号才是 self-check 的内容。

## Quick reference: §8 三段式骨架

```
段 1 (测了什么):
  "我们看：<日常物理 / 日常因果讲清楚的现象>"

段 2 (怎么测的):
  "如果完全随机的话，<数学预期>。实测 <实际数字>，<比较结果>。"
  (多层时: "我们做了两层检验：<第一层朴素话>… <第二层朴素话>…")

段 3 (揭示了什么):
  "在这个尺度上 <看起来像 / 不像 / 没看清>，<理由>。"
  (有 sensitivity 限制时必须明说: "在 X 这种情况下我们分辨不开")

(内部归档代号：<PR-X, field_a, field_b, …>)
```

## Reference

- `CLAUDE.md` §8 — source-of-truth narrative with 反例/正例 sample paragraphs and 触发动作 self-check; this skill operationalizes §8 the same way `hfosp-deep-contract-verify` operationalizes §6.
- `MEMORY.md → feedback_first_principles_plain_language.md` — 2026-05-20 conversation snapshot that triggered §8 to be written.
- Related: `hfosp-deep-contract-verify` (CLAUDE.md §6 ritual — for implementation, not reporting); `superpowers:requesting-code-review` (when finalizing a section for review, this skill should fire on its 引言/abstract paragraph first).
