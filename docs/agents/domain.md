# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

说明工程 skill 在探索代码库时应如何消费本仓库的领域文档。

## Before exploring, read these / 探索前请先阅读

- **`CONTEXT.md`** at the repo root / 仓库根目录下的 `CONTEXT.md`
- **`docs/adr/`** — read ADRs that touch the area you're about to work in / 阅读与你即将工作的领域相关的 ADR

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill creates them lazily when terms or decisions actually get resolved.

如果这些文件不存在，**请静默继续**。不要标记缺少这些文件，也不要建议提前创建它们。`/domain-modeling` skill 会在生成术语或决策时按需创建。

## File structure / 文件结构

Single-context repo (most repos) / 单上下文仓库（大多数项目）：

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-example-architecture-decision.md
│   └── 0002-example-technology-choice.md
└── src/
```

> **Note**: ADR files listed above are examples only. Real ADRs are created only when an architectural decision is actually made.
>
> **注意**：上列 ADR 文件仅为示例。实际的 ADR 只有在做出架构决策时才会创建。

## Use the glossary's vocabulary / 使用术语表词汇

When your output names a domain concept, use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

当你的输出提到领域概念时，请使用 `CONTEXT.md` 中定义的术语。不要使用术语表明确规避的同义词。

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

如果需要的概念尚未收录在术语表中，这是一个信号——要么你在创造项目不使用的语言（重新考虑），要么存在真正的空白（记录下来供 `/domain-modeling` 处理）。

## Flag ADR conflicts / 标记 ADR 冲突

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

如果你的输出与现有 ADR 存在矛盾，请明确提出来，而不是静默覆盖：

> _Contradicts ADR-0001 (event-sourced orders) — but worth reopening because…_
>
> _与 ADR-0001（事件溯源订单）存在矛盾——但值得重新讨论，因为……_
