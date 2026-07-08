# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

Skill 使用五个规范的 triage 角色术语。本文件将这些角色映射到本仓库 issue tracker 中实际使用的标签字符串。

| Role | Label | Meaning / 含义 |
| ---- | ----- | -------------- |
| `needs-triage` | `needs-triage` | Maintainer needs to evaluate this issue / 维护者需要评估此 issue |
| `needs-info` | `needs-info` | Waiting on reporter for more information / 等待报告人提供更多信息 |
| `ready-for-agent` | `ready-for-agent` | Fully specified, ready for an AFK agent / 已明确描述，AFK agent 可处理 |
| `ready-for-human` | `ready-for-human` | Requires human implementation / 需要人工实现 |
| `wontfix` | `wontfix` | Will not be actioned / 不会处理 |

## Usage / 使用方法

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from the table above.

In local markdown files, record the triage state as a `Labels:` or `Status:` line in the issue's frontmatter or header:

在本地 Markdown 文件中，通过 issue frontmatter 或头部中的 `Labels:` 或 `Status:` 行记录 triage 状态：

```markdown
---
title: Example issue
labels: needs-triage
---
```

Edit the right-hand column of the table above to match whatever vocabulary you actually use.

可根据需要编辑上表中的 Label 列，以匹配你实际使用的标签词汇。
