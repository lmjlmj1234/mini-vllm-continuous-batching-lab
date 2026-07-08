# Issue tracker: Local Markdown

Issues and PRDs for this repo live as markdown files in `docs/issues/`. Temporary planning and working notes go in `.scratch/`.

本仓库的 issue 和 PRD 以 Markdown 文件形式存储在 `docs/issues/` 中。临时规划和工作笔记存放在 `.scratch/`。

## Conventions / 约定

- **Final issues**: `docs/issues/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- **PRDs**: `docs/issues/<feature-slug>/PRD.md`
- **Temporary notes**: `.scratch/<feature-slug>/` — brainstorming, drafts, working state
- Triage state is recorded as a `Status:` line near the top of each issue file (see `triage-labels.md` for the role strings)
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

- 最终 issue：`docs/issues/<feature-slug>/issues/<NN>-<slug>.md`，从 `01` 开始编号
- PRD：`docs/issues/<feature-slug>/PRD.md`
- 临时笔记：`.scratch/<feature-slug>/` — 头脑风暴、草稿、工作状态
- 在 issue 文件顶部通过 `Status:` 行记录 triage 状态（标签字符串见 `triage-labels.md`）
- 评论和对话历史追加到文件底部 `## Comments` 标题下方

## When a skill says "publish to the issue tracker"

当 skill 要求"发布到 issue tracker"时：

Create a new file under `docs/issues/<feature-slug>/` (creating the directory if needed).

在 `docs/issues/<feature-slug>/` 下创建新文件（按需创建目录）。

## When a skill says "fetch the relevant ticket"

当 skill 要求"获取相关 ticket"时：

Read the file at the referenced path. The user will normally pass the path or the issue number directly.

读取指定路径下的文件。用户通常会直接提供路径或 issue 编号。

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

由 `/wayfinder` 使用。**map** 是一个包含每个 ticket 对应 **child** 文件的索引文件。

- **Map**: `.scratch/<effort>/map.md` — the Notes / Decisions-so-far / Fog body.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked, and unclaimed; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.

- **map 文件**：`.scratch/<effort>/map.md` — 记录笔记、当前决策、待探索部分
- **子 ticket**：`.scratch/<effort>/issues/NN-<slug>.md`，从 `01` 开始编号，正文包含问题。通过 `Type:` 行记录 ticket 类型（`research`/`prototype`/`grilling`/`task`）；通过 `Status:` 行记录 `claimed`/`resolved`
- **阻塞**：通过文件顶部 `Blocked by: NN, NN` 行表示。当所列的所有文件均为 `resolved` 时，ticket 解除阻塞
- **前沿**：扫描 `.scratch/<effort>/issues/` 寻找打开、未阻塞、未认领的文件；按编号取最先的
- **认领**：设置 `Status: claimed` 并保存后再开始工作
- **解决**：在 `## Answer` 标题下追加答案，设置 `Status: resolved`，然后在 `map.md` 的决策记录中追加上下文指针（要点 + 链接）
