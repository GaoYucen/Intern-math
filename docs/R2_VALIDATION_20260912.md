# R2 实施与验证记录 — 2026-09-12

## 当前结论

实现和代码层验收已完成；真实 API 的 3 道官方公开例题均答对。34 题同题对照尚未返回完整报告，因此目前不能宣布 R2 优于 R1，更不能把公开例题的 3/3 外推为官方隐藏集成绩。暂不消耗官方提交机会。

## 代码与隔离

- 分支：`r2-budgeted-tools-v1`；草稿 PR：#1，基于 `r1-robust-baseline`。
- 核心候选代码版本：`945f56ec1a54e7c7c9740b2764424f1d5a92899a`。
- 后续 `426ffb029f6dbcae4fa3b3949b3a1f5ae80f8f30` 只增加公开例题 smoke 工作流，没有修改候选算法。
- `main`、R1 和所有原提交分支保持不变。没有向赛事平台提交作品，也没有自动同步 AtomGit。
- 官方平台需选择 `intern-s2-preview-397b`。推理开关在候选代码中明确为 false，不依赖 Actions 环境变量。
- 每题最多 3 次模型请求，预算上限依次 4096/4096/2048；最多 2 次受限数学 Python 运算。完整答案立即返回，不做默认二次改写。

## 已完成：完整测试套件

GitHub Actions run `34625550549`，job `103349654514`：

> 91 passed in 2.28s

包括历史 R1 合约测试（针对不变的 `r1_agent.py`）、新 Agent、工具限制/超时、评分器不采用子串判对、部署配置与被测配置一致性。

生成的最小包 artifact ID：`10274317640`。

包内只有 `user_agent.py`、`r2_agent.py`、`math_tools.py`、`requirements.txt`、`DEPLOYMENT.txt`、`SHA256SUMS`。下载后逐项校验全部通过；在无 API key 且环境变量 `INTERN_THINKING_MODE=1` 的隔离导入测试中，官方入口仍明确以 thinking_mode=false 调用注入的模拟 client。

这是接口和代码测试，不是数学准确率测试。

## 已完成：真实 API 公开样例测试

GitHub Actions run `34626073808`；结果 artifact ID：`10274333784`。通过实际 `user_agent.py` 入口和 Intern API 调用，而非模拟模型。题目来自官方 baseline 的 3 道公开示例。

| 题目 | 模型最终答案 | 对照结果 | 单题耗时 |
| --- | --- | --- | --- |
| F81 中生成整个扩域的元素数 | 72 | 正确 | 24.30 秒 |
| 反函数积分的导数 F'(5) | -1 | 正确 | 29.78 秒 |
| z=1 简单极点的留数 | -1/8 | 正确 | 13.74 秒 |

3/3 正确，3/3 输出明确终答，每题 1 次模型调用；这三道简单题未触发工具。结果只能证明连接、实际入口和简单题基本链路工作正常，不能证明复杂工具链有效，也不是隐藏集高分证据。

## 运行中：34 题同题对照

- 初始原型 run：`34624791585`，commit `b1d9b0843b659da59a795516ce82c597d95ea9bd`。
- 工具安全加固后的重复对照 run：`34625550536`，commit `945f56ec1a54e7c7c9740b2764424f1d5a92899a`，串行排队。
- 从既有 Benchmark-v1 按预先固定的 SHA256 顺序，每学科选 2 题，17 学科共 34 题。是回归集，不是新 holdout。
- 对照为 R1 thinking-off 单次求解与 R2。两者均调用明确的 397B；计算上限不同，必须同时报告调用数、token 和耗时。
- 全部 34 题进入分母；严格匹配不能判定的符号/证明题，使用同一模型的双顺序匿名裁判。裁判分歧或失败保持待审，不从分母排除。
- 同模型裁判一致仍然只是代理评估，不能视为独立数学证明。
- 报告必须同时检查新增正确题、退步题、待审题、完整终答、调用异常和截断，不能凭工作流绿色状态或格式成功宣布高准确率。

截至本记录写入时，这两个 run 尚无可审阅的完整对照结果。是否值得正式提交，仍待这些证据。

## 来源

- 官方公开示例：`InternLM/Challenge-Cup-2026/sample_data/dev.jsonl`，读取时 blob `56f37b425619bac1d446c2e5924ccdd516c4a0c2`。
- 完整测试日志：https://github.com/GaoYucen/Intern-math/actions/runs/34625550549
- 公开样例日志：https://github.com/GaoYucen/Intern-math/actions/runs/34626073808
- 初始同题对照：https://github.com/GaoYucen/Intern-math/actions/runs/34624791585
- 加固后同题对照：https://github.com/GaoYucen/Intern-math/actions/runs/34625550536
