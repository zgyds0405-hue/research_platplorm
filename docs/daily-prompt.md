# 科研日报选择规则

每日北京时间 03:00 云端运行。只从 OpenAlex、Crossref、Semantic Scholar 的真实元数据选三篇：两篇近期、一篇经典。优先 Nature、Science、Physical Review Letters、PNAS 等期刊，参考引用、作者影响力、时间和用户反馈；排除 90 天内重复。研究主题由 `config/research-topics.json` 配置。检索、去重、排序的可执行实现位于 `nightly.js`。结果写入 `cloud/daily.json`，每条保留来源、DOI 或原始链接。云端不调用 ChatGPT Plus；用户在本机点击“需要解读”时才调用已登录的 Codex `gpt-6.1-sol`，解读保存在本机。
