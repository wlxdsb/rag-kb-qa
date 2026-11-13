"""检索评测：判断 top-k 检索结果是否命中预期关键词，统计"检索命中率"。

运行：python eval/run_eval.py   （先确保已跑过 ingest.py）

说明：这是一个轻量评测——只衡量"检索"这一步有没有把正确资料捞回来。
真实项目还可加"答案正确率"的人工/模型评测（见 README 的"下一步"）。
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)  # 让 eval/ 能 import 到上一层的 qa.py / config.py

from qa import get_store, retrieve  # noqa: E402


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    store = get_store()
    questions = load_questions(os.path.join(os.path.dirname(__file__), "questions.jsonl"))

    hit = 0
    for q in questions:
        docs = retrieve(store, q["question"])
        merged = "\n".join(d.page_content for d in docs)
        ok = q["expect"] in merged
        hit += int(ok)
        print(("✅ 命中" if ok else "❌ 未命中"), "|", q["question"])

    total = len(questions)
    rate = hit / total if total else 0
    print(f"\n检索命中率：{hit}/{total} = {rate:.0%}")


if __name__ == "__main__":
    main()
