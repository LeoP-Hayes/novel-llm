"""
微调前后生成质量对比评估

三种评估方法:
  1. LLM-as-Judge: DeepSeek V4 Pro 盲评打分（文风/爽感/连贯性/专业感/梗密度）
  2. 文本统计: 句长分布 / 对话占比 / 重复率 / 高潮词密度
  3. 训练语料相似度: 与《全职艺术家》《那年华娱》的 n-gram 重叠度

输出: output/evaluation_report.json
"""

import json, re, os
from pathlib import Path
from collections import Counter
from typing import Optional

import httpx

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))
from src.env import load_env, get_api_key
load_env()

# ============================================================
# 配置
# ============================================================
AFTER_DIR  = Path("output/novel_after")
BEFORE_DIR = Path("output/novel_before")
OUTPUT_FILE = Path("output/evaluation_report.json")
API_KEY = get_api_key("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"


# ============================================================
# 1. 文本统计
# ============================================================

def text_statistics(text: str) -> dict:
    """计算文本层面的量化指标"""
    # 基本统计
    chars = len(text.replace("\n", "").replace(" ", ""))
    sentences = re.split(r"[。！？…]", text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    paragraphs = [p for p in text.split("\n") if p.strip()]

    avg_sent_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
    avg_para_len = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)

    # 对话占比（引号内的内容）
    dialogues = re.findall(r"[“”](.+?)[“”]", text)
    if not dialogues:
        dialogues = re.findall(r"“(.+?)”", text)
    dialogue_chars = sum(len(d) for d in dialogues)
    dialogue_ratio = dialogue_chars / max(chars, 1)

    # 重复率（4-gram 去重比）
    chars_only = re.sub(r"\s", "", text)
    ngrams_4 = set()
    total_4 = 0
    for i in range(len(chars_only) - 3):
        ngrams_4.add(chars_only[i:i+4])
        total_4 += 1
    repetition_rate = 1 - len(ngrams_4) / max(total_4, 1)

    # 高潮词密度
    climax_words = ["震惊", "爆了", "目瞪口呆", "逆袭", "打脸", "奇迹",
                    "炸了", "热搜", "破纪录", "沸腾", "掌声", "轰动"]
    climax_count = sum(text.count(w) for w in climax_words)
    climax_density = climax_count / max(chars, 1) * 10000  # 每万字

    # 都市文娱高频词
    entertainment_words = ["票房", "首映", "综艺", "导演", "剧本", "拍摄",
                           "剧组", "上映", "排行", "微博", "粉丝", "经纪人",
                           "系统", "任务", "奖励", "重生", "前世"]
    ent_count = sum(text.count(w) for w in entertainment_words)
    ent_density = ent_count / max(chars, 1) * 10000

    # 句长分布（短/中/长句比例）
    short = sum(1 for s in sentences if len(s) < 20)
    medium = sum(1 for s in sentences if 20 <= len(s) < 60)
    long_s = sum(1 for s in sentences if len(s) >= 60)
    total_s = max(len(sentences), 1)

    return {
        "总字数": chars,
        "段落数": len(paragraphs),
        "平均句长": round(avg_sent_len, 1),
        "平均段长": round(avg_para_len, 1),
        "短句占比(<20字)": round(short/total_s, 3),
        "中句占比(20-60字)": round(medium/total_s, 3),
        "长句占比(≥60字)": round(long_s/total_s, 3),
        "对话占比": round(dialogue_ratio, 3),
        "重复率(4-gram)": round(repetition_rate, 3),
        "高潮词密度(/万字)": round(climax_density, 1),
        "文娱词密度(/万字)": round(ent_density, 1),
    }


# ============================================================
# 2. 训练语料相似度
# ============================================================

def corpus_similarity(text: str, corpus_dir: Optional[Path] = None) -> dict:
    """计算生成文本与训练语料的 n-gram 重叠度"""
    # 从训练数据中抽样（取前 200 条的 assistant 部分）
    sft_path = Path("data/sft/train.jsonl")
    if not sft_path.exists():
        return {"说明": "本地无训练数据，在 GPU 服务器上计算"}

    # 读取训练语料中的高频 4-gram
    corpus_ngrams = Counter()
    with open(sft_path) as f:
        for i, line in enumerate(f):
            if i > 200:
                break
            item = json.loads(line)
            assistant = item["messages"][-1]["content"]
            for j in range(len(assistant) - 3):
                corpus_ngrams[assistant[j:j+4]] += 1

    # 计算生成文本的 n-gram 在语料中的命中率
    gen_chars = re.sub(r"\s", "", text)
    hits = 0
    total = 0
    for i in range(len(gen_chars) - 3):
        if gen_chars[i:i+4] in corpus_ngrams:
            hits += 1
        total += 1

    overlap = hits / max(total, 1)

    # 不在语料中的新颖 n-gram（过高=偏离风格，过低=抄袭）
    return {
        "语料n-gram命中率": round(overlap, 3),
        "语料n-gram总量": len(corpus_ngrams),
        "说明": "0.3-0.5 = 风格匹配良好 | <0.2 = 偏离风格 | >0.6 = 可能过拟合"
    }


# ============================================================
# 3. LLM-as-Judge
# ============================================================

JUDGE_PROMPT = """你是网文质量评审专家。请对以下两段都市文娱小说片段（A和B）各自打分。

评分维度（每项 1-10 分）:
1. 文风匹配度: 是否符合都市文娱轻松诙谐、有梗、口语化的风格
2. 爽感: 是否有爽文的节奏感、打脸/逆袭的满足感
3. 连贯性: 情节发展是否自然，前后是否一致
4. 专业感: 对娱乐圈/影视行业的描写是否有真实感
5. 钩子: 结尾是否制造了悬念/期待感

【片段A】
%s

【片段B】
%s

请按以下 JSON 格式输出:
{{
  "A": {{"文风匹配度": X, "爽感": X, "连贯性": X, "专业感": X, "钩子": X}},
  "B": {{"文风匹配度": X, "爽感": X, "连贯性": X, "专业感": X, "钩子": X}},
  "总评": "一句话总结A和B的差异"
}}"""


def llm_judge(before_text: str, after_text: str) -> dict:
    """用 DeepSeek V4 Pro 盲评两段文本"""
    if not API_KEY:
        return {"error": "未配置 DEEPSEEK_API_KEY"}

    # 随机打乱 A/B 顺序（盲评）
    import random
    if random.random() > 0.5:
        text_a, text_b = before_text[:1500], after_text[:1500]
        a_is = "微调前"
    else:
        text_a, text_b = after_text[:1500], before_text[:1500]
        a_is = "微调后"

    prompt = JUDGE_PROMPT % (text_a, text_b)

    try:
        resp = httpx.post(
            API_URL,
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=60,
        )
        result = resp.json()["choices"][0]["message"]["content"]
        scores = json.loads(result)
        # 还原 A/B 对应关系
        if a_is == "微调后":
            scores["微调前"] = scores.pop("B")
            scores["微调后"] = scores.pop("A")
        else:
            scores["微调前"] = scores.pop("A")
            scores["微调后"] = scores.pop("B")
        return scores
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 主流程
# ============================================================

def main():
    # 读取全文
    before_full = (BEFORE_DIR / "before_full.txt").read_text()
    after_full  = (AFTER_DIR / "after_full.txt").read_text()

    report = {}

    # 1. 文本统计
    print("📊 文本统计...")
    report["微调前_统计"] = text_statistics(before_full)
    report["微调后_统计"] = text_statistics(after_full)

    # 2. 训练语料相似度
    print("📚 语料相似度...")
    report["微调前_语料相似度"] = corpus_similarity(before_full)
    report["微调后_语料相似度"] = corpus_similarity(after_full)

    # 3. LLM 盲评（取第 1 章作为样本，减少 API 费用）
    print("🤖 LLM 盲评...")
    before_ch1 = (BEFORE_DIR / "chapter_01.txt").read_text()
    after_ch1  = (AFTER_DIR / "chapter_01.txt").read_text()
    report["LLM盲评(第1章)"] = llm_judge(before_ch1, after_ch1)

    # 4. 逐章统计对比
    print("📈 逐章对比...")
    chapter_stats = []
    for i in range(1, 11):
        bf = (BEFORE_DIR / f"chapter_{i:02d}.txt").read_text()
        af = (AFTER_DIR / f"chapter_{i:02d}.txt").read_text()
        chapter_stats.append({
            "章": i,
            "微调前_字数": len(bf.replace("\n","").replace(" ","")),
            "微调后_字数": len(af.replace("\n","").replace(" ","")),
            "微调前_高潮词": sum(1 for w in ["震惊","爆了","逆袭","打脸","奇迹"] if w in bf),
            "微调后_高潮词": sum(1 for w in ["震惊","爆了","逆袭","打脸","奇迹"] if w in af),
        })
    report["逐章对比"] = chapter_stats

    # 保存
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n✅ 评估报告: {OUTPUT_FILE}")

    # 快速摘要
    print("\n" + "="*50)
    print("快速对比摘要")
    print("="*50)
    for k in ["总字数", "平均句长", "对话占比", "重复率(4-gram)", "高潮词密度(/万字)", "文娱词密度(/万字)"]:
        b = report["微调前_统计"].get(k, "?")
        a = report["微调后_统计"].get(k, "?")
        print(f"  {k:12s} | 微调前: {str(b):>8s} | 微调后: {str(a):>8s}")


if __name__ == "__main__":
    main()
