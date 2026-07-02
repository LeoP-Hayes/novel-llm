"""
逐章生成都市文娱小说
用法: python scripts/generate_novel.py --chapters 10
产出: output/novel/chapter_01.txt ~ chapter_10.txt + full.txt
"""

import torch, gc, argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 配置
# ============================================================
MODEL_FT   = "models/novel-merged"       # 微调后（LoRA合并模型）
MODEL_BASE = "Qwen/Qwen3-30B-A3B"        # 微调前（基座模型）
OUTPUT_FT   = Path("output/novel")        # 微调后输出
OUTPUT_BASE = Path("output/novel_before") # 微调前输出

SYSTEM_PROMPT = (
    "你是一个都市文娱小说作家，文风轻松诙谐、爽感十足。"
    "语言口语化、有梗、拒绝说教。每章 2000-3000 字，起承转合，结尾留钩子。"
)

# 黄金三章梗概
GOLDEN_THREE = """
【小说设定】
书名：重生之文娱帝国
主角：林风，北电导演系学生，重生回到2008年，绑定【文娱之神系统】
金手指：系统发布任务→完成后获得奖励（前世记忆中的剧本/歌曲/综艺模式）
风格：轻松诙谐，有梗，爽文，参考《全职艺术家》和《那年华娱》
"""


def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, device_map="auto"
    ).eval()
    return model, tokenizer


def generate_chapter(model, tokenizer, ch_num, prev_chapters, chapter_instruction):
    """生成一章"""

    # 构造上下文：前文摘要 + 本章指令
    context = GOLDEN_THREE + "\n\n"
    if prev_chapters:
        # 只保留最近 3 章作为上下文（节省 tokens）
        recent = prev_chapters[-3:]
        for ch in recent:
            # 每章取最后 500 字（保留最近的上下文）
            snippet = ch["text"][-500:] if len(ch["text"]) > 500 else ch["text"]
            context += f"\n【第{ch['num']}章】\n...{snippet}\n"

    user_msg = (
        f"{context}\n"
        f"【本章指令】\n{chapter_instruction}\n\n"
        f"请写第{ch_num}章。要求 2000-3000 字，有起承转合，结尾留悬念。"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=2000, temperature=0.8, top_p=0.9,
            do_sample=True, repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response


# ============================================================
# 10 章的指令大纲
# ============================================================
CHAPTER_PLANS = [
    "黄金三章第1章：林风重生回2008年北电宿舍，发现【文娱之神系统】激活。当前困境：穷学生+前女友分手。系统发布第一个新手任务。",
    "黄金三章第2章：林风完成新手任务，获得一首前世热门歌曲的记忆。在班级才艺展示中小试牛刀，引起小范围关注。",
    "黄金三章第3章：有人质疑林风抄袭，林风用实力证明自己——第一次打脸成功。更大的机遇（校园歌手大赛）降临，留下悬念。",
    "林风报名校园歌手大赛，用前世歌曲参赛。排练中遇到困难（设备、资金），开始接触校外的音乐制作人，埋下进入娱乐圈的伏笔。",
    "小高潮章：校园歌手大赛登台。林风的歌引爆全场，视频被传到网上，开始有经纪公司联系他。第一次感受到'红'的滋味。",
    "过渡章：林风在几家经纪公司间周旋，同时系统发布新任务——拍摄一部微电影。他第一次接触导演工作，埋下导演身份线。",
    "林风用系统奖励的微电影剧本，在学校组织小剧组拍摄。过程中遭遇各种搞笑+真实的剧组日常，认识女主角。",
    "微电影完成，在视频平台上线。起初没人看，林风发动同学转发。转折点：被一个大V转发，播放量爆发。",
    "小高潮章：微电影火了，林风受邀参加一个行业会议。遇到前女友和她的新男友（一个富二代导演），被当面嘲讽。林风淡定反击——打脸。",
    "大高潮章：林风凭借微电影拿到第一个行业奖项，正式进入娱乐圈。系统发布大任务——拍摄第一部院线电影。结尾：新的征程开始，留下悬念。",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", type=int, default=10)
    parser.add_argument("--base", action="store_true", help="使用微调前基座模型生成")
    parser.add_argument("--model", default=None, help="模型路径（覆盖默认值）")
    parser.add_argument("--output", default=None, help="输出目录（覆盖默认值）")
    args = parser.parse_args()

    if args.base:
        model_path = args.model or MODEL_BASE
        output_dir = Path(args.output) if args.output else OUTPUT_BASE
        label = "微调前（基座模型）"
    else:
        model_path = args.model or MODEL_FT
        output_dir = Path(args.output) if args.output else OUTPUT_FT
        label = "微调后（LoRA）"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"加载模型 ({label})...")
    model, tokenizer = load_model(model_path)

    # 逐章生成
    chapters = []
    full_text = GOLDEN_THREE + "\n\n"

    for i in range(min(args.chapters, len(CHAPTER_PLANS))):
        ch_num = i + 1
        plan = CHAPTER_PLANS[i]

        print(f"\n{'='*60}")
        print(f"生成第 {ch_num} 章...")
        print(f"指令: {plan[:60]}...")
        print(f"{'='*60}")

        response = generate_chapter(model, tokenizer, ch_num, chapters, plan)
        chapters.append({"num": ch_num, "text": response})

        # 保存单章
        ch_path = output_dir / f"chapter_{ch_num:02d}.txt"
        ch_path.write_text(f"第{ch_num}章\n\n{response}", encoding="utf-8")
        full_text += f"\n\n第{ch_num}章\n{response}"

        print(f"  ✅ 第{ch_num}章: {len(response)} 字 → 保存到 {ch_path.name}")

        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()

    # 保存全文
    full_path = output_dir / "full.txt"
    full_path.write_text(full_text, encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"✅ 全文保存到 {full_path}")
    print(f"   总字数: {sum(len(c['text']) for c in chapters)}")
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()
