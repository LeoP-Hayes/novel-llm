"""
大纲规划器

输入: 用户梗概（可选指定锚点章节）
输出: 结构化大纲 JSON（章节级高潮标注 + 三线编排）

设计原则: 用户锚点 > LLM 自由生成 > 约束规则兜底
节奏策略: 张力累积模型 — 不硬编码"每5章小高潮"，而是追踪张力值，
         达到阈值时自然触发高潮，产生有机的节奏波。

用法:
    from src.constraints.outline_planner import OutlinePlanner
    planner = OutlinePlanner()
    outline = planner.plan(user_prompt="重生2008年，北电导演系学生...", chapters=50)
"""

import json, os, re, random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx


# ============================================================
# 配置
# ============================================================

# 从 .env 加载
_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# 张力累积参数
TENSION_PARAMS = {
    "daily":        0.3,    # 日常章
    "build_up":     0.5,    # 铺垫章
    "conflict":     0.7,    # 冲突章
    "face_slap":    0.9,    # 打脸章
    "romance":      0.4,    # 感情章
    "release":      0.6,    # 发布章
    "decay":        0.15,   # 每章自然衰减（读者兴奋度消退）
    "mini_threshold": (2.0, 3.0),   # 小高潮张力阈值范围
    "major_threshold": (3.5, 5.0),  # 大高潮张力阈值范围
    "max_consecutive_flat": 6,       # 最多连续平淡章
    "max_consecutive_climax": 2,     # 最多连续高潮章
}

# 节奏模板池（避免单一节奏模式）
RHYTHM_TEMPLATES = [
    {"name": "快节奏", "mini_interval": (3,5), "major_interval": (8,12)},
    {"name": "慢热型", "mini_interval": (4,7), "major_interval": (10,16)},
    {"name": "过山车", "mini_interval": (2,4), "major_interval": (6,10)},
    {"name": "渐进式", "mini_interval": (3,6), "major_interval": (9,14)},
]


# ============================================================
# 大纲规划器
# ============================================================

@dataclass
class ChapterNode:
    chapter: int
    title: str = ""
    summary: str = ""              # 本章摘要
    climax_type: str = ""          # 高潮类型: mini/major/无
    scene_type: str = "daily"      # 场景类型
    tension_target: float = 0      # 目标张力值
    lines: dict = field(default_factory=lambda: {  # 三线
        "career": "",
        "romance": "",
        "daily": ""
    })
    user_anchor: bool = False      # 是否用户指定的锚点


class OutlinePlanner:
    """LLM 驱动的网文大纲规划器"""

    def __init__(self, model: str = "deepseek-v4-flash"):
        self.model = model
        self._template = random.choice(RHYTHM_TEMPLATES)

    def plan(self, user_prompt: str, chapters: int = 50) -> dict:
        """
        主入口: 根据用户梗概生成结构化大纲

        Returns:
            {
              "total_chapters": 50,
              "prompt": "用户原始梗概",
              "rhythm_template": "快节奏",
              "golden_three": [...],
              "chapters": [{chapter: 1, ...}, ...],
              "climax_map": {5: "mini", 10: "major", ...}
            }
        """
        print(f"📋 大纲规划中... ({chapters}章, 模板={self._template['name']})")

        # 1. 解析用户锚点
        anchors = self._parse_anchors(user_prompt, chapters)

        # 2. 生成章节骨架（LLM 填充锚点之间的空白）
        skeleton = self._generate_skeleton(user_prompt, chapters, anchors)

        # 3. 张力分配：为每章标定高潮位置
        chapters_data = self._assign_tension(skeleton, chapters, anchors)

        # 4. 约束兜底检查
        chapters_data = self._apply_constraints(chapters_data, chapters)

        result = {
            "total_chapters": chapters,
            "prompt": user_prompt,
            "rhythm_template": self._template["name"],
            "golden_three": chapters_data[:3],
            "chapters": chapters_data,
            "climax_map": {c["chapter"]: c["climax_type"]
                          for c in chapters_data if c["climax_type"]},
        }
        return result

    # ================================================================
    # 内部方法
    # ================================================================

    def _parse_anchors(self, prompt: str, total: int) -> dict[int, dict]:
        """
        从用户 prompt 中解析锚点章节。
        识别模式:
          - "第X章..." → 该章是锚点
          - "开头要虐" → ch1-3 锚点
          - "最后大结局..." → 最后 3 章锚点
          - "第15章拿奥斯卡" → ch15 是大高潮锚点
        """
        anchors = {}

        # 解析 "第X章" 模式
        for m in re.finditer(r"第\s*(\d+)\s*章[^，。]*[要会是需]([^，。]{2,30})", prompt):
            ch = int(m.group(1))
            goal = m.group(2)
            if 1 <= ch <= total:
                anchors[ch] = {
                    "summary": goal,
                    "is_major_climax": any(w in goal for w in ["奖","爆","冠","奇迹","巅峰","教父","封神"]),
                }

        # "开头" → 前三章锚点
        if "开头" in prompt and 1 not in anchors:
            anchors[1] = {"summary": "穿越/重生+金手指激活", "is_major_climax": False}
            anchors[3] = {"summary": "第一次打脸+悬念钩子", "is_major_climax": False}

        # "大结局" → 最后几章
        for tail_word in ["结局", "最后", "终章", "大结局"]:
            if tail_word in prompt and total not in anchors:
                anchors[total] = {"summary": "大结局", "is_major_climax": True}
                anchors[total-1] = {"summary": "终局铺垫", "is_major_climax": False}

        return anchors

    def _generate_skeleton(
        self, prompt: str, chapters: int, anchors: dict[int, dict]
    ) -> list[ChapterNode]:
        """
        调用 LLM 生成章节骨架。用户锚点作为硬约束嵌入 prompt。
        当 API 不可用或本地模式时，使用规则生成。
        """
        # 构造 LLM prompt
        anchor_desc = "\n".join(
            f"  - 第{ch}章必须: {info['summary']}"
            + (" [大高潮]" if info.get("is_major_climax") else "")
            for ch, info in sorted(anchors.items())
        ) if anchors else "  （无指定锚点，自由发挥）"

        system = (
            "你是一个网文大纲规划专家。请根据用户梗概生成章节大纲。\n"
            "输出 JSON 数组，每章一个对象: {\"chapter\": 1, \"title\": \"章节名\", "
            "\"summary\": \"30-60字摘要\", \"scene_type\": \"daily|release|variety|business|romance|face_slapping\", "
            "\"career\": \"事业线\", \"romance\": \"感情线\", \"daily\": \"日常线\"}\n"
            "注意: 黄金三章(ch1-3)、单元化叙事(每5章左右一个单元)、高潮节奏有波动感。"
        )

        user = (
            f"【用户梗概】\n{prompt}\n\n"
            f"【硬约束-必须满足】\n{anchor_desc}\n\n"
            f"【要求】生成 {chapters} 章大纲。"
        )

        try:
            resp = httpx.post(
                DEEPSEEK_URL,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.8,
                    "response_format": {"type": "json_object"},
                },
                headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
                timeout=120,
            )
            data = resp.json()["choices"][0]["message"]["content"]
            raw = json.loads(data)
            # 提取 chapters 数组（LLM 可能用不同 key）
            items = raw if isinstance(raw, list) else raw.get("chapters", raw.get("outline", [raw]))
            skeleton = []
            for i, item in enumerate(items[:chapters]):
                skeleton.append(ChapterNode(
                    chapter=i + 1,
                    title=item.get("title", f"第{i+1}章"),
                    summary=item.get("summary", ""),
                    scene_type=item.get("scene_type", "daily"),
                    lines={
                        "career": item.get("career", ""),
                        "romance": item.get("romance", ""),
                        "daily": item.get("daily", ""),
                    },
                    user_anchor=(i + 1) in anchors,
                ))
            # 补齐不足的章
            while len(skeleton) < chapters:
                skeleton.append(ChapterNode(chapter=len(skeleton)+1, summary="内容待生成"))
            return skeleton
        except Exception:
            pass

        # LLM 不可用时：降级为规则骨架
        return [ChapterNode(chapter=i+1, summary="内容待生成") for i in range(chapters)]

    def _assign_tension(
        self, skeleton: list[ChapterNode], total: int, anchors: dict[int, dict]
    ) -> list[dict]:
        """
        张力累积模型：为每章分配高潮标注。

        原理: 每章有一个"场景张力值"（由 scene_type 决定），累积到阈值时触发高潮，
        高潮后张力衰减。不硬编码每5章小高潮，而是在模板约束下自然波动。
        """
        template = self._template

        # 计算小高潮和大高潮的阈值（随机波动，避免固定间隔）
        mini_lo, mini_hi = TENSION_PARAMS["mini_threshold"]
        major_lo, major_hi = TENSION_PARAMS["major_threshold"]
        mini_threshold = random.uniform(mini_lo, mini_hi)
        major_threshold = random.uniform(major_lo, major_hi)

        tension = 2.0  # 起始张力
        chapters_data = []
        last_mini = 0
        last_major = 0
        consecutive_flat = 0
        consecutive_climax = 0

        for node in skeleton:
            ch = node.chapter

            # 场景基础张力
            scene_tension = {
                "daily": 0.3, "release": 0.6, "variety": 0.5,
                "business": 0.5, "romance": 0.4, "face_slapping": 0.9,
            }.get(node.scene_type, 0.3)

            # 锚点章直接标记（优先级最高）
            if node.user_anchor and anchors.get(ch, {}).get("is_major_climax"):
                climax = "major"
                tension = 1.5  # 高潮后衰减
                last_major = ch
                consecutive_climax = 0
            elif node.user_anchor:
                climax = "mini" if random.random() > 0.3 else "major"
                tension = max(1.0, tension - 1.0)
            else:
                # 正常累积
                tension += scene_tension
                tension -= TENSION_PARAMS["decay"]  # 自然衰减

                # 密度约束：太久没高潮 → 强制触发
                if ch - last_mini > TENSION_PARAMS["max_consecutive_flat"]:
                    climax = "mini"
                    tension = 1.5
                    last_mini = ch
                elif ch - last_major > 18:  # 18章没有大高潮 → 强制
                    climax = "major"
                    tension = 1.0
                    last_major = ch
                elif consecutive_climax >= TENSION_PARAMS["max_consecutive_climax"]:
                    climax = ""  # 不能连续高潮
                    tension = max(1.5, tension)
                elif tension > major_threshold and ch - last_major >= template["major_interval"][0]:
                    climax = "major"
                    tension = 1.0
                    last_major = ch
                    consecutive_climax += 1
                    major_threshold = random.uniform(major_lo, major_hi)  # 重新随机阈值
                elif tension > mini_threshold and ch - last_mini >= template["mini_interval"][0]:
                    climax = "mini"
                    tension = 1.5
                    last_mini = ch
                    consecutive_climax += 1
                    mini_threshold = random.uniform(mini_lo, mini_hi)
                else:
                    climax = ""
                    consecutive_climax = 0
                    consecutive_flat += 0 if scene_tension > 0.5 else 1

            chapters_data.append({
                "chapter": ch,
                "title": node.title or f"第{ch}章",
                "summary": node.summary,
                "scene_type": node.scene_type,
                "climax_type": climax,
                "tension": round(tension, 2),
                "lines": node.lines,
                "user_anchor": node.user_anchor,
            })

        return chapters_data

    def _apply_constraints(self, chapters_data: list[dict], total: int) -> list[dict]:
        """
        约束兜底检查：
          1. 黄金三章（硬约束）
          2. 密度约束（任意连续10章至少1个爽点）
          3. 高潮分布合理性
        """
        # 黄金三章兜底
        if not chapters_data[0].get("climax_type"):
            chapters_data[0]["summary"] += "（含金手指激活）"
            chapters_data[0]["scene_type"] = "face_slapping"
        if not chapters_data[2].get("climax_type"):
            chapters_data[2]["climax_type"] = "mini"

        # 密度兜底：任意连续10章至少1个小高潮
        for i in range(total - 9):
            window = chapters_data[i:i+10]
            if not any(c.get("climax_type") for c in window):
                # 在第6章加一个小高潮
                mid = i + 5
                chapters_data[mid]["climax_type"] = "mini"
                chapters_data[mid]["summary"] += "（约束补充高潮）"

        return chapters_data


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    planner = OutlinePlanner()

    test_prompt = (
        "重生2008年，林风成为北电导演系学生，绑定文娱之神系统。"
        "开头要展示前世困境和金手指激活。第15章靠一部小成本电影拿国际奖，"
        "第25章遇到事业最大危机（被资本封杀），第40章成为行业教父。"
        "结局要开放式，暗示新的征程。"
    )

    outline = planner.plan(test_prompt, chapters=50)

    print("\n" + "="*60)
    print("生成的大纲")
    print("="*60)
    print(f"节奏模板: {outline['rhythm_template']}")
    print(f"\n高潮地图:")
    for ch in outline["chapters"][:30]:
        marker = {"mini": "🔥", "major": "💥"}.get(ch["climax_type"], "  ")
        anchor = "⚓" if ch["user_anchor"] else "  "
        print(f"  {anchor} {marker} ch{ch['chapter']:2d} [{ch['scene_type']:12s}] tension={ch['tension']:.1f}  {ch['summary'][:50]}")
    if len(outline["chapters"]) > 30:
        print(f"  ... 共 {len(outline['chapters'])} 章")
