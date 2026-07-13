# -*- coding: utf-8 -*-
"""图片显示名：数据库里存什么还是什么，这里只决定「界面上怎么念它」。

视频抽帧落库的文件名长这样：

    20260712_000602_575947_frame_000223.jpg

时间戳是为了不重名，帧号才是人要看的东西。一屏几十行全是这种名字时，
前 22 个字符完全一样，用户得横着扫到第 23 位才能分辨两张图——列表再宽也没用。
所以列表里显示「帧 223」，完整文件名留在 tooltip 里。

普通图片导入时 filename 存的就是用户自己的文件名（时间戳只加在磁盘副本上），
本来就可读——原样返回，让界面自己去打省略号，不要替用户改写文件名。

只做显示：不改文件名，不动数据库。
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# 抽帧文件名：<日期>_<时间>_<微秒>_frame_<帧号>.<扩展名>
# 由 core.import_manager.import_video 生成，格式变了这里会认不出来，
# 认不出来就原样显示——退化成「难看但正确」，不会显示成错的帧号。
_FRAME_NAME = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<micro>\d{6})_frame_(?P<frame>\d+)\.[^.]+$"
)


def parse_frame_name(filename: str) -> Optional[Dict[str, str]]:
    """认出抽帧生成的文件名，拆出日期 / 时间 / 帧号；不是这种名字就返回 None。"""
    match = _FRAME_NAME.match((filename or "").strip())
    return match.groupdict() if match else None


def display_name(filename: str) -> str:
    """一个文件名在界面上显示成什么。

    抽帧名 → 「帧 223」（前导零去掉；全是零就是「帧 0」，别显示成空的）。
    其余一律原样返回。
    """
    parsed = parse_frame_name(filename)
    if not parsed:
        return filename or ""
    return f"帧 {int(parsed['frame'])}"


def _discriminator(filename: str, level: str) -> Optional[str]:
    """重名时用来区分的那一小截，取自文件名里的生成时间。"""
    parsed = parse_frame_name(filename)
    if not parsed:
        return None

    if level == "date":
        date = parsed["date"]
        return f"{date[4:6]}-{date[6:8]}"          # 20260712 → 07-12

    time = parsed["time"]
    return f"{time[0:2]}:{time[2:4]}:{time[4:6]}"  # 000602 → 00:06:02


def display_names(filenames: Sequence[str]) -> List[str]:
    """一整个列表的显示名：重名的补一个最短的区分信息。

    同一段视频里帧号不会重复，重名只发生在「两段视频都抽到了第 223 帧」。
    这时先用抽帧日期分（多半就够了），同一天的用时间分，同一秒的才退到序号。
    """
    aliases = [display_name(name) for name in filenames]

    groups: Dict[str, List[int]] = defaultdict(list)
    for index, alias in enumerate(aliases):
        groups[alias].append(index)

    for alias, indexes in groups.items():
        if len(indexes) < 2:
            continue

        for level in ("date", "time"):
            marks = [_discriminator(filenames[i], level) for i in indexes]
            # 每个都拿得到、且两两不同，这一档才真的能把它们分开
            if all(marks) and len(set(marks)) == len(indexes):
                for index, mark in zip(indexes, marks):
                    aliases[index] = f"{alias} · {mark}"
                break
        else:
            # 时间也分不开（同一秒抽的，或者压根不是抽帧名）：给一个稳定的序号
            for ordinal, index in enumerate(indexes, start=1):
                aliases[index] = f"{alias} ({ordinal})"

    return aliases


# ==================== 项目级显示名称规则（U6） ====================
#
# 上面那一段只管「抽帧名怎么念」，这里再往上加一层：整个项目可以选一种
# 命名规则（original / source / custom），存进 projects.display_name_rule，
# 关掉软件重开还在。规则只决定界面上显示什么字符串，不碰 filename /
# original_path / storage_path，也不碰磁盘上的文件。

RULE_MODES = ("original", "source", "custom")

# Windows 不允许出现在文件名里的字符；虽然我们从不真的去改磁盘文件名，
# 但显示名要是员工照着抄去重命名文件，撞上这些字符会在 Windows 上直接失败，
# 所以在生成阶段就挡掉。
_ILLEGAL_CHARS = re.compile(r'[\\/:*?"<>|]')
MAX_DISPLAY_NAME_LENGTH = 96


def default_rule() -> Dict:
    """默认规则：保留原名（旧项目没设置过规则时，行为要和以前完全一样）。"""
    return {"mode": "original"}


def validate_display_name(name: str) -> Optional[str]:
    """一个即将展示的名字是否合法；合法返回 None，不合法返回中文原因。

    只用来把关「我们生成的名字」（source / custom 模式），不用来检查
    original 模式——那种模式就是把用户自己的文件名原样端出去，怎么样都不该
    因为里面有个冒号就报错给他看。
    """
    if not name:
        return "显示名称不能为空"
    if name != name.strip():
        return "显示名称首尾不能有空格"
    if name.endswith('.'):
        return "显示名称末尾不能是句点"
    illegal = sorted(set(_ILLEGAL_CHARS.findall(name)))
    if illegal:
        return f"显示名称不能包含以下字符：{' '.join(illegal)}"
    if len(name) > MAX_DISPLAY_NAME_LENGTH:
        return f"显示名称过长，建议不超过 {MAX_DISPLAY_NAME_LENGTH} 个字符"
    return None


def validate_custom_rule_fields(prefix, start, digits) -> Optional[str]:
    """custom 模式的三个参数是否合法；合法返回 None，不合法返回中文原因。"""
    if not prefix:
        return "前缀不能为空"
    if prefix != prefix.strip():
        return "前缀首尾不能有空格"
    if prefix.endswith('.'):
        return "前缀末尾不能是句点"
    illegal = sorted(set(_ILLEGAL_CHARS.findall(prefix)))
    if illegal:
        return f"前缀不能包含以下字符：{' '.join(illegal)}"
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        return "起始编号必须是不小于 0 的整数"
    if not isinstance(digits, int) or isinstance(digits, bool) or not (1 <= digits <= 10):
        return "编号位数必须是 1 到 10 之间的整数"
    return None


def validate_rule(rule) -> Optional[str]:
    """整条规则是否合法；合法返回 None，不合法返回中文原因。"""
    if not isinstance(rule, dict):
        return "规则格式不正确"
    mode = rule.get("mode")
    if mode not in RULE_MODES:
        return f"不支持的规则类型：{mode!r}"
    if mode == "custom":
        return validate_custom_rule_fields(
            rule.get("prefix"), rule.get("start"), rule.get("digits")
        )
    return None


def parse_display_name_rule(raw) -> Dict:
    """从数据库列（JSON 字符串，可能是 None / 空字符串 / 坏数据）解析出规则。

    解析失败一律退回 original——不能因为一条脏 JSON 让旧项目打不开。
    """
    if raw is None or raw == "":
        return default_rule()

    if isinstance(raw, dict):
        candidate = raw
    else:
        try:
            candidate = json.loads(raw)
        except (TypeError, ValueError):
            return default_rule()

    if validate_rule(candidate) is not None:
        return default_rule()

    mode = candidate.get("mode")
    if mode == "custom":
        return {
            "mode": "custom",
            "prefix": candidate["prefix"],
            "start": int(candidate["start"]),
            "digits": int(candidate["digits"]),
        }
    return {"mode": mode}


def serialize_display_name_rule(rule: Dict) -> str:
    """把规则字典存成数据库列里的 JSON 字符串。"""
    return json.dumps(rule, ensure_ascii=False)


def _clean_source_component(raw: str, fallback: str) -> str:
    """把一个来源标识（文件夹名 / 视频文件 stem）洗成能当显示名前缀的样子。"""
    text = (raw or "").strip()
    text = _ILLEGAL_CHARS.sub("_", text)
    text = text.rstrip('.').strip()
    return text or fallback


def _validate_generated_names(names: Sequence[str]) -> None:
    """生成完之后再兜底查一遍：不合法字符/超长，以及万一漏网的重名。"""
    for name in names:
        error = validate_display_name(name)
        if error:
            raise ValueError(f"生成的显示名称“{name}”不合法：{error}")

    seen = set()
    for name in names:
        if name in seen:
            raise ValueError(f"生成的显示名称重复：“{name}”，请更换规则参数后再试")
        seen.add(name)


def _build_source_names(images: Sequence[Dict], project_name: str) -> List[str]:
    fallback = _clean_source_component(project_name, "项目")

    raw_keys: List[str] = []
    clean_keys: List[str] = []
    frame_numbers: List[Optional[int]] = []

    for img in images:
        filename = img.get("filename", "") or ""
        parsed = parse_frame_name(filename)
        if parsed:
            source_path = img.get("original_path") or ""
            raw_key = Path(source_path).stem if source_path else (project_name or "项目")
            frame_numbers.append(int(parsed["frame"]))
        else:
            source_path = img.get("original_path") or ""
            raw_key = Path(source_path).parent.name if source_path else (project_name or "项目")
            frame_numbers.append(None)
        raw_keys.append(raw_key or (project_name or "项目"))
        clean_keys.append(_clean_source_component(raw_keys[-1], fallback))

    # 冲突检测：两个不同的来源标识，清洗后变成了同一个前缀——不能悄悄合并，
    # 用户分不清这两批图到底是谁，必须让他去改来源名字。
    raw_by_clean: Dict[str, str] = {}
    for raw_key, clean_key in zip(raw_keys, clean_keys):
        seen_raw = raw_by_clean.setdefault(clean_key, raw_key)
        if seen_raw != raw_key:
            raise ValueError(
                f"来源名称清洗后重复：“{seen_raw}”与“{raw_key}”都会变成显示名前缀"
                f"“{clean_key}”，请重命名其中一个来源后再试"
            )

    counters: Dict[str, int] = defaultdict(int)
    names = []
    for clean_key, frame_no in zip(clean_keys, frame_numbers):
        if frame_no is not None:
            names.append(f"{clean_key}_帧{frame_no:06d}")
        else:
            counters[clean_key] += 1
            names.append(f"{clean_key}_{counters[clean_key]:06d}")

    _validate_generated_names(names)
    return names


def _build_custom_names(rule: Dict, images: Sequence[Dict]) -> List[str]:
    prefix = rule.get("prefix", "")
    start = rule.get("start", 1)
    digits = rule.get("digits", 6)

    error = validate_custom_rule_fields(prefix, start, digits)
    if error:
        raise ValueError(error)

    # 前缀和编号之间要能一眼分清界限：自动补一个下划线，前缀已经以下划线
    # 结尾就不重复补——“阀门”+1+5位 要念成“阀门_00001”，不是“阀门00001”。
    separator = "" if prefix.endswith("_") else "_"
    names = [
        f"{prefix}{separator}{start + index:0{digits}d}" for index in range(len(images))
    ]
    _validate_generated_names(names)
    return names


def build_project_display_names(
    rule: Dict, images: Sequence[Dict], project_name: str = ""
) -> List[str]:
    """按项目规则批量生成显示名，返回与 images 一一对应、顺序一致的列表。

    images 需要按项目图片的稳定顺序传入（比如按 id / created_at 升序）——
    这个顺序既用来给 source / custom 模式编号，也是「同样的输入永远给同样
    结果」的前提。每个元素只读 filename / original_path 两个字段，不修改
    传入的字典，也不碰数据库或磁盘文件。

    Raises:
        ValueError: 规则参数不合法，或生成的名称违反命名限制 / 互相冲突——
            一律不静默处理，把错误消息原样展示给用户即可。
    """
    error = validate_rule(rule)
    if error:
        raise ValueError(error)

    mode = rule.get("mode")
    if mode == "original":
        return display_names([img.get("filename", "") for img in images])
    if mode == "source":
        return _build_source_names(images, project_name)
    return _build_custom_names(rule, images)
