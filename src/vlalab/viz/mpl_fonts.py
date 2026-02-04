"""
Matplotlib 字体配置工具

目标：
- 自动选择可用的中文字体（优先 WenQuanYi / Noto / Source Han）
- 必要时通过字体文件路径 addfont（绕过 matplotlib 缓存导致的"明明装了字体却找不到"）
- 抑制常见的 Glyph missing / findfont 噪音警告
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class FontSetupResult:
    chosen_font: Optional[str]
    available_chinese_like_fonts: list[str]


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def setup_matplotlib_fonts(verbose: bool = True) -> FontSetupResult:
    """
    配置 matplotlib 中文字体。

    返回：
    - chosen_font: 选中的字体名称；若为 None 表示未找到合适中文字体
    - available_chinese_like_fonts: 检测到的"疑似中文字体"列表（用于诊断）
    """
    import os
    import warnings

    import matplotlib as mpl
    import matplotlib.font_manager as fm

    # 抑制字体相关噪音 warning（不影响实际渲染）
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing.*")
    warnings.filterwarnings("ignore", category=UserWarning, message="findfont: Font family.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

    # 常见中文字体文件路径（存在则 addfont，避免缓存/扫描问题）
    font_paths = [
        # WenQuanYi
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        # Noto CJK（不同发行版路径/后缀会不同，尽量覆盖）
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJKtc-Regular.otf",
        # Source Han
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansCN-Regular.otf",
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf",
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansTC-Regular.otf",
    ]

    for p in font_paths:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
            except Exception:
                # addfont 失败不应中断主流程
                pass

    # 重新获取字体列表（包含 addfont 后的新字体）
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    all_fonts_lower = [n.lower() for n in all_fonts]

    # 首选字体名称（按优先级）
    preferred_names = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Source Han Sans CN",
        "Source Han Sans SC",
        "Source Han Sans TC",
        # Windows/macOS 常见
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "STHeiti",
    ]

    chosen: Optional[str] = None

    # 1) 先严格按名称匹配
    available_set = set(all_fonts)
    for name in preferred_names:
        if name in available_set:
            chosen = name
            break

    # 2) 再做一次"模糊匹配"（不同系统字体命名略有差异）
    if chosen is None:
        preferred_keywords = [
            "wenquanyi",
            "wqy",
            "noto sans cjk",
            "noto cjk",
            "source han sans",
            "simhei",
            "yahei",
            "pingfang",
            "stheiti",
            "cjk",
        ]
        for kw in preferred_keywords:
            for i, name_l in enumerate(all_fonts_lower):
                if kw in name_l:
                    chosen = all_fonts[i]
                    break
            if chosen is not None:
                break

    # 诊断信息：列出"疑似中文字体"
    chinese_like = []
    chinese_keywords = [
        "wenquanyi",
        "wqy",
        "noto",
        "cjk",
        "source han",
        "simhei",
        "yahei",
        "pingfang",
        "heiti",
    ]
    for i, name_l in enumerate(all_fonts_lower):
        if any(k in name_l for k in chinese_keywords):
            chinese_like.append(all_fonts[i])
    chinese_like = sorted(set(chinese_like))

    if chosen is not None:
        # 注意：不要覆盖掉用户可能已有的 font.sans-serif 配置，采用"前置 + 去重"
        current = list(mpl.rcParams.get("font.sans-serif", []))
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = _unique_preserve_order([chosen, *current, "DejaVu Sans"])
        mpl.rcParams["axes.unicode_minus"] = False
        if verbose:
            print(f"[字体] 使用字体: {chosen}")
    else:
        # 没找到中文字体也不报错，只提示（图里中文会是方块）
        mpl.rcParams["axes.unicode_minus"] = False
        if verbose:
            print("[字体] 警告: 未找到可用中文字体，中文可能显示为方块")

    return FontSetupResult(chosen_font=chosen, available_chinese_like_fonts=chinese_like)
