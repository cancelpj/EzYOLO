# -*- coding: utf-8 -*-
"""
EzYOLO 样式定义

一套集中管理的设计令牌（颜色 / 字体 / 圆角 / 尺寸）+ 全局样式表。
页面通过 objectName 声明「这是什么」，样式在这里统一给出，
避免每个页面各写一份内联 QSS。

外观基调：浅色 / 暗色两套令牌，克制、层级靠留白和字重拉开，而不是靠色块和粗边框。
表面只有三层——画布（background）、卡片（panel）、控件（inset）；
彩色只留给「主操作」和「状态」。

按钮层级（用 setObjectName 指定）:
    primary   —— 当前页面唯一的主操作（实心蓝）
    (默认)     —— 次级操作（白底描边）
    secondary —— 同默认，历史兼容
    ghost     —— 弱化的第三级操作（无边框）
    danger    —— 破坏性操作（红字描边）
    link      —— 纯文字入口

文字层级（objectName）:
    h1 页面标题 / h2 区块标题 / title 卡片标题
    subtitle 说明 / caption 辅助小字 / muted 弱化 / value 大数字
"""

from pathlib import Path
from typing import Dict, List

# ==================== 颜色令牌 ====================

THEME_LIGHT = 'light'
THEME_DARK = 'dark'
THEME_CHOICES = (
    (THEME_LIGHT, '浅色'),
    (THEME_DARK, '暗色'),
)

# 语义色有「文字版」和「填充版」两套：
# 同一个亮色在浅底上当填充好看，当小字就糊了（#34C759 在白底上对比度只有 2:1）。
# 页面里 COLORS['success'] / ['error'] / ['warning'] 绝大多数是拿去当 color: 用的，
# 所以这三个键给可读的深色版，亮色版另存 *_fill，只用在圆点、进度条这类填充上。
LIGHT_COLORS: Dict[str, str] = {
    # 表面：画布 → 侧栏 → 卡片 → 控件
    'background': '#F5F5F7',   # 页面画布
    'sidebar': '#EFEFF2',      # 左侧导航，比画布再灰一点
    'panel': '#FFFFFF',        # 卡片 / 分组面板
    'inset': '#FFFFFF',        # 输入框、列表等控件表面
    'track': '#E6E6EB',        # 进度条、滑杆的底槽
    'hover': '#EBEBF0',
    'selected': '#E4EEFF',     # 选中态的蓝色底纹

    # 边框：1px、淡
    'border': '#E3E3E8',
    'border_strong': '#D2D2D7',

    # 强调色
    'primary': '#007AFF',
    'primary_hover': '#1A88FF',
    'primary_pressed': '#0062CC',
    'accent_text': '#0066CC',  # 浅底上的蓝色文字，够对比度
    'secondary': '#6E6E73',

    # 语义色（text 版本，浅底上可读）
    'success': '#248A3D',
    'success_soft': '#E7F6EC',  # 「已标注」这类状态底色
    'warning': '#9A5B00',
    'error': '#D70015',

    # 语义色（fill 版本，用于圆点 / 进度 / 实心块）
    'success_fill': '#34C759',
    'warning_fill': '#FF9500',
    'error_fill': '#FF3B30',

    # 文字
    'text_primary': '#1D1D1F',
    'text_secondary': '#6E6E73',
    'text_disabled': '#AEAEB2',
}

DARK_COLORS: Dict[str, str] = {
    # 表面：深灰分层，避免纯黑刺眼
    'background': '#1C1C1E',
    'sidebar': '#242426',
    'panel': '#2C2C2E',
    'inset': '#3A3A3C',
    'track': '#48484A',
    'hover': '#3A3A3C',
    'selected': '#1A3A5C',

    'border': '#3A3A3C',
    'border_strong': '#545456',

    'primary': '#0A84FF',
    'primary_hover': '#409CFF',
    'primary_pressed': '#0066CC',
    'accent_text': '#64B5FF',
    'secondary': '#98989D',

    # 暗底上文字版语义色要更亮
    'success': '#30D158',
    'success_soft': '#1E3A28',
    'warning': '#FFD60A',
    'error': '#FF453A',

    'success_fill': '#30D158',
    'warning_fill': '#FF9F0A',
    'error_fill': '#FF453A',

    'text_primary': '#F5F5F7',
    'text_secondary': '#98989D',
    'text_disabled': '#636366',
}

# 当前生效的调色板；页面里 `from gui.styles import COLORS` 后读到的始终是这一套。
COLORS: Dict[str, str] = dict(LIGHT_COLORS)
_current_theme = THEME_LIGHT


def normalize_theme(theme: str) -> str:
    """把设置里存的值规范成 'light' 或 'dark'。"""
    key = (theme or '').strip().lower()
    if key in (THEME_DARK, 'dark', '暗色', '暗色主题', '深色', '深色主题'):
        return THEME_DARK
    return THEME_LIGHT


def get_theme() -> str:
    """返回当前主题键。"""
    return _current_theme


def set_theme(theme: str) -> str:
    """切换活动调色板（原地更新 COLORS，已有引用自动跟上）。"""
    global _current_theme
    _current_theme = normalize_theme(theme)
    palette = DARK_COLORS if _current_theme == THEME_DARK else LIGHT_COLORS
    COLORS.clear()
    COLORS.update(palette)
    return _current_theme


def get_full_stylesheet(theme: str = None) -> str:
    """根据主题生成完整样式表。theme 省略时使用当前主题。"""
    if theme is not None:
        palette = DARK_COLORS if normalize_theme(theme) == THEME_DARK else LIGHT_COLORS
        return generate_stylesheet(palette)
    return generate_stylesheet(COLORS)


def apply_theme_to_app(theme: str = None) -> str:
    """切换主题并套到 QApplication（若已创建）。"""
    if theme is not None:
        set_theme(theme)
    else:
        set_theme(_current_theme)

    try:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(get_full_stylesheet())
    except Exception:
        pass
    return _current_theme

# ==================== 尺寸令牌 ====================

RADIUS = 12         # 卡片
RADIUS_SM = 8       # 控件

CONTROL_HEIGHT = 30      # 按钮 / 输入框统一高度
CONTROL_HEIGHT_LG = 34   # 主操作按钮

# ==================== 字体 ====================

# 每个平台上优先选用的字体，按顺序取第一个真实存在的。
# 只请求系统上真的装了的字体，Qt 才不会报字体缺失警告，
# 也不会悄悄回退到一个难看的默认族。
_FONT_CANDIDATES = [
    "PingFang SC",        # macOS 中文
    "SF Pro Text",        # macOS 西文（较新系统）
    "Helvetica Neue",     # macOS 回退
    "Hiragino Sans GB",
    "Microsoft YaHei",    # Windows
    "Segoe UI",
    "Noto Sans CJK SC",   # Linux
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "Arial",
]

# 日志、代码这类等宽场景。Consolas 只有 Windows 有，macOS 上要用 Menlo / SF Mono。
_MONO_CANDIDATES = [
    "SF Mono",
    "Menlo",
    "Consolas",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Courier New",
]

_font_stack_cache: List[str] = []
_mono_stack_cache: List[str] = []


def _available_families() -> set:
    try:
        from PyQt6.QtGui import QFontDatabase

        return set(QFontDatabase.families())
    except Exception:
        return set()


def get_font_stack() -> List[str]:
    """返回当前系统上真实存在的界面字体族列表（结果缓存）。

    需要在 QApplication 创建之后调用；未创建时退化为空列表，
    样式表就不写 font-family，交给 Qt 默认字体。
    """
    global _font_stack_cache
    if _font_stack_cache:
        return _font_stack_cache

    available = _available_families()
    _font_stack_cache = [name for name in _FONT_CANDIDATES if name in available]
    return _font_stack_cache


def get_mono_font_stack() -> List[str]:
    """返回当前系统上真实存在的等宽字体族列表（日志框用）。"""
    global _mono_stack_cache
    if _mono_stack_cache:
        return _mono_stack_cache

    available = _available_families()
    _mono_stack_cache = [name for name in _MONO_CANDIDATES if name in available]
    return _mono_stack_cache


def get_primary_font_family() -> str:
    """返回首选界面字体族名；系统上一个都没有时返回空字符串，交给 Qt 默认字体。"""
    stack = get_font_stack()
    return stack[0] if stack else ""


def mono_font_family_css() -> str:
    """给日志框等等宽场景用：返回可直接写进 QSS 的 font-family 值。

    只列系统上真实存在的字体，避免出现 “missing font family Consolas” 这类警告。
    没有可用等宽字体时返回空字符串，调用方应整行不写 font-family。
    """
    stack = get_mono_font_stack()
    if not stack:
        return ""
    return ", ".join(f"'{name}'" for name in stack)


def _font_family_declaration() -> str:
    """生成 QSS 的 font-family 声明。

    Qt 的样式表不认 CSS 的通用族（sans-serif 会被当成一个真实字体名去找，
    找不到同样会报字体缺失），所以这里只列真实存在的字体；
    一个都没有时就整行不写，交给 Qt 默认字体。
    """
    families = [f'"{name}"' for name in get_font_stack()]
    if not families:
        return ""
    return f"    font-family: {', '.join(families)};\n"


# ==================== 箭头图标 ====================

# 下拉框和数字框的箭头。Qt 的规矩是：一旦用样式表接管 ::drop-down / ::up-button，
# 系统画的那个箭头就一起没了——下拉框会剩一个 macOS 原生的黑色小三角贴在角上，
# 数字框则连上下箭头都不剩（试过：只写 ::up-button 不写 ::up-arrow，步进器整个消失）。
# 所以箭头得自己给。QSS 里能画出箭头的只有 image: url(...)：border 拼三角那套
# CSS 技巧 Qt 不认（它不做斜接，直接糊成一个方块）。
#
# 这些 svg 由本项目自己画，没有第三方素材。颜色跟下面的令牌对齐——
# 正常态 COLORS['primary']、禁用态 COLORS['text_disabled']；改令牌时记得一起改。
_ASSETS_DIR = Path(__file__).parent / "assets"


def _arrow_url(name: str) -> str:
    """QSS 的 url() 要一个能直接打开的路径，所以给绝对路径（POSIX 分隔符，Windows 也认）。"""
    return (_ASSETS_DIR / name).as_posix()


def set_menu_indicator(button, enabled: bool = True):
    """标记带菜单的按钮，并立即刷新动态属性对应的样式。"""
    button.setProperty("menuIndicator", bool(enabled))
    style = button.style()
    if style is not None:
        style.unpolish(button)
        style.polish(button)
    button.updateGeometry()
    button.update()


# ==================== 样式表 ====================

def generate_stylesheet(colors: dict) -> str:
    """根据颜色令牌生成全局样式表。"""
    c = colors
    font_family = _font_family_declaration()

    chevron_down = _arrow_url("chevron_down.svg")
    chevron_down_white = _arrow_url("chevron_down_white.svg")
    chevron_up = _arrow_url("chevron_up.svg")
    chevron_down_off = _arrow_url("chevron_down_disabled.svg")
    chevron_up_off = _arrow_url("chevron_up_disabled.svg")

    return f"""
/* ---------- 基础 ---------- */
QMainWindow, QDialog {{
    background-color: {c['background']};
    color: {c['text_primary']};
}}

QWidget {{
    background-color: {c['background']};
    color: {c['text_primary']};
{font_family}    font-size: 13px;
}}

QFrame {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS}px;
}}

/* QStackedWidget / QSplitter 都继承自 QFrame，别让它们套上卡片的边框和底色 */
QStackedWidget {{
    background-color: transparent;
    border: none;
    border-radius: 0px;
}}

QSplitter {{
    background-color: transparent;
    border: none;
}}

QSplitter::handle {{
    background-color: transparent;
}}

QSplitter::handle:horizontal {{
    width: 10px;
}}

QSplitter::handle:vertical {{
    height: 10px;
}}

QScrollArea {{
    background-color: transparent;
    border: none;
}}

QToolTip {{
    background-color: #3A3A3C;
    color: #FFFFFF;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
    padding: 8px 10px;
}}

/* ---------- 文字 ---------- */
QLabel {{
    background-color: transparent;
    border: none;
    color: {c['text_primary']};
    font-size: 13px;
}}

QLabel#h1 {{
    font-size: 21px;
    font-weight: 600;
    color: {c['text_primary']};
}}

QLabel#h2 {{
    font-size: 15px;
    font-weight: 600;
    color: {c['text_primary']};
}}

QLabel#title {{
    font-size: 14px;
    font-weight: 600;
    color: {c['text_primary']};
}}

QLabel#subtitle, QLabel#muted {{
    font-size: 13px;
    color: {c['text_secondary']};
}}

QLabel#caption {{
    font-size: 12px;
    color: {c['text_secondary']};
}}

QLabel#value {{
    font-size: 22px;
    font-weight: 600;
    color: {c['text_primary']};
}}

/* ---------- 卡片 ---------- */
QFrame#card {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS}px;
}}

QFrame#toolbar {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS}px;
}}

QFrame#divider {{
    background-color: {c['border']};
    border: none;
    border-radius: 0px;
    max-height: 1px;
    min-height: 1px;
}}

/* ---------- 主窗口结构 ---------- */
QWidget#sidebar {{
    background-color: {c['sidebar']};
    border: none;
    border-right: 1px solid {c['border']};
}}

QWidget#sidebar QLabel {{
    background-color: transparent;
}}

QLabel#brand {{
    font-size: 17px;
    font-weight: 700;
    color: {c['text_primary']};
}}

QLabel#nav_section {{
    font-size: 11px;
    font-weight: 600;
    color: {c['text_disabled']};
}}

QWidget#page_header {{
    background-color: {c['background']};
    border: none;
    border-bottom: 1px solid {c['border']};
}}

QWidget#page_header QLabel {{
    background-color: transparent;
}}

/* 页头的「第 N 步」：灰色小胶囊，不跟主操作抢颜色 */
QLabel#step_badge {{
    background-color: {c['hover']};
    color: {c['text_secondary']};
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 600;
}}

/* ---------- 流程导航项 ---------- */
/* 内边距不写在这里：step_item 里面是自己的布局，QSS 的 padding 影响不到子控件，
   只会把按钮的 sizeHint 虚增一截。内边距由 StepNavItem 的布局边距给。 */
QPushButton#step_item {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: {RADIUS_SM}px;
    text-align: left;
    padding: 0px;
}}

QPushButton#step_item:hover {{
    background-color: {c['hover']};
}}

QPushButton#step_item:checked {{
    background-color: {c['selected']};
    border: 1px solid transparent;
}}

QPushButton#step_item:disabled {{
    background-color: transparent;
}}

/* 兼容旧的侧边栏按钮样式 */
QPushButton#nav_button {{
    background-color: transparent;
    color: {c['text_primary']};
    border: none;
    border-radius: {RADIUS_SM}px;
    padding: 9px 12px;
    text-align: left;
    font-size: 13px;
}}

QPushButton#nav_button:hover {{
    background-color: {c['hover']};
}}

QPushButton#nav_button:checked {{
    background-color: {c['selected']};
    color: {c['text_primary']};
}}

QPushButton#nav_button:disabled {{
    color: {c['text_disabled']};
}}

/* ---------- 按钮层级 ---------- */
/* 默认 = 次级操作：白底描边 */
QPushButton {{
    background-color: {c['panel']};
    color: {c['text_primary']};
    border: 1px solid {c['border_strong']};
    border-radius: {RADIUS_SM}px;
    padding: 5px 14px;
    min-height: {CONTROL_HEIGHT - 12}px;
    font-size: 13px;
}}

QPushButton:hover {{
    background-color: {c['hover']};
}}

QPushButton:pressed {{
    background-color: {c['border']};
}}

QPushButton:disabled {{
    background-color: {c['panel']};
    color: {c['text_disabled']};
    border-color: {c['border']};
}}

/* QPushButton / QToolButton 的菜单箭头不交给平台绘制：macOS 会退化成黑点或空白。
   右侧专用区域固定 34px（图标 16px + 两侧留白），保证文字不会被箭头挤到贴边。 */
QPushButton[menuIndicator="true"], QToolButton[menuIndicator="true"] {{
    padding-right: 34px;
}}

QPushButton[menuIndicator="true"]::menu-indicator,
QToolButton[menuIndicator="true"]::menu-indicator {{
    image: url({chevron_down});
    subcontrol-origin: padding;
    subcontrol-position: right center;
    width: 16px;
    height: 16px;
    right: 9px;
}}

QPushButton#primary[menuIndicator="true"]::menu-indicator {{
    image: url({chevron_down_white});
}}

QPushButton[menuIndicator="true"]::menu-indicator:disabled,
QPushButton#primary[menuIndicator="true"]::menu-indicator:disabled,
QToolButton[menuIndicator="true"]::menu-indicator:disabled {{
    image: url({chevron_down_off});
}}

QPushButton#secondary {{
    background-color: {c['panel']};
    color: {c['text_primary']};
    border: 1px solid {c['border_strong']};
}}

QPushButton#secondary:hover {{
    background-color: {c['hover']};
}}

/* 主操作：每页最多一个 */
QPushButton#primary {{
    background-color: {c['primary']};
    color: #FFFFFF;
    border: 1px solid {c['primary']};
    font-weight: 600;
}}

QPushButton#primary:hover {{
    background-color: {c['primary_hover']};
    border-color: {c['primary_hover']};
}}

QPushButton#primary:pressed {{
    background-color: {c['primary_pressed']};
    border-color: {c['primary_pressed']};
}}

QPushButton#primary:disabled {{
    background-color: {c['track']};
    color: {c['text_disabled']};
    border-color: {c['track']};
}}

QPushButton#ghost {{
    background-color: transparent;
    color: {c['text_secondary']};
    border: 1px solid transparent;
}}

QPushButton#ghost:hover {{
    background-color: {c['hover']};
    color: {c['text_primary']};
}}

QPushButton#ghost:checked {{
    background-color: {c['selected']};
    color: {c['text_primary']};
}}

QPushButton#danger {{
    background-color: {c['panel']};
    color: {c['error']};
    border: 1px solid {c['border_strong']};
}}

QPushButton#danger:hover {{
    background-color: {c['hover']};
    border-color: {c['error_fill']};
}}

QPushButton#danger:disabled, QPushButton#ghost:disabled {{
    color: {c['text_disabled']};
    border-color: {c['border']};
}}

QPushButton#link {{
    background-color: transparent;
    color: {c['accent_text']};
    border: none;
    padding: 2px 4px;
    text-align: left;
}}

QPushButton#link:hover {{
    color: {c['primary']};
}}

QToolButton {{
    background-color: {c['panel']};
    color: {c['text_primary']};
    border: 1px solid {c['border_strong']};
    border-radius: {RADIUS_SM}px;
    padding: 5px 10px;
}}

QToolButton:hover {{
    background-color: {c['hover']};
}}

/* ---------- 输入控件 ---------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
    background-color: {c['inset']};
    color: {c['text_primary']};
    border: 1px solid {c['border_strong']};
    border-radius: {RADIUS_SM}px;
    padding: 5px 10px;
    min-height: {CONTROL_HEIGHT - 12}px;
    selection-background-color: {c['primary']};
    selection-color: #FFFFFF;
}}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QPlainTextEdit:focus {{
    border: 1px solid {c['primary']};
}}

QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {c['background']};
    color: {c['text_disabled']};
    border-color: {c['border']};
}}

/* 箭头区要先占住位置，否则长选项的文字会压到箭头底下 */
QComboBox {{
    padding-right: 34px;
}}

QSpinBox, QDoubleSpinBox {{
    padding-right: 26px;
    min-height: 24px;
}}

/* 下拉框：整块都能点开（专用区固定 34px），所以不给箭头区画边框/底色，
   只在悬停时亮一个浅色圆角块 */
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 34px;
    border: none;
    border-radius: 6px;
    background-color: transparent;
}}

QComboBox::drop-down:hover {{
    background-color: {c['hover']};
}}

QComboBox::down-arrow {{
    image: url({chevron_down});
    width: 14px;
    height: 14px;
}}

QComboBox::down-arrow:disabled {{
    image: url({chevron_down_off});
}}

QComboBox QAbstractItemView {{
    background-color: {c['panel']};
    color: {c['text_primary']};
    border: 1px solid {c['border_strong']};
    border-radius: {RADIUS_SM}px;
    padding: 4px;
    outline: none;
    selection-background-color: {c['selected']};
    selection-color: {c['text_primary']};
}}

/* 行距。注意：macOS 上下拉弹层是系统画的，样式表管不到它——选中行仍是系统那条
   蓝底白字（清楚可读，所以不去动它；强行换成 QListView 反而会让当前项没有任何
   高亮）。这条和上面的 selection-* 是给其他平台的。 */
QComboBox QAbstractItemView::item {{
    padding: 5px 8px;
    border-radius: 6px;
    min-height: 20px;
}}

/* 数字框的步进器：跟下拉框同一套雪佛龙，上下各占边框盒一半高（18px），
   subcontrol-origin 取 border 而不是 padding，两个按钮首尾相接、点击区不缩水。 */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    width: 20px;
    height: 18px;
    border: none;
    border-radius: 4px;
    margin-right: 3px;
    background-color: transparent;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-position: top right;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-position: bottom right;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {c['hover']};
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({chevron_up});
    width: 12px;
    height: 12px;
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({chevron_down});
    width: 12px;
    height: 12px;
}}

/* 到头了 / 整个控件禁用：箭头转灰，不再假装还能点 */
QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off,
QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:off {{
    image: url({chevron_up_off});
}}

QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off,
QDoubleSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:off {{
    image: url({chevron_down_off});
}}

QTextEdit {{
    background-color: {c['inset']};
    color: {c['text_primary']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS_SM}px;
    padding: 6px;
    selection-background-color: {c['primary']};
    selection-color: #FFFFFF;
}}

/* 勾选框和单选钮：框和填充都得自己画。
   试过交给系统画（去掉 ::indicator 规则）——不行：控件一旦被样式表接管，
   Qt 画的对勾/圆点就只剩一个孤零零的字形，外面那个框来自控件继承下来的背景，
   而背景在这里是 transparent（不设成 transparent，白卡片上每一行都会拖一条灰底）。
   结果是「勾上的只有一个勾、没勾的什么都没有」，单选钮更是整个圈都不见。
   所以框自己画，选中态用实心表示——填满 = 选上了，空的 = 没选，不会看错。 */
QCheckBox, QRadioButton {{
    background-color: transparent;
    color: {c['text_primary']};
    spacing: 8px;
    padding: 2px 0;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {c['border_strong']};
    background-color: {c['inset']};
}}

QCheckBox::indicator {{
    border-radius: 4px;
}}

QRadioButton::indicator {{
    border-radius: 8px;
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {c['primary']};
    border-color: {c['primary']};
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {c['primary']};
}}

QSlider::groove:horizontal {{
    background-color: {c['track']};
    height: 4px;
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background-color: {c['primary']};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background-color: #FFFFFF;
    border: 1px solid {c['border_strong']};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 8px;
}}

/* ---------- 列表 ---------- */
QListWidget {{
    background-color: {c['inset']};
    color: {c['text_primary']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS_SM}px;
    padding: 4px;
    outline: none;
}}

QListWidget::item {{
    padding: 6px 10px;
    border-radius: 6px;
    border: none;
}}

QListWidget::item:selected {{
    background-color: {c['selected']};
    color: {c['text_primary']};
}}

QListWidget::item:hover {{
    background-color: {c['hover']};
}}

QMenu {{
    background-color: {c['panel']};
    color: {c['text_primary']};
    border: 1px solid {c['border_strong']};
    border-radius: {RADIUS_SM}px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 20px 6px 12px;
    border-radius: 5px;
}}

QMenu::item:selected {{
    background-color: {c['selected']};
}}

QMenu::separator {{
    height: 1px;
    background-color: {c['border']};
    margin: 4px 8px;
}}

/* 自动标注这类「选一个操作」的菜单（objectName = actionMenu）。
   默认的 QMenu 是给命令列表用的：一行 20 出头像素、裸文字，点开像系统临时弹层，
   而这里每一项的份量差得很远——「设置」只是打开一个窗口，「批量标注…」会跑一整批。
   所以行高抬到 40px、图标 16px 各就各位、文字左对齐；hover 只用浅蓝底，
   靠 1px 边框分层，不加厚阴影。 */
QMenu#actionMenu {{
    background-color: {c['panel']};
    border: 1px solid {c['border_strong']};
    border-radius: {RADIUS_SM}px;
    padding: 6px;
}}

QMenu#actionMenu::item {{
    min-height: 40px;
    padding: 0px 24px 0px 8px;
    border-radius: 6px;
}}

QMenu#actionMenu::item:selected {{
    background-color: {c['selected']};
    color: {c['text_primary']};
}}

QMenu#actionMenu::icon {{
    padding-left: 12px;
}}

/* ---------- 分组框 ---------- */
/* 标题整行落在卡片外面（margin 区），不再骑在边框线上。
   原来 margin-top 只有 14px，标题盒子比它高，边框就从字的下半截穿过去——
   设置页、关于页、批量处理弹窗里每一个分组标题都被划了一道。
   标题移到卡片上方之后，无论卡片底下是什么颜色都不会重叠。
   字重只给标题：写在 QGroupBox 上会连里面的按钮和输入框一起加粗。 */
QGroupBox {{
    background-color: {c['panel']};
    color: {c['text_primary']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS}px;
    margin-top: 26px;
    padding: 16px;
    font-size: 13px;
    font-weight: 400;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 2px;
    top: 0px;
    padding: 0 0 8px 0;
    font-size: 13px;
    font-weight: 600;
    color: {c['text_primary']};
    background-color: transparent;
}}

/* ---------- 进度条 ---------- */
QProgressBar {{
    background-color: {c['track']};
    border: none;
    border-radius: 3px;
    text-align: center;
    color: {c['text_primary']};
    height: 6px;
    font-size: 11px;
}}

QProgressBar::chunk {{
    background-color: {c['primary']};
    border-radius: 3px;
}}

/* ---------- 标签页 ---------- */
QTabWidget::pane {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS_SM}px;
    top: -1px;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {c['text_secondary']};
    padding: 6px 14px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: {RADIUS_SM}px;
    border-top-right-radius: {RADIUS_SM}px;
}}

QTabBar::tab:selected {{
    background-color: {c['panel']};
    color: {c['text_primary']};
    border-color: {c['border']};
    border-bottom-color: {c['panel']};
    font-weight: 600;
}}

QTabBar::tab:hover:!selected {{
    color: {c['text_primary']};
}}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {{
    background-color: transparent;
    width: 10px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background-color: {c['border_strong']};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {c['text_disabled']};
}}

QScrollBar:horizontal {{
    background-color: transparent;
    height: 10px;
    margin: 2px;
}}

QScrollBar::handle:horizontal {{
    background-color: {c['border_strong']};
    border-radius: 5px;
    min-width: 30px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {c['text_disabled']};
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0px;
    width: 0px;
}}

QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}
"""


