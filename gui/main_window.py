# -*- coding: utf-8 -*-
"""
EzYOLO 主窗口

窗口 = 左边一条固定的流程导航 + 右边「页头 + 当前步骤」。
项目的选择放在窗口层（侧边栏顶部），所有页面共用同一个当前项目，
不用在每个页面里再选一次。
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QLabel, QPushButton, QFrame, QComboBox,
)
from PyQt6.QtCore import Qt, QSettings, QSize, QPoint

from gui.styles import get_full_stylesheet, apply_theme_to_app, normalize_theme
from gui.workflow import (
    WORKFLOW_STEPS, STEP_BY_INDEX, PAGE_SETTINGS, PAGE_ABOUT,
    STEP_IMPORT, STEP_ANNOTATE, STEP_TRAIN, STEP_RESULT, STEP_TEST,
    get_project_snapshot, compute_step_states, step_status_text,
    get_blocker, get_next_action,
)
from gui.widgets.workflow_widgets import StepNav, PageHeader, StepGate, NoticeBar
from gui.widgets.elided_combo import ElidedComboBox
from gui.pages.import_page import ImportPage
from gui.pages.annotate_page import AnnotatePage
from gui.pages.train_page import APP_ROOT, TrainPage
from gui.pages.result_page import ResultPage
from gui.pages.test_page import TestPage
from gui.pages.settings_page import SettingsPage, THEME_SETTING_KEY
from gui.pages.about_page import AboutPage
from gui.remote_training_runtime import (
    RemoteTrainingRuntimeError,
    build_system_remote_backend,
    current_remote_training_runtime_paths,
)
from gui.remote_training_thread import RemoteConnectionTestThread
from core.remote_training.transport import ClientTransportUnavailable, RemoteTransportError
from models.database import db

GATE_INDEX = 7  # 前置条件说明页在 content_stack 中的位置

# 导入/标注是业务页：左侧 StepNav 已经承担了流程和下一步导航，PageHeader
# 整块（标题/序号/说明/下一步按钮）在这两页上是和 StepNav 重复的第二套导航，
# 隐藏掉、把页面内容往上提。其余页面（含前置条件不满足时显示的 Gate）
# 仍然显示 PageHeader。
HIDDEN_HEADER_STEPS = {STEP_IMPORT, STEP_ANNOTATE}

AUX_PAGES = {
    PAGE_SETTINGS: ("设置", "模型路径、自动保存与快捷键。"),
    PAGE_ABOUT: ("关于 EzYOLO", ""),
}


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()

        self.settings = QSettings("EzYOLO", "MainWindow")
        self.current_project_id = None
        self.current_index = STEP_IMPORT
        self.snapshot = get_project_snapshot(None)
        self.step_states = compute_step_states(self.snapshot)
        self._bypassed_steps = set()
        self._remote_profile_test_thread = None

        self.init_ui()
        self.load_window_state()

    # ==================== 界面搭建 ====================

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("EzYOLO - 本地YOLO训练全流程")
        self.setMinimumSize(1100, 720)

        saved_theme = normalize_theme(
            QSettings("EzYOLO", "Settings").value(THEME_SETTING_KEY, 'light')
        )
        apply_theme_to_app(saved_theme)
        self.setStyleSheet(get_full_stylesheet())

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 先建内容区（页面在这里创建），再建侧边栏，侧边栏要连到导入页
        content = self.create_content_area()

        self.sidebar = self.create_sidebar()
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(content, 1)

        self.settings_page.theme_changed.connect(self.on_theme_changed)
        self.settings_page.auto_label_config_requested.connect(self.open_auto_label_config)
        self.settings_page.remote_profiles_changed.connect(
            self.train_page.refresh_remote_targets
        )
        self.settings_page.remote_profile_test_requested.connect(
            self.start_remote_profile_test
        )
        self.import_page.projects_changed.connect(self.on_projects_changed)
        self.import_page.project_data_changed.connect(self.refresh_workflow)
        self.import_page.display_name_rule_changed.connect(
            self.annotate_page.on_display_name_rule_changed
        )

        # 启动时同步数据库与真实文件
        self.sync_database_files()

        self.load_projects()
        self.switch_page(STEP_IMPORT)

    def create_sidebar(self) -> QWidget:
        """创建侧边栏：品牌 → 当前项目 → 流程步骤 → 辅助入口"""
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(216)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(0)

        brand = QLabel("EzYOLO")
        brand.setObjectName("brand")
        layout.addWidget(brand)

        layout.addSpacing(16)

        # 当前项目：全窗口共用，切换项目后所有步骤跟着走
        project_label = QLabel("项目")
        project_label.setObjectName("nav_section")
        layout.addWidget(project_label)
        layout.addSpacing(6)

        # 项目名可以很长，下拉框不能被它撑出侧栏；装不下时要省略号收尾，
        # 不能像原生 QComboBox 那样把最后一个中文字切掉一半
        self.project_combo = ElidedComboBox()
        self.project_combo.setToolTip("图片、标注和训练结果都存在当前项目里")
        self.project_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.project_combo.setMinimumContentsLength(8)
        self.project_combo.currentIndexChanged.connect(self.on_project_combo_changed)
        layout.addWidget(self.project_combo)

        layout.addSpacing(6)

        self.btn_new_project = QPushButton("新建项目")
        self.btn_new_project.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_new_project.clicked.connect(self.import_page.create_new_project)
        layout.addWidget(self.btn_new_project)

        layout.addSpacing(18)

        flow_label = QLabel("流程")
        flow_label.setObjectName("nav_section")
        layout.addWidget(flow_label)
        layout.addSpacing(6)

        self.step_nav = StepNav()
        self.step_nav.step_clicked.connect(self.switch_page)
        layout.addWidget(self.step_nav)

        layout.addStretch()

        divider = QFrame()
        divider.setObjectName("divider")
        layout.addWidget(divider)
        layout.addSpacing(8)

        aux_row = QHBoxLayout()
        aux_row.setContentsMargins(0, 0, 0, 0)
        aux_row.setSpacing(6)

        self.btn_settings = QPushButton("设置")
        self.btn_settings.setObjectName("ghost")
        self.btn_settings.setCheckable(True)
        self.btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_settings.clicked.connect(lambda: self.switch_page(PAGE_SETTINGS))
        aux_row.addWidget(self.btn_settings)

        self.btn_about = QPushButton("关于")
        self.btn_about.setObjectName("ghost")
        self.btn_about.setCheckable(True)
        self.btn_about.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_about.clicked.connect(lambda: self.switch_page(PAGE_ABOUT))
        aux_row.addWidget(self.btn_about)

        layout.addLayout(aux_row)

        return sidebar

    def create_content_area(self) -> QWidget:
        """创建内容区：页头 + 页面堆栈"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = PageHeader()
        self.header.next_clicked.connect(self.switch_page)
        layout.addWidget(self.header)

        # 启动自检这类「说一声就行」的消息落在这里，默认不占位
        self.notice = NoticeBar()
        layout.addWidget(self.notice)

        self.content_stack = QStackedWidget()

        self.import_page = ImportPage()
        self.content_stack.addWidget(self.import_page)          # 0

        self.annotate_page = AnnotatePage()
        self.content_stack.addWidget(self.annotate_page)        # 1

        self.train_page = TrainPage()
        self.content_stack.addWidget(self.train_page)           # 2

        self.result_page = ResultPage()
        self.content_stack.addWidget(self.result_page)          # 3

        self.test_page = TestPage()
        self.content_stack.addWidget(self.test_page)            # 4

        self.settings_page = SettingsPage()
        self.content_stack.addWidget(self.settings_page)        # 5

        self.about_page = AboutPage()
        self.content_stack.addWidget(self.about_page)           # 6

        self.gate = StepGate()
        self.gate.goto_requested.connect(self.switch_page)
        self.gate.bypass_requested.connect(self.on_gate_bypassed)
        self.content_stack.addWidget(self.gate)                 # 7

        layout.addWidget(self.content_stack, 1)
        return container

    # ==================== 项目 ====================

    def load_projects(self, select_id: int = None):
        """把项目列表灌进侧边栏下拉框。"""
        self.project_combo.blockSignals(True)
        self.project_combo.clear()

        projects = db.get_all_projects()
        if projects:
            self.project_combo.addItem("请选择项目…", None)
            for project in projects:
                self.project_combo.addItem(project['name'], project['id'])
                # 名字长到要省略时，完整名字至少还能在悬停里看到
                self.project_combo.setItemData(
                    self.project_combo.count() - 1,
                    project['name'],
                    Qt.ItemDataRole.ToolTipRole,
                )
        else:
            self.project_combo.addItem("还没有项目", None)

        target = select_id if select_id is not None else self.current_project_id
        index = self.project_combo.findData(target) if target else -1
        self.project_combo.setCurrentIndex(index if index >= 0 else 0)
        self.project_combo.blockSignals(False)

        self.set_active_project(self.project_combo.currentData())

    def on_project_combo_changed(self, _index: int):
        self.set_active_project(self.project_combo.currentData())

    def set_active_project(self, project_id):
        """切换当前项目：导入页立刻跟着换，其余页面在进入时同步。"""
        self.current_project_id = project_id
        self._bypassed_steps.clear()
        self.import_page.set_project(project_id)
        self.refresh_workflow()
        if self.current_index != STEP_IMPORT:
            self.apply_page_project(self.current_index)

    def on_projects_changed(self, project_id):
        """导入页新建 / 删除项目后，刷新下拉框。"""
        self.load_projects(select_id=project_id)

    # ==================== 页面切换 ====================

    def switch_page(self, index: int):
        """切换到某一步。前置条件不满足时不再弹窗，而是就地说明原因并给出入口。"""
        self.refresh_workflow(update_views=False)
        self.current_index = index

        blocker = None
        if index not in self._bypassed_steps:
            blocker = get_blocker(index, self.snapshot)

        if blocker:
            self.gate.set_blocker(blocker)
            self.content_stack.setCurrentIndex(GATE_INDEX)
        else:
            self.apply_page_project(index)
            self.content_stack.setCurrentIndex(index)

        self.update_header(index)
        self.update_nav(index)

    def apply_page_project(self, index: int):
        """把当前项目同步给将要显示的页面。"""
        if index == STEP_IMPORT:
            # 别的步骤可能改过标注状态，回到导入页时对一下
            self.import_page.refresh_project_images()
            return

        page_setters = {
            STEP_ANNOTATE: self.annotate_page.set_project,
            STEP_TRAIN: self.train_page.set_project,
            STEP_RESULT: self.result_page.set_project,
            STEP_TEST: self.test_page.set_project,
        }
        setter = page_setters.get(index)
        if setter and self.current_project_id:
            setter(self.current_project_id)

    def open_auto_label_config(self, section: str = ""):
        """设置页点「打开自动标注配置」→ 就在这儿把配置窗口开出来。

        配置窗口是标注页那一个实例（AutoLabelDialog），但用户不用先切到标注页、
        再去工具栏里找按钮——设置页发信号，窗口层直接开。SAM / LLM 的配置是
        全局文件，没有项目也能配。

        关窗之后回到「打开它的那个页面」，而不是跳去别的步骤：从设置页点开的，
        保存完还留在设置页，并且就地把 AI 状态刷新成刚存的值 + 给一句「已保存」——
        否则用户点完保存，窗口一关，界面上没有任何东西变过，跟没保存一样。
        """
        origin_index = self.current_index

        saved = self.annotate_page.open_auto_label_config(section)

        # 配置窗口开着的这段时间里页面不该被换掉；真被换了（以后有人加了新逻辑）
        # 也要回到用户点开配置的那一页，不能把人甩到别的步骤去。
        if self.current_index != origin_index:
            self.switch_page(origin_index)

        self.settings_page.refresh_ai_status()

        if origin_index == PAGE_SETTINGS:
            if saved:
                self.settings_page.set_status("自动标注设置已保存。", 'success')
            else:
                self.settings_page.set_status("没有改动自动标注设置。")

    def start_remote_profile_test(self, profile) -> None:
        """设置页的「测试连接」只做 preflight，网络 I/O 始终放到后台线程。"""
        if (
            self._remote_profile_test_thread is not None
            and self._remote_profile_test_thread.isRunning()
        ):
            self.settings_page.set_remote_profile_test_status(
                "已有连接预检正在进行，请稍候。",
                success=None,
            )
            return
        try:
            paths = current_remote_training_runtime_paths(APP_ROOT)
            backend = build_system_remote_backend(paths)
        except (RemoteTrainingRuntimeError, ClientTransportUnavailable, RemoteTransportError) as exc:
            self.settings_page.set_remote_profile_test_status(str(exc), success=False)
            return

        thread = RemoteConnectionTestThread(profile=profile, backend=backend)
        self._remote_profile_test_thread = thread
        thread.preflight_finished.connect(self.on_remote_profile_test_finished)
        thread.finished.connect(self._clear_remote_profile_test_thread)
        thread.start()

    def on_remote_profile_test_finished(self, success: bool, message: str, _capabilities) -> None:
        self.settings_page.set_remote_profile_test_status(message, success=success)

    def _clear_remote_profile_test_thread(self) -> None:
        self._remote_profile_test_thread = None

    def on_gate_bypassed(self):
        """用户选择「我有现成的模型，直接测试」这类跳过。"""
        self._bypassed_steps.add(self.current_index)
        self.switch_page(self.current_index)

    # ==================== 流程状态 ====================

    def refresh_workflow(self, update_views: bool = True):
        """重新读取项目事实（图片数、标注数、训练结果），刷新侧边栏与页头。"""
        self.snapshot = get_project_snapshot(self.current_project_id)
        self.step_states = compute_step_states(self.snapshot)
        if update_views:
            self.update_nav(self.current_index)
            self.update_header(self.current_index)

    def update_nav(self, index: int):
        status_texts = {
            step['index']: step_status_text(step['index'], self.snapshot, self.step_states)
            for step in WORKFLOW_STEPS
        }
        # 停留在设置/关于时，主流程不高亮任何一步
        highlight = -1 if index in AUX_PAGES else index
        self.step_nav.update_states(self.step_states, status_texts, highlight)

        self.btn_settings.setChecked(index == PAGE_SETTINGS)
        self.btn_about.setChecked(index == PAGE_ABOUT)

    def update_header(self, index: int):
        """页头数据永远按「请求的那一步」来算，不受 Gate 遮挡影响；
        只有显示与否才看 content_stack 实际显示的是业务页还是 Gate。
        """
        if index in AUX_PAGES:
            title, desc = AUX_PAGES[index]
            self.header.set_step(index, title, desc)
            self.header.set_next_action(None)
        elif index in STEP_BY_INDEX:
            self.header.set_step(index)
            self.header.set_next_action(
                get_next_action(self.snapshot, self.step_states, index)
            )
        else:
            return

        self.header.setVisible(self.content_stack.currentIndex() not in HIDDEN_HEADER_STEPS)

    # ==================== 窗口状态 ====================

    def load_window_state(self):
        """加载窗口状态"""
        size = self.settings.value("size", QSize(1440, 900))
        self.resize(size)

        pos = self.settings.value("pos", QPoint(100, 100))
        self.move(pos)

        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)

    def save_window_state(self):
        """保存窗口状态"""
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())
        self.settings.setValue("windowState", self.saveState())

    def on_theme_changed(self, theme_key=None):
        """主题变化：刷新全局样式表，并让各页面重套内联颜色。"""
        apply_theme_to_app(theme_key)
        self.setStyleSheet(get_full_stylesheet())

        for page in (
            self.import_page,
            self.annotate_page,
            self.train_page,
            self.result_page,
            self.test_page,
            self.settings_page,
        ):
            refresh = getattr(page, 'refresh_theme', None)
            if callable(refresh):
                refresh()

    def closeEvent(self, event):
        """关闭事件：后台线程先停干净，再让窗口销毁。

        QThread 还在 run() 里时被销毁会直接崩，所以推理和 LLM 批量这两条
        长任务在这里各自收工（都带超时，不会把退出流程拖住）。
        """
        # 训练线程不在主线程里做网络/训练 I/O。不能为了退出窗口而粗暴 terminate：
        # 本机训练需要收尾，远程训练还必须等 runner 确认取消，避免界面把未确认的
        # 服务器任务误说成已经停止。
        if not self.train_page.request_close():
            self.notice.show_message(
                "训练仍在安全收尾，窗口暂不关闭。",
                "远程训练会等待服务器确认停止；本机训练会等待当前后台线程结束。"
                "训练结束后请再关闭窗口。",
            )
            event.ignore()
            return

        # 连接预检同样是 QThread。预检不支持强杀；它只含带超时的只读 SSH 请求，
        # 等结果返回后重新关闭即可，不能销毁仍在运行的线程对象。
        if (
            self._remote_profile_test_thread is not None
            and self._remote_profile_test_thread.isRunning()
        ):
            self.notice.show_message(
                "远程连接预检仍在进行，窗口暂不关闭。",
                "预检只读取服务器能力，不上传数据；结束后请再关闭窗口。",
            )
            event.ignore()
            return

        for page in (self.test_page, self.annotate_page):
            shutdown = getattr(page, 'shutdown', None)
            if shutdown:
                shutdown()

        self.save_window_state()
        event.accept()

    def sync_database_files(self):
        """启动自检：只看数据库和磁盘对不对得上，不动任何数据。

        数据层只扫描、只报数（orphan_db_count / orphan_disk_count / issues），
        删不删由用户自己决定。所以这里也只是「说一声」：
        对得上就什么都不显示；对不上就在窗口里留一条可关掉的通知，
        不弹窗、不阻塞启动，更不会替用户删东西。
        """
        try:
            result = db.sync_files_with_database() or {}
        except Exception as exc:  # noqa: BLE001  自检失败不该拦住启动
            print(f"[自检] 跳过：{exc}")
            return

        orphan_db = result.get('orphan_db_count', 0)
        orphan_disk = result.get('orphan_disk_count', 0)
        if not (orphan_db or orphan_disk):
            return

        parts = []
        if orphan_db:
            parts.append(f"{orphan_db} 条记录找不到对应文件")
        if orphan_disk:
            parts.append(f"{orphan_disk} 个文件不在数据库里")

        issues = result.get('issues') or []
        detail = "\n".join(str(item) for item in issues[:20])
        self.notice.show_message(
            "数据自检：" + "，".join(parts) + "。未做任何改动。",
            detail,
        )
