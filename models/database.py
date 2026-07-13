# -*- coding: utf-8 -*-
"""
数据库管理模块
使用SQLite作为本地数据库
"""

import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from contextlib import contextmanager


class Database:
    """数据库管理类"""
    
    def __init__(self, db_path: str = None):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径，默认为项目目录下的data/EzYOLO.db
        """
        if db_path is None:
            # 默认存储在软件所在目录
            current_dir = Path(__file__).parent.parent
            data_dir = current_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "EzYOLO.db")
        else:
            self.db_path = db_path
        
        # 初始化数据库
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 创建项目表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    type TEXT DEFAULT 'detection',
                    classes TEXT DEFAULT '[]',
                    status TEXT DEFAULT 'created',
                    storage_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建数据集表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT DEFAULT 'train',
                    image_count INTEGER DEFAULT 0,
                    annotation_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)
            
            # 创建图像表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    dataset_id INTEGER,
                    filename TEXT NOT NULL,
                    original_path TEXT,
                    storage_path TEXT,
                    width INTEGER,
                    height INTEGER,
                    size INTEGER,
                    format TEXT,
                    status TEXT DEFAULT 'pending',
                    annotated_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE SET NULL
                )
            """)
            
            # 创建标注表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    project_id INTEGER NOT NULL,
                    class_id INTEGER DEFAULT 0,
                    class_name TEXT,
                    type TEXT DEFAULT 'bbox',
                    data TEXT NOT NULL,
                    attributes TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)
            
            # 创建训练任务表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    model_version TEXT DEFAULT 'v8',
                    model_type TEXT DEFAULT 'n',
                    task_type TEXT DEFAULT 'detect',
                    config TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'pending',
                    progress INTEGER DEFAULT 0,
                    current_epoch INTEGER DEFAULT 0,
                    total_epochs INTEGER DEFAULT 100,
                    metrics TEXT DEFAULT '{}',
                    weights_path TEXT,
                    log_path TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)
            
            # 创建训练指标历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    metrics TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_project_status ON images(project_id, status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotations_image ON annotations(image_id)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_annotations_project_class_image "
                "ON annotations(project_id, class_id, image_id)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_project ON training_jobs(project_id)")
            
            self._migrate_database(cursor)
            
            conn.commit()

    def _migrate_database(self, cursor):
        """数据库结构迁移（兼容已有数据库）"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                UNIQUE(project_id, name)
            )
        """)

        cursor.execute("PRAGMA table_info(images)")
        image_columns = {row[1] for row in cursor.fetchall()}
        if 'group_id' not in image_columns:
            cursor.execute("""
                ALTER TABLE images ADD COLUMN group_id INTEGER
                REFERENCES image_groups(id) ON DELETE SET NULL
            """)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_group ON images(project_id, group_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_image_groups_project ON image_groups(project_id)"
        )

        cursor.execute("PRAGMA table_info(projects)")
        project_columns = {row[1] for row in cursor.fetchall()}
        if 'display_name_rule' not in project_columns:
            # 只加列，不填默认值：NULL 就表示「没设置过规则」，
            # gui.display_names.parse_display_name_rule(None) 会把它当成 original 处理，
            # 旧项目的显示行为不会因为升级数据库结构而改变。
            cursor.execute("ALTER TABLE projects ADD COLUMN display_name_rule TEXT")
    
    # ==================== 项目操作 ====================
    
    def create_project(self, name: str, description: str = "", 
                       project_type: str = "detection", 
                       classes: List[Dict] = None) -> int:
        """
        创建新项目
        
        Args:
            name: 项目名称
            description: 项目描述
            project_type: 项目类型 (detection/segmentation/classification)
            classes: 类别列表 [{"id": 0, "name": "person", "color": "#FF0000"}]
            
        Returns:
            项目ID
        """
        if classes is None:
            classes = []
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 项目存储路径也放在软件所在目录
            current_dir = Path(__file__).parent.parent
            storage_path = current_dir / "projects" / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            storage_path.mkdir(parents=True, exist_ok=True)
            
            cursor.execute("""
                INSERT INTO projects (name, description, type, classes, storage_path)
                VALUES (?, ?, ?, ?, ?)
            """, (
                name, 
                description, 
                project_type, 
                json.dumps(classes, ensure_ascii=False),
                str(storage_path)
            ))
            return cursor.lastrowid
    
    def get_project(self, project_id: int) -> Optional[Dict]:
        """获取项目信息"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_all_projects(self) -> List[Dict]:
        """获取所有项目"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects ORDER BY updated_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def update_project(self, project_id: int, **kwargs) -> bool:
        """更新项目信息"""
        allowed_fields = [
            'name', 'description', 'type', 'classes', 'status', 'storage_path',
            'display_name_rule',
        ]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return False

        # 处理classes字段
        if 'classes' in updates and isinstance(updates['classes'], list):
            updates['classes'] = json.dumps(updates['classes'], ensure_ascii=False)

        # display_name_rule 存的是 JSON 字符串；传字典进来时顺手序列化，
        # 调用方也可以自己先用 serialize_display_name_rule 转好再传字符串。
        if 'display_name_rule' in updates and isinstance(updates['display_name_rule'], dict):
            updates['display_name_rule'] = json.dumps(
                updates['display_name_rule'], ensure_ascii=False
            )
        
        updates['updated_at'] = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [project_id]
            cursor.execute(f"UPDATE projects SET {set_clause} WHERE id = ?", values)
            return cursor.rowcount > 0
    
    def delete_project(self, project_id: int) -> bool:
        """删除项目"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            return cursor.rowcount > 0
    
    # ==================== 图像操作 ====================
    
    def add_image(self, project_id: int, filename: str, storage_path: str,
                  width: int = None, height: int = None, size: int = None,
                  image_format: str = None, original_path: str = None,
                  dataset_id: int = None, group_id: int = None) -> int:
        """添加图像记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO images 
                (project_id, dataset_id, group_id, filename, original_path, storage_path, 
                 width, height, size, format)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (project_id, dataset_id, group_id, filename, original_path, storage_path,
                  width, height, size, image_format))
            return cursor.lastrowid
    
    def get_project_images(self, project_id: int, status: str = None,
                           group_id: int = None, ungrouped_only: bool = False) -> List[Dict]:
        """获取项目下的所有图像

        Args:
            group_id: 指定分组 ID；与 ungrouped_only 互斥
            ungrouped_only: 仅返回未分组图片（group_id IS NULL）
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM images WHERE project_id = ?"
            params: List[Any] = [project_id]

            if status:
                query += " AND status = ?"
                params.append(status)
            if ungrouped_only:
                query += " AND group_id IS NULL"
            elif group_id is not None:
                query += " AND group_id = ?"
                params.append(group_id)

            query += " ORDER BY created_at, id"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_project_images_by_groups(self, project_id: int,
                                     group_ids: List[int]) -> List[Dict]:
        """按多个分组获取图片。group_ids 中 0 表示未分组（group_id IS NULL）。"""
        if not group_ids:
            return []

        conditions = []
        params: List[Any] = [project_id]
        normal_ids = [gid for gid in group_ids if gid != 0]
        include_ungrouped = 0 in group_ids

        if normal_ids:
            placeholders = ", ".join("?" * len(normal_ids))
            conditions.append(f"group_id IN ({placeholders})")
            params.extend(normal_ids)
        if include_ungrouped:
            conditions.append("group_id IS NULL")

        where_clause = " OR ".join(conditions)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM images WHERE project_id = ? AND ({where_clause}) "
                "ORDER BY created_at, id",
                params
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_project_images_by_class(self, project_id: int, class_id: int) -> List[Dict]:
        """获取项目下包含指定类别的图像（按图片去重）"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT images.*
                FROM images
                INNER JOIN annotations ON images.id = annotations.image_id
                WHERE images.project_id = ? AND annotations.class_id = ?
                ORDER BY images.created_at, images.id
            """, (project_id, class_id))
            return [dict(row) for row in cursor.fetchall()]

    def get_project_image_counts_by_class(self, project_id: int) -> Dict[int, int]:
        """按类别统计项目中包含该类别的图片数量（按图片去重）"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT class_id, COUNT(DISTINCT image_id) AS image_count
                FROM annotations
                WHERE project_id = ?
                GROUP BY class_id
            """, (project_id,))
            return {row['class_id']: row['image_count'] for row in cursor.fetchall()}

    def get_negative_sample_images(self, project_id: int, annotated_only: bool = True) -> List[Dict]:
        """获取项目下的负样本图像（已标注但无任何标注框）"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT images.*
                FROM images
                LEFT JOIN annotations ON images.id = annotations.image_id
                WHERE images.project_id = ? AND annotations.id IS NULL
            """
            params = [project_id]
            if annotated_only:
                query += " AND images.status = ?"
                params.append('annotated')
            query += " ORDER BY images.created_at, images.id"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_negative_sample_image_count(self, project_id: int, annotated_only: bool = True) -> int:
        """获取项目下负样本图像数量"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT COUNT(*)
                FROM images
                LEFT JOIN annotations ON images.id = annotations.image_id
                WHERE images.project_id = ? AND annotations.id IS NULL
            """
            params = [project_id]
            if annotated_only:
                query += " AND images.status = ?"
                params.append('annotated')
            cursor.execute(query, params)
            row = cursor.fetchone()
            return row[0] if row else 0

    def get_all_images(self) -> List[Dict]:
        """获取所有图像记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM images ORDER BY created_at")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_image(self, image_id: int) -> Optional[Dict]:
        """获取单个图像信息"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM images WHERE id = ?",
                (image_id,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def delete_image_annotations(self, image_id: int) -> bool:
        """删除图像的所有标注"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM annotations WHERE image_id = ?",
                (image_id,)
            )
            return cursor.rowcount > 0
    
    def update_image_status(self, image_id: int, status: str) -> bool:
        """更新图像状态"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            annotated_at = datetime.now().isoformat() if status == 'annotated' else None
            cursor.execute(
                "UPDATE images SET status = ?, annotated_at = ? WHERE id = ?",
                (status, annotated_at, image_id)
            )
            return cursor.rowcount > 0
    
    def assign_images_to_group(self, image_ids: List[int],
                               group_id: Optional[int] = None) -> int:
        """批量设置图片分组。group_id=None 表示移出分组。"""
        if not image_ids:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ", ".join("?" * len(image_ids))
            cursor.execute(
                f"UPDATE images SET group_id = ? WHERE id IN ({placeholders})",
                [group_id, *image_ids]
            )
            return cursor.rowcount

    # ==================== 分组操作 ====================

    def create_image_group(self, project_id: int, name: str) -> int:
        """创建图片分组，同项目内名称不可重复。"""
        name = name.strip()
        if not name:
            raise ValueError("分组名称不能为空")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM image_groups WHERE project_id = ? AND name = ?",
                (project_id, name)
            )
            existing = cursor.fetchone()
            if existing:
                return existing['id']

            cursor.execute(
                "INSERT INTO image_groups (project_id, name) VALUES (?, ?)",
                (project_id, name)
            )
            return cursor.lastrowid

    def get_project_image_groups(self, project_id: int) -> List[Dict]:
        """获取项目下所有分组。"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM image_groups WHERE project_id = ? ORDER BY name",
                (project_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_image_group(self, group_id: int) -> Optional[Dict]:
        """获取单个分组信息。"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM image_groups WHERE id = ?", (group_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def rename_image_group(self, group_id: int, name: str) -> bool:
        """重命名分组。"""
        name = name.strip()
        if not name:
            return False

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT project_id FROM image_groups WHERE id = ?", (group_id,))
            row = cursor.fetchone()
            if not row:
                return False

            cursor.execute(
                "SELECT id FROM image_groups WHERE project_id = ? AND name = ? AND id != ?",
                (row['project_id'], name, group_id)
            )
            if cursor.fetchone():
                raise ValueError(f"分组名称“{name}”已存在")

            cursor.execute(
                "UPDATE image_groups SET name = ? WHERE id = ?",
                (name, group_id)
            )
            return cursor.rowcount > 0

    def delete_image_group(self, group_id: int) -> bool:
        """删除分组，组内图片变为未分组。"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM image_groups WHERE id = ?", (group_id,))
            return cursor.rowcount > 0

    def get_group_image_counts(self, project_id: int) -> Dict[Optional[int], int]:
        """统计各分组图片数量。键 None 表示未分组。"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT group_id, COUNT(*) AS cnt FROM images "
                "WHERE project_id = ? GROUP BY group_id",
                (project_id,)
            )
            return {row['group_id']: row['cnt'] for row in cursor.fetchall()}

    def delete_image(self, image_id: int) -> bool:
        """删除图像"""
        import os
        
        # 先获取图像的存储路径
        storage_path = None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT storage_path FROM images WHERE id = ?", (image_id,))
            row = cursor.fetchone()
            if row:
                storage_path = row['storage_path']
        
        # 删除实际文件（如果存在）
        if storage_path and os.path.exists(storage_path):
            try:
                os.remove(storage_path)
            except Exception:
                pass  # 文件删除失败不影响数据库操作
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 首先删除相关的标注
            cursor.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
            
            # 然后删除图像记录
            cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
            
            return cursor.rowcount > 0
    
    # ==================== 标注操作 ====================
    
    def add_annotation(self, image_id: int, project_id: int, class_id: int,
                       class_name: str, annotation_type: str, data: Dict,
                       attributes: Dict = None) -> int:
        """
        添加标注
        
        Args:
            image_id: 图像ID
            project_id: 项目ID
            class_id: 类别ID
            class_name: 类别名称
            annotation_type: 标注类型 (bbox/polygon/keypoint)
            data: 标注数据，如 {"x": 10, "y": 20, "width": 100, "height": 100}
            attributes: 额外属性
        """
        if attributes is None:
            attributes = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO annotations 
                (image_id, project_id, class_id, class_name, type, data, attributes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (image_id, project_id, class_id, class_name, annotation_type,
                  json.dumps(data), json.dumps(attributes)))
            
            # 更新图像状态
            cursor.execute(
                "UPDATE images SET status = 'annotated', annotated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), image_id)
            )
            
            return cursor.lastrowid
    
    def update_annotation(
        self,
        annotation_id: int,
        data: Dict = None,
        class_id: int = None,
        class_name: str = None
    ) -> bool:
        """
        更新标注
        
        Args:
            annotation_id: 标注ID
            data: 标注数据
            class_id: 类别ID
            class_name: 类别名称
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = []
            values = []
            
            if data is not None:
                updates.append("data = ?")
                values.append(json.dumps(data))
            
            if class_id is not None:
                updates.append("class_id = ?")
                values.append(class_id)

            if class_name is not None:
                updates.append("class_name = ?")
                values.append(class_name)
            
            # 总是更新updated_at时间戳
            updates.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            
            if not updates:
                return False
            
            values.append(annotation_id)
            
            cursor.execute(
                f"UPDATE annotations SET {', '.join(updates)} WHERE id = ?",
                values
            )
            return cursor.rowcount > 0
    
    def get_image_annotations(self, image_id: int) -> List[Dict]:
        """获取图像的所有标注"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM annotations WHERE image_id = ?", (image_id,))
            rows = cursor.fetchall()
            annotations = []
            for row in rows:
                ann = dict(row)
                ann['data'] = json.loads(ann['data'])
                ann['attributes'] = json.loads(ann['attributes'])
                annotations.append(ann)
            return annotations
    
    def _read_bbox_previews(self, cursor, project_id: int) -> Dict[int, List[Dict]]:
        """在给定 cursor 上读框。调用方负责事务边界。"""
        cursor.execute("""
            SELECT image_id, class_id, data
            FROM annotations
            WHERE project_id = ? AND type = 'bbox'
            ORDER BY image_id, id
        """, (project_id,))

        previews: Dict[int, List[Dict]] = {}
        for row in cursor.fetchall():
            try:
                data = json.loads(row['data'])
            except (TypeError, ValueError):
                continue  # 坏掉的一条标注不该让整页缩略图画不出来
            previews.setdefault(row['image_id'], []).append({
                'type': 'bbox',
                'class_id': row['class_id'],
                'data': data,
            })
        return previews

    def _read_annotation_versions(self, cursor, project_id: int) -> Dict[int, Tuple[int, int, str]]:
        """在给定 cursor 上读版本号。调用方负责事务边界。"""
        cursor.execute("""
            SELECT image_id,
                   COUNT(*) AS box_count,
                   MAX(id) AS max_id,
                   MAX(COALESCE(updated_at, created_at)) AS last_changed
            FROM annotations
            WHERE project_id = ?
            GROUP BY image_id
        """, (project_id,))
        return {
            row['image_id']: (row['box_count'], row['max_id'], row['last_changed'])
            for row in cursor.fetchall()
        }

    def get_project_bbox_previews(self, project_id: int) -> Dict[int, List[Dict]]:
        """整个项目的 bbox，一次查完，按 image_id 分好组。

        导入页要在几百张缩略图上画框。逐图调 get_image_annotations 就是典型的
        N+1：600 张图 = 600 次查询，翻页时全压在主线程上。这里一次查回来，
        而且只取画框用得上的三列，不去读 attributes 那些用不到的字段。

        只返回 bbox；缩略图预览不画多边形和关键点。
        """
        with self.get_connection() as conn:
            return self._read_bbox_previews(conn.cursor(), project_id)

    def get_project_annotation_versions(self, project_id: int) -> Dict[int, Tuple[int, int, str]]:
        """每张图的标注版本号：(框数, 最大标注 id, 最近改动时间)，同样只查一次。

        版本号是给缩略图缓存用的——「这张图的标注变了没有」。三个字段缺一不可：

          框数        加框、删框
          最大 id     删一个再加一个：框数没变，但新标注的 id 一定更大
          改动时间    框被拖动 / 改类别：走 update_annotation，updated_at 会刷新

        只看框数会漏掉后两种，缩略图就会一直停在旧的框上；只看 status 更糟，
        改完框图片还是 annotated，界面永远不刷新。
        """
        with self.get_connection() as conn:
            return self._read_annotation_versions(conn.cursor(), project_id)

    def get_project_bbox_preview_snapshot(
        self, project_id: int
    ) -> Tuple[Dict[int, List[Dict]], Dict[int, Tuple[int, int, str]]]:
        """框和版本号必须来自同一个读快照，所以只能一起读。

        分两次调 get_project_bbox_previews / get_project_annotation_versions 会各开
        一条连接、各拿一个快照：sqlite 默认对 SELECT 不开事务。中间只要有人挪了一
        个框，读到的就是「旧框 + 新版本号」——缩略图按新版本号缓存住旧框，从此再也
        不会失效，框在界面上永远停在旧位置。

        显式 BEGIN 把两条 SELECT 圈进同一个读事务：第一条 SELECT 定下快照，第二条
        看到的还是它。旧框配旧版本号，下一轮刷新照样能发现版本变了、把它换掉。
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN")  # 没有它，两条 SELECT 各自 autocommit，各看各的库
            return (
                self._read_bbox_previews(cursor, project_id),
                self._read_annotation_versions(cursor, project_id),
            )

    def delete_annotation(self, annotation_id: int) -> bool:
        """删除标注"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
            return cursor.rowcount > 0
    
    # ==================== 训练任务操作 ====================
    
    def create_training_job(self, project_id: int, name: str,
                           model_version: str = 'v8', model_type: str = 'n',
                           task_type: str = 'detect', config: Dict = None) -> int:
        """创建训练任务"""
        if config is None:
            config = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_jobs 
                (project_id, name, model_version, model_type, task_type, config)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (project_id, name, model_version, model_type, task_type,
                  json.dumps(config)))
            return cursor.lastrowid
    
    def update_training_status(self, job_id: int, status: str,
                               progress: int = None, metrics: Dict = None) -> bool:
        """更新训练状态"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = ["status = ?"]
            values = [status]
            
            if progress is not None:
                updates.append("progress = ?")
                values.append(progress)
            
            if metrics is not None:
                updates.append("metrics = ?")
                values.append(json.dumps(metrics))
            
            if status == 'running' and progress == 0:
                updates.append("started_at = ?")
                values.append(datetime.now().isoformat())
            elif status in ['completed', 'failed']:
                updates.append("completed_at = ?")
                values.append(datetime.now().isoformat())
            
            values.append(job_id)
            
            cursor.execute(
                f"UPDATE training_jobs SET {', '.join(updates)} WHERE id = ?",
                values
            )
            return cursor.rowcount > 0
    
    def get_training_jobs(self, project_id: int) -> List[Dict]:
        """获取项目的训练任务"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM training_jobs WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,)
            )
            rows = cursor.fetchall()
            jobs = []
            for row in rows:
                job = dict(row)
                job['config'] = json.loads(job['config'])
                job['metrics'] = json.loads(job['metrics'])
                jobs.append(job)
            return jobs

    def sync_files_with_database(self, projects_dir: str = None) -> Dict:
        """扫描数据库与 projects/ 目录之间的差异

        只读扫描，绝不删除任何数据库记录或磁盘文件。第一阶段的数据保护要求
        任何不一致都必须交给用户查看后再决定，而不是自动清理。

        Args:
            projects_dir: 仅供测试使用，指定要扫描的 projects 目录；
                默认为软件所在目录下的 projects/。

        Returns:
            Dict: 扫描结果。deleted_db_count/deleted_file_count/total_deleted
                为兼容旧字段，永远为 0；orphan_db_count/orphan_disk_count/issues
                描述发现的不一致，供 UI 提示用户。
        """
        def _normalize(raw_path: str) -> str:
            return str(Path(raw_path).expanduser().resolve(strict=False))

        issues = []

        orphan_db_count = 0
        db_files = set()
        try:
            images = self.get_all_images()
        except Exception as exc:
            images = []
            issues.append({'type': 'scan_error', 'detail': str(exc)})

        for image in images:
            storage_path = image.get('storage_path')
            if not storage_path:
                continue
            try:
                normalized_image_path = _normalize(storage_path)
            except Exception as exc:
                issues.append({'type': 'scan_error', 'detail': str(exc), 'path': storage_path})
                continue
            db_files.add(normalized_image_path)
            try:
                exists = os.path.exists(storage_path)
            except Exception as exc:
                issues.append({'type': 'scan_error', 'detail': str(exc), 'path': storage_path})
                continue
            if not exists:
                orphan_db_count += 1
                issues.append({
                    'type': 'missing_file',
                    'image_id': image.get('id'),
                    'project_id': image.get('project_id'),
                    'path': storage_path,
                })

        if projects_dir is None:
            projects_dir_path = Path(__file__).parent.parent / "projects"
        else:
            projects_dir_path = Path(projects_dir)

        orphan_disk_count = 0
        try:
            db_projects = self.get_all_projects()

            known_project_dirs = {}
            for project in db_projects:
                project_storage_path = project.get('storage_path')
                if not project_storage_path:
                    continue
                try:
                    normalized_project_path = _normalize(project_storage_path)
                except Exception as exc:
                    issues.append({
                        'type': 'scan_error',
                        'detail': str(exc),
                        'path': project_storage_path,
                    })
                    continue
                known_project_dirs[normalized_project_path] = project

                if not Path(normalized_project_path).exists():
                    orphan_db_count += 1
                    issues.append({
                        'type': 'missing_project_dir',
                        'project_id': project.get('id'),
                        'path': project_storage_path,
                    })

            if projects_dir_path.exists():
                for project_folder in projects_dir_path.iterdir():
                    if not project_folder.is_dir():
                        continue

                    normalized_folder_path = _normalize(str(project_folder))
                    project = known_project_dirs.get(normalized_folder_path)

                    if project is None:
                        orphan_disk_count += 1
                        issues.append({'type': 'orphan_project_dir', 'path': str(project_folder)})
                    else:
                        for root, dirs, files in os.walk(project_folder):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    normalized_file_path = _normalize(file_path)
                                except Exception as exc:
                                    issues.append({
                                        'type': 'scan_error',
                                        'detail': str(exc),
                                        'path': file_path,
                                    })
                                    continue
                                if normalized_file_path not in db_files:
                                    orphan_disk_count += 1
                                    issues.append({'type': 'orphan_file', 'path': file_path})
        except Exception as exc:
            issues.append({'type': 'scan_error', 'detail': str(exc)})

        return {
            'deleted_db_count': 0,
            'deleted_file_count': 0,
            'deleted_db_files': [],
            'deleted_actual_files': [],
            'total_deleted': 0,
            'orphan_db_count': orphan_db_count,
            'orphan_disk_count': orphan_disk_count,
            'issues': issues,
            'has_issues': bool(issues),
        }


# 全局数据库实例
db = Database()
