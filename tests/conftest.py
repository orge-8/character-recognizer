# -*- coding: utf-8 -*-
"""pytest 公共夹具：把插件根目录放进 sys.path，使纯模块可被直接导入。"""

import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))
