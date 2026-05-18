# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime


class BatchProcessor(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, files, method, params, output_dir=None, save_original=True):
        super().__init__()
        self.files = files
        self.method = method
        self.params = params
        self.output_dir = output_dir
        self.save_original = save_original
        self.results = []
    

HELP_TEXT = """

- 所有算法仅作用于“数值幅值”，不改变时间戳
- 平滑后每个点仍对应原始时间采样点
- 延迟指“等效时间延迟”，用于工程对比

==================================================
1) SMA（Simple Moving Average，简单滑动平均）
==================================================

【原理】
对最近 N 个历史点求平均：

y[i] = (1/N) * sum( x[i-k] ),  k = 0 .. N-1

【参数】
- N (window_size)：参与平均的点数

【延迟特性】
- 等效延迟 ≈ (N - 1) / 2 个采样周期
- 延迟固定，随 N 线性增加

【优点】
- 实现极其简单
- 对白噪声抑制明显
- 易解释、易调试

【缺点】
- 动态响应慢
- 对阶跃 / 突变非常迟钝
- N 大时拖尾明显

【适用场景】
- 离线分析
- 显示用平滑曲线
- 对实时性要求不高的系统

==================================================
2) EWMA（Exponential Weighted Moving Average）
==================================================

【原理】
当前输出由当前输入和过去输出加权：

y[i] = α * x[i] + (1 - α) * y[i-1]

【参数】
- α (0 ~ 1)：平滑因子
  - α 小 → 更平滑、延迟大
  - α 大 → 响应快、噪声多

【延迟特性】
- 等效延迟 ≈ (1 - α) / α 个采样周期
- 延迟连续可调、非整数

【优点】
- 不需要窗口长度
- 延迟可精细调节
- 计算量极低（MCU 友好）

【缺点】
- 无法区分趋势与噪声
- 参数直觉性稍弱

【适用场景】
- 实时传感器滤波
- MCU / FPGA
- 温湿度、压力、气体浓度

==================================================
3) Median Filter（中值滤波）
==================================================

【原理】
对最近 N 个点取中位数：

y[i] = median( x[i-N+1 : i] )

【参数】
- N (window_size)

【延迟特性】
- 类似 SMA
- 对突变无“拖尾效应”

【优点】
- 对尖峰噪声 / 毛刺极强
- 不会拉偏均值

【缺点】
- 对连续高斯噪声抑制一般
- 平滑程度有限

【适用场景】
- 毛刺 / 干扰较多
- I2C / SPI 抖动
- 传感器偶发异常点

==================================================
4) Median + EWMA（组合滤波，工程推荐）
==================================================

【原理】
先中值滤波去毛刺，再 EWMA 平滑：

raw → median → ewma → output

【参数】
- median_window (N)
- α (EWMA)

【延迟特性】
- 延迟略高于单一 EWMA
- 稳定性显著提升

【优点】
- 同时抑制：
  - 尖峰噪声
  - 连续噪声
- 工业界非常常见、稳健

【缺点】
- 参数稍多
- 需要简单调参

【适用场景】
- 实测数据波动大
- 工业 / 环境传感器
- ⭐ 强烈推荐用于你当前这类数据

==================================================
5) Holt（Holt's Linear Trend Method）
==================================================

【原理】
同时估计"水平 + 趋势"：

level[i] = α * x[i] + (1 - α) * (level[i-1] + trend[i-1])
trend[i] = β * (level[i] - level[i-1]) + (1 - β) * trend[i-1]

【参数】
- α：水平平滑系数
- β：趋势平滑系数

【延迟特性】
- 对趋势的延迟明显小于 SMA / EWMA
- 可"提前跟上"缓慢变化

【优点】
- 非常适合慢变化趋势
- 延迟小

【缺点】
- 高频噪声抑制能力一般
- 不适合剧烈抖动

【适用场景】
- 温度 / 湿度缓慢变化
- 漂移分析
- 长时间趋势监测

==================================================
6) Alpha-Beta Filter（α–β 滤波）
==================================================

【原理】
Holt 的工程化形式（线性运动模型）：

预测：
  x_hat = x_hat_prev + v_prev * dt

更新：
  x_hat = x_hat + α * (z - x_hat)
  v     = v     + β * (z - x_hat) / dt

【参数】
- α：位置更新强度
- β：速度（趋势）更新强度

【延迟特性】
- 延迟小
- 具备趋势预测能力

【优点】
- 计算量极低
- 非常适合实时系统
- MCU 友好

【缺点】
- 假设趋势线性
- 噪声模型简单

【适用场景】
- MCU 实时处理
- 传感器趋势跟踪
- 低功耗设备

==================================================
7) Kalman Filter（一维卡尔曼滤波）
==================================================

【原理】
基于状态空间模型，在已知噪声统计下的最优估计
（线性 + 高斯假设）

【参数】
- Q：过程噪声（系统不确定性）
- R：测量噪声（传感器噪声）

【延迟特性】
- 理论上延迟最小
- 可在"平滑 vs 跟踪"间自适应权衡

【优点】
- 噪声大、不稳定时表现最好
- 数学最优（线性系统）

【缺点】
- 参数理解成本高
- 对初值敏感

【适用场景】
- 波动大
- 噪声不稳定
- 高级算法评估 / 对比

==================================================
Pareto 参数扫描说明
==================================================

- 横轴：时间延迟（秒）
- 纵轴：平滑度提升
- 红点：Pareto 最优解
- 自动回填其中一个最优参数组合

==================================================

"""


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Algorithm Instructions / Parameter Guide")
        self.resize(650, 600)
        layout = QVBoxLayout(self)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(HELP_TEXT)
        layout.addWidget(text)


    
    def show_help(self):
        """Show help dialog"""
        dialog = HelpDialog(self)
        dialog.exec_()


# ===============================
# Main Application
# ===============================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = BatchSmoothingUI()
    window.show()
    
    sys.exit(app.exec_())
