#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傅立叶滤波和平滑 - 道琼斯工业平均指数分析

本模块实现了对道Jones工业平均指数数据的傅立叶分析和滤波处理。
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傅立叶滤波和平滑 - 道琼斯工业平均指数分析
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """
    加载道Jones工业平均指数数据
    """
    try:
        data = np.loadtxt(filename)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_data(data, title="Dow Jones Industrial Average"):
    """
    绘制时间序列数据
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data, 'b-', linewidth=1)
    plt.title(title, fontsize=14)
    plt.xlabel("Trading Day (2006-2010)", fontsize=12)
    plt.ylabel("Index Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def fourier_filter(data, keep_fraction=0.1):
    """
    执行傅立叶变换并滤波
    """
    # 计算实数傅立叶变换
    fft_coeff = np.fft.rfft(data)
    
    # 计算保留的系数数量
    n_coeff = len(fft_coeff)
    cutoff = int(n_coeff * keep_fraction)
    
    # 创建滤波后的系数数组
    filtered_coeff = fft_coeff.copy()
    filtered_coeff[cutoff:] = 0
    
    # 计算逆变换
    filtered_data = np.fft.irfft(filtered_coeff, n=len(data))
    
    return filtered_data, fft_coeff

def plot_comparison(original, filtered, title="Fourier Filter Result"):
    """
    绘制原始数据和滤波结果的比较
    """
    plt.figure(figsize=(10, 5))
    plt.plot(original, 'b-', linewidth=1, alpha=0.5, label="Original Data")
    plt.plot(filtered, 'r-', linewidth=2, label=f"Filtered (Keep {len(filtered_coeff[filtered_coeff!=0])/len(filtered_coeff)*100:.1f}%)")
    plt.title(title, fontsize=14)
    plt.xlabel("Trading Day (2006-2010)", fontsize=12)
    plt.ylabel("Index Value", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def main():
    # 任务1：数据加载与可视化
    data = load_data('dow.txt')
    if data is None:
        return
    
    plot_data(data, "Dow Jones Industrial Average - Original Data")
    
    # 任务2：傅立叶变换与滤波（保留前10%系数）
    filtered_10, coeff_10 = fourier_filter(data, 0.1)
    plot_comparison(data, filtered_10, "Fourier Filter (Keep Top 10% Coefficients)")
    
    # 任务3：修改滤波参数（保留前2%系数）
    filtered_2, coeff_2 = fourier_filter(data, 0.02)
    plot_comparison(data, filtered_2, "Fourier Filter (Keep Top 2% Coefficients)")

if __name__ == "__main__":
    main()
