#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析
"""

import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen

def load_sunspot_data(url):
    """
    从本地文件读取太阳黑子数据
    
    参数:
        url (str): 本地文件路径
        
    返回:
        tuple: (years, sunspots) 年份和太阳黑子数
    """
    # 读取数据文件，跳过前2行表头，只读取第2和3列
    data = np.loadtxt(url, skiprows=2, usecols=(1, 2))
    years = data[:, 0]
    sunspots = data[:, 1]
    return years, sunspots

def plot_sunspot_data(years, sunspots):
    """
    绘制太阳黑子数据随时间变化图
    
    参数:
        years (numpy.ndarray): 年份数组
        sunspots (numpy.ndarray): 太阳黑子数数组
    """
    plt.figure(figsize=(12, 6))
    plt.plot(years, sunspots, linewidth=1)
    plt.title('Sunspot Number Variation (1749-Present)', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Monthly Sunspot Number', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('sunspot_time_series.png', dpi=300)
    plt.show()

def compute_power_spectrum(sunspots):
    """
    计算太阳黑子数据的功率谱
    
    参数:
        sunspots (numpy.ndarray): 太阳黑子数数组
        
    返回:
        tuple: (frequencies, power) 频率数组和功率谱
    """
    # 去除均值
    sunspots = sunspots - np.mean(sunspots)
    
    # 傅里叶变换
    n = len(sunspots)
    fft_result = np.fft.fft(sunspots)
    
    # 计算功率谱 (取绝对值平方)
    power = np.abs(fft_result)**2
    
    # 计算频率 (只取正频率部分)
    frequencies = np.fft.fftfreq(n, d=1)[:n//2]
    power = power[:n//2]
    
    return frequencies, power

def plot_power_spectrum(frequencies, power):
    """
    绘制功率谱图
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
    """
    plt.figure(figsize=(12, 6))
    plt.semilogy(1/frequencies[1:], power[1:], linewidth=1)
    plt.title('Power Spectrum of Sunspot Numbers', fontsize=14)
    plt.xlabel('Period (months)', fontsize=12)
    plt.ylabel('Power (log scale)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('sunspot_power_spectrum.png', dpi=300)
    plt.show()

def find_main_period(frequencies, power):
    """
    找出功率谱中的主周期
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
        
    返回:
        float: 主周期（月）
    """
    # 找到最大功率对应的索引（跳过0频率）
    max_idx = np.argmax(power[1:]) + 1
    
    # 计算对应的周期
    main_period = 1 / frequencies[max_idx]
    
    return main_period

def main():
    # 数据文件路径
    data = "sunspot_data.txt"
    
    # 1. 加载并可视化数据
    print("Loading and plotting sunspot data...")
    years, sunspots = load_sunspot_data(data)
    plot_sunspot_data(years, sunspots)
    
    # 2. 傅里叶变换分析
    print("Computing power spectrum...")
    frequencies, power = compute_power_spectrum(sunspots)
    plot_power_spectrum(frequencies, power)
    
    # 3. 确定主周期
    main_period = find_main_period(frequencies, power)
    print(f"\nMain period of sunspot cycle: {main_period:.2f} months")
    print(f"Approximately {main_period/12:.2f} years")

if __name__ == "__main__":
    main()
