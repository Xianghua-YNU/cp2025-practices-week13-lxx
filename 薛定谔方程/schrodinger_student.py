#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薛定谔方程 - 方势阱能级计算
"""

import numpy as np
import matplotlib.pyplot as plt

# 物理常数
HBAR = 1.0545718e-34  # 约化普朗克常数 (J·s)
ELECTRON_MASS = 9.1094e-31  # 电子质量 (kg)
EV_TO_JOULE = 1.6021766208e-19  # 电子伏转换为焦耳的系数


def calculate_y_values(E_values, V, w, m):
    """
    计算方势阱能级方程中的三个函数值
    """
    # 转换为焦耳单位
    E_J = E_values * EV_TO_JOULE
    V_J = V * EV_TO_JOULE
    
    # 计算k值
    k = np.sqrt(2 * m * E_J) / HBAR
    
    # 计算y1 = tan(kw/2)
    y1 = np.tan(k * w / 2)
    
    # 计算y2 = sqrt((V-E)/E)
    y2 = np.sqrt((V_J - E_J) / E_J)
    
    # 计算y3 = -sqrt(E/(V-E))
    y3 = -np.sqrt(E_J / (V_J - E_J))
    
    return y1, y2, y3


def plot_energy_functions(E_values, y1, y2, y3):
    """
    Plot the three functions for energy level calculation in a square potential well
    
    Parameters:
        E_values (numpy.ndarray): Array of energy values (eV)
        y1 (numpy.ndarray): Values of y1 = tan(kw/2)
        y2 (numpy.ndarray): Values of y2 = sqrt((V-E)/E) (even parity)
        y3 (numpy.ndarray): Values of y3 = -sqrt(E/(V-E)) (odd parity)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=(12, 8))
    
    # Plot y1 = tan(kw/2)
    plt.plot(E_values, y1, 'b-', linewidth=2, label=r'$y_1 = \tan(kw/2)$')
    
    # Plot y2 = sqrt((V-E)/E) (even parity)
    plt.plot(E_values, y2, 'r--', linewidth=2, 
             label=r'$y_2 = \sqrt{(V-E)/E}$ (even parity)')
    
    # Plot y3 = -sqrt(E/(V-E)) (odd parity)
    plt.plot(E_values, y3, 'g-.', linewidth=2, 
             label=r'$y_3 = -\sqrt{E/(V-E)}$ (odd parity)')
    
    # Set plot properties
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Function value', fontsize=12)
    plt.title('Energy Level Equations in Square Potential Well', fontsize=14)
    
    # Set y-axis limits to better visualize crossings
    plt.ylim(-10, 10)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper right')
    
    # Highlight the energy range of interest
    plt.axvspan(0, 20, facecolor='yellow', alpha=0.1)
    
    return plt.gcf()


def find_energy_level_bisection(n, V, w, m, precision=0.001, E_min=0.001, E_max=None):
    """
    使用二分法求解方势阱中的第n个能级
    """
    if E_max is None:
        E_max = V - precision
    
    def f(E):
        y1, y2, y3 = calculate_y_values(np.array([E]), V, w, m)
        if n % 2 == 0:  # 偶宇称
            return y1[0] - y2[0]
        else:  # 奇宇称
            return y1[0] - y3[0]
    
    # 确保函数在区间内有解
    a, b = E_min, E_max
    while f(a) * f(b) > 0:
        a += 0.1
        if a >= b:
            return None
    
    # 二分法求解
    while (b - a) > precision:
        c = (a + b) / 2
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2


def main():
    """
    主函数，执行方势阱能级的计算和可视化
    """
    # 参数设置
    V = 20.0  # 势阱高度 (eV)
    w = 1e-9  # 势阱宽度 (m)
    m = ELECTRON_MASS  # 粒子质量 (kg)
    
    # 1. 计算并绘制函数曲线
    E_values = np.linspace(0.001, 19.999, 1000)  # 能量范围 (eV)
    y1, y2, y3 = calculate_y_values(E_values, V, w, m)
    fig = plot_energy_functions(E_values, y1, y2, y3)
    plt.savefig('energy_functions.png', dpi=300)
    plt.show()
    
    # 2. 使用二分法计算前6个能级
    energy_levels = []
    for n in range(6):
        energy = find_energy_level_bisection(n, V, w, m)
        energy_levels.append(energy)
        print(f"能级 {n}: {energy:.3f} eV")
    
    # 与参考值比较
    reference_levels = [0.318, 1.270, 2.851, 5.050, 7.850, 11.215]
    print("\n参考能级值:")
    for n, ref in enumerate(reference_levels):
        print(f"能级 {n}: {ref:.3f} eV")


if __name__ == "__main__":
    main()
