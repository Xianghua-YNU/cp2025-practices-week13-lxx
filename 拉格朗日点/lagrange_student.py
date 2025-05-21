#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉格朗日点 - 地球-月球系统L1点位置计算

本模块实现了求解地球-月球系统L1拉格朗日点位置的数值方法。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# 物理常数
G = 6.674e-11  # 万有引力常数 (m^3 kg^-1 s^-2)
M = 5.974e24   # 地球质量 (kg)
m = 7.348e22   # 月球质量 (kg)
R = 3.844e8    # 地月距离 (m)
omega = 2.662e-6  # 月球角速度 (s^-1)


def lagrange_equation(r):
    """
    L1拉格朗日点位置方程
    
    参数:
        r (float): 从地心到L1点的距离 (m)
    
    返回:
        float: 方程左右两边的差值，当r是L1点位置时返回0
    """
    # 地球引力 - 月球引力 - 离心力 = 0
    # 方程形式: GM/r^2 - Gm/(R-r)^2 - ω²r = 0
    earth_gravity = G * M / (r ​**​ 2)
    moon_gravity = G * m / ((R - r) ​**​ 2)
    centrifugal = omega ​**​ 2 * r
    equation_value = earth_gravity - moon_gravity - centrifugal
    
    return equation_value


def lagrange_equation_derivative(r):
    """
    L1拉格朗日点位置方程的导数，用于牛顿法
    
    参数:
        r (float): 从地心到L1点的距离 (m)
    
    返回:
        float: 方程对r的导数值
    """
    # 对lagrange_equation函数求导
    # d/dr[GM/r^2 - Gm/(R-r)^2 - ω²r] = -2GM/r^3 + 2Gm/(R-r)^3 - ω²
    derivative = -2 * G * M / (r ​**​ 3) + 2 * G * m / ((R - r) ​**​ 3) - omega ​**​ 2
    
    return derivative


def newton_method(f, df, x0, tol=1e-8, max_iter=100):
    """
    使用牛顿法（切线法）求解方程f(x)=0
    
    参数:
        f (callable): 目标方程，形式为f(x)=0
        df (callable): 目标方程的导数
        x0 (float): 初始猜测值
        tol (float): 收敛容差
        max_iter (int): 最大迭代次数
    
    返回:
        tuple: (近似解, 迭代次数, 收敛标志)
    """
    x = x0
    iterations = 0
    converged = False
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(fx) < tol:
            converged = True
            break
            
        if dfx == 0:
            break
            
        x_new = x - fx / dfx
        
        # 检查相对变化是否小于容差
        if abs(x_new - x) < tol * abs(x_new):
            converged = True
            x = x_new
            iterations = i + 1
            break
            
        x = x_new
        iterations = i + 1
    
    return x, iterations, converged


def secant_method(f, a, b, tol=1e-8, max_iter=100):
    """
    使用弦截法求解方程f(x)=0
    
    参数:
        f (callable): 目标方程，形式为f(x)=0
        a (float): 区间左端点
       
