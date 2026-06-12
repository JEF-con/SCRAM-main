#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
newFlame 时间统计数据分析模块
用于分析和计算 newFlame 函数各项操作的时间统计信息
"""

import os
import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import glob


class NewFlameTimingAnalyzer:
    """newFlame 时间统计数据分析器"""
    
    def __init__(self, timing_dir: str = "/home/jfl/code/FLAME-main/timing_analysis"):
        """
        初始化分析器
        
        Args:
            timing_dir: 时间统计数据目录路径
        """
        self.timing_dir = timing_dir
        self.operation_names = [
            "模型拆分成算术共享操作",
            "张量A和D生成及算术共享操作",
            "A和D验证过程",
            "张量alpha、beta、gamma生成及算术共享操作",
            "alpha、beta、gamma验证过程",
            "zeta相关操作",
            "deta相关操作",
            "同态加密操作",
            "聚类算法",
            "余弦相似度矩阵计算",
            "HDBSCAN聚类",
            "客户端分类和选择",
            "总体用时"
        ]
    
    def load_csv_data(self, session_id: Optional[str] = None) -> pd.DataFrame:
        """
        加载 CSV 格式的时间统计数据
        
        Args:
            session_id: 指定会话ID，如果为None则加载所有数据
            
        Returns:
            包含时间统计数据的 DataFrame
        """
        if session_id:
            csv_files = glob.glob(os.path.join(self.timing_dir, f"newflame_timing_{session_id}.csv"))
        else:
            csv_files = glob.glob(os.path.join(self.timing_dir, "newflame_timing_*.csv"))
        
        all_data = []
        
        for csv_file in csv_files:
            try:
                # 读取CSV文件，跳过可能的标题行
                df = pd.read_csv(csv_file, header=None, 
                               names=['session_id', 'round', 'timestamp', 'operation', 'time', 'percentage'])
                all_data.append(df)
            except Exception as e:
                print(f"读取文件 {csv_file} 时出错: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 数据类型转换
        combined_df['time'] = pd.to_numeric(combined_df['time'], errors='coerce')
        combined_df['percentage'] = pd.to_numeric(combined_df['percentage'], errors='coerce')
        combined_df['round'] = pd.to_numeric(combined_df['round'], errors='coerce')
        
        return combined_df
    
    def load_json_data(self, session_id: Optional[str] = None) -> List[Dict]:
        """
        加载 JSON 格式的时间统计数据
        
        Args:
            session_id: 指定会话ID，如果为None则加载所有数据
            
        Returns:
            包含时间统计数据的列表
        """
        if session_id:
            json_files = glob.glob(os.path.join(self.timing_dir, f"newflame_timing_{session_id}.json"))
        else:
            json_files = glob.glob(os.path.join(self.timing_dir, "newflame_timing_*.json"))
        
        all_data = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                print(f"读取文件 {json_file} 时出错: {e}")
                continue
        
        return all_data
    
    def calculate_operation_statistics(self, data_source: str = "csv") -> Dict:
        """
        计算各项操作的统计信息
        
        Args:
            data_source: 数据源类型，"csv" 或 "json"
            
        Returns:
            包含统计信息的字典
        """
        if data_source == "csv":
            return self._calculate_csv_statistics()
        else:
            return self._calculate_json_statistics()
    
    def _calculate_csv_statistics(self) -> Dict:
        """基于 CSV 数据计算统计信息"""
        df = self.load_csv_data()
        
        if df.empty:
            return {"error": "没有找到有效的CSV数据"}
        
        statistics = {}
        
        # 按操作类型分组计算统计信息
        for operation in self.operation_names:
            operation_data = df[df['operation'] == operation]
            
            if not operation_data.empty:
                time_stats = {
                    'count': len(operation_data),
                    'mean_time': operation_data['time'].mean(),
                    'std_time': operation_data['time'].std(),
                    'min_time': operation_data['time'].min(),
                    'max_time': operation_data['time'].max(),
                    'median_time': operation_data['time'].median(),
                    'mean_percentage': operation_data['percentage'].mean(),
                    'std_percentage': operation_data['percentage'].std()
                }
                statistics[operation] = time_stats
        
        # 计算总体统计信息
        total_data = df[df['operation'] == '总体用时']
        if not total_data.empty:
            statistics['overall_summary'] = {
                'total_rounds': len(total_data),
                'sessions': df['session_id'].nunique(),
                'avg_total_time': total_data['time'].mean(),
                'std_total_time': total_data['time'].std(),
                'min_total_time': total_data['time'].min(),
                'max_total_time': total_data['time'].max()
            }
        
        return statistics
    
    def _calculate_json_statistics(self) -> Dict:
        """基于 JSON 数据计算统计信息"""
        json_data = self.load_json_data()
        
        if not json_data:
            return {"error": "没有找到有效的JSON数据"}
        
        # 提取各项操作的时间数据
        operation_times = {
            'arithmetic_share': [],
            'tensor_generation': [],
            'ad_verification': [],
            'alpha_beta_gamma': [],
            'abg_verification': [],
            'zeta_operations': [],
            'deta_operations': [],
            'homomorphic_operations': [],
            'clustering': [],
            'cosine_similarity': [],
            'hdbscan_clustering': [],
            'client_classification': [],
            'total_time': []
        }
        
        for record in json_data:
            if 'operations' in record:
                ops = record['operations']
                operation_times['total_time'].append(record.get('total_time', 0))
                
                # 提取各项操作时间
                if 'arithmetic_share' in ops:
                    operation_times['arithmetic_share'].append(ops['arithmetic_share']['time'])
                
                if 'tensor_generation' in ops:
                    operation_times['tensor_generation'].append(ops['tensor_generation']['time'])
                    if 'ad_verification' in ops['tensor_generation']:
                        operation_times['ad_verification'].append(ops['tensor_generation']['ad_verification']['time'])
                
                if 'alpha_beta_gamma' in ops:
                    operation_times['alpha_beta_gamma'].append(ops['alpha_beta_gamma']['time'])
                    if 'abg_verification' in ops['alpha_beta_gamma']:
                        operation_times['abg_verification'].append(ops['alpha_beta_gamma']['abg_verification']['time'])
                
                if 'zeta_operations' in ops:
                    operation_times['zeta_operations'].append(ops['zeta_operations']['time'])
                
                if 'deta_operations' in ops:
                    operation_times['deta_operations'].append(ops['deta_operations']['time'])
                
                if 'homomorphic_operations' in ops:
                    operation_times['homomorphic_operations'].append(ops['homomorphic_operations']['time'])
                
                # 聚类相关操作
                if 'clustering' in ops:
                    operation_times['clustering'].append(ops['clustering']['time'])
                    if 'cosine_similarity' in ops['clustering']:
                        operation_times['cosine_similarity'].append(ops['clustering']['cosine_similarity']['time'])
                    if 'hdbscan_clustering' in ops['clustering']:
                        operation_times['hdbscan_clustering'].append(ops['clustering']['hdbscan_clustering']['time'])
                    if 'client_classification' in ops['clustering']:
                        operation_times['client_classification'].append(ops['clustering']['client_classification']['time'])
        
        # 计算统计信息
        statistics = {}
        for operation, times in operation_times.items():
            if times:
                times_array = np.array(times)
                statistics[operation] = {
                    'count': len(times),
                    'mean_time': np.mean(times_array),
                    'std_time': np.std(times_array),
                    'min_time': np.min(times_array),
                    'max_time': np.max(times_array),
                    'median_time': np.median(times_array)
                }
        
        return statistics
    
    def generate_summary_report(self, output_file: Optional[str] = None) -> str:
        """
        生成汇总报告
        
        Args:
            output_file: 输出文件路径，如果为None则只返回字符串
            
        Returns:
            汇总报告字符串
        """
        csv_stats = self.calculate_operation_statistics("csv")
        json_stats = self.calculate_operation_statistics("json")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("newFlame 时间统计数据分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # CSV 数据统计
        if 'error' not in csv_stats:
            report_lines.append("基于 CSV 数据的统计分析:")
            report_lines.append("-" * 50)
            
            if 'overall_summary' in csv_stats:
                summary = csv_stats['overall_summary']
                report_lines.append(f"总训练轮数: {summary['total_rounds']}")
                report_lines.append(f"训练会话数: {summary['sessions']}")
                report_lines.append(f"平均总用时: {summary['avg_total_time']:.6f} 秒")
                report_lines.append(f"总用时标准差: {summary['std_total_time']:.6f} 秒")
                report_lines.append(f"最短总用时: {summary['min_total_time']:.6f} 秒")
                report_lines.append(f"最长总用时: {summary['max_total_time']:.6f} 秒")
                report_lines.append("")
            
            report_lines.append("各项操作详细统计:")
            report_lines.append(f"{'操作名称':<25} {'平均用时(秒)':<12} {'标准差':<10} {'平均占比(%)':<12} {'样本数':<8}")
            report_lines.append("-" * 80)
            
            for operation in self.operation_names:
                if operation in csv_stats and operation != '总体用时':
                    stats = csv_stats[operation]
                    report_lines.append(
                        f"{operation:<25} {stats['mean_time']:<12.6f} {stats['std_time']:<10.6f} "
                        f"{stats['mean_percentage']:<12.2f} {stats['count']:<8}"
                    )
            
            report_lines.append("")
        
        # JSON 数据统计
        if 'error' not in json_stats:
            report_lines.append("基于 JSON 数据的统计分析:")
            report_lines.append("-" * 50)
            
            operation_mapping = {
                'arithmetic_share': '模型拆分成算术共享',
                'tensor_generation': '张量A和D生成及算术共享',
                'ad_verification': 'A和D验证过程',
                'alpha_beta_gamma': '张量alpha、beta、gamma生成及算术共享',
                'abg_verification': 'alpha、beta、gamma验证过程',
                'zeta_operations': 'zeta相关操作',
                'deta_operations': 'deta相关操作',
                'homomorphic_operations': '同态加密操作',
                'clustering': '聚类算法',
                'cosine_similarity': '余弦相似度矩阵计算',
                'hdbscan_clustering': 'HDBSCAN聚类',
                'client_classification': '客户端分类和选择',
                'total_time': '总体用时'
            }
            
            report_lines.append(f"{'操作名称':<25} {'平均用时(秒)':<12} {'标准差':<10} {'最小值':<10} {'最大值':<10} {'样本数':<8}")
            report_lines.append("-" * 85)
            
            for key, chinese_name in operation_mapping.items():
                if key in json_stats:
                    stats = json_stats[key]
                    report_lines.append(
                        f"{chinese_name:<25} {stats['mean_time']:<12.6f} {stats['std_time']:<10.6f} "
                        f"{stats['min_time']:<10.6f} {stats['max_time']:<10.6f} {stats['count']:<8}"
                    )
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_content = "\n".join(report_lines)
        
        # 保存到文件
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"分析报告已保存到: {output_file}")
            except Exception as e:
                print(f"保存报告时出错: {e}")
        
        return report_content
    
    def get_latest_session_stats(self) -> Dict:
        """获取最新会话的统计信息"""
        csv_files = glob.glob(os.path.join(self.timing_dir, "newflame_timing_*.csv"))
        
        if not csv_files:
            return {"error": "没有找到时间统计文件"}
        
        # 找到最新的文件
        latest_file = max(csv_files, key=os.path.getctime)
        session_id = os.path.basename(latest_file).replace("newflame_timing_", "").replace(".csv", "")
        
        # 加载该会话的数据
        df = self.load_csv_data(session_id)
        
        if df.empty:
            return {"error": "最新会话数据为空"}
        
        # 计算该会话的统计信息
        session_stats = {
            'session_id': session_id,
            'total_rounds': df['round'].nunique(),
            'operations': {}
        }
        
        for operation in self.operation_names:
            operation_data = df[df['operation'] == operation]
            if not operation_data.empty:
                session_stats['operations'][operation] = {
                    'mean_time': operation_data['time'].mean(),
                    'std_time': operation_data['time'].std(),
                    'mean_percentage': operation_data['percentage'].mean()
                }
        
        return session_stats


def analyze_newflame_timing(output_dir: str = "/home/jfl/code/FLAME-main/timing_analysis") -> None:
    """
    分析 newFlame 时间统计数据的主函数
    
    Args:
        output_dir: 输出目录
    """
    analyzer = NewFlameTimingAnalyzer()
    
    # 生成汇总报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"newflame_timing_analysis_{timestamp}.txt")
    
    print("正在分析 newFlame 时间统计数据...")
    report = analyzer.generate_summary_report(report_file)
    
    # 打印到控制台
    print(report)
    
    # 获取最新会话统计
    latest_stats = analyzer.get_latest_session_stats()
    if 'error' not in latest_stats:
        print(f"\n最新会话 ({latest_stats['session_id']}) 统计信息:")
        print(f"训练轮数: {latest_stats['total_rounds']}")
        for op_name, stats in latest_stats['operations'].items():
            if op_name == '总体用时':
                print(f"{op_name}: 平均 {stats['mean_time']:.6f} 秒")


if __name__ == "__main__":
    analyze_newflame_timing()