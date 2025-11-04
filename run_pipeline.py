"""
完整的训练和筛选管道运行脚本
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

def run_data_collection():
    """运行数据收集"""
    print("\n" + "="*60)
    print("步骤 1/5: 数据收集")
    print("="*60)
    
    from src.data_collection import AntioxidantDataCollector
    
    collector = AntioxidantDataCollector()
    dataset = collector.collect_full_dataset()
    
    print("✓ 数据收集完成")
    return dataset

def run_feature_extraction():
    """运行特征提取"""
    print("\n" + "="*60)
    print("步骤 2/5: 特征提取")
    print("="*60)
    
    from src.feature_extraction import MolecularFeatureExtractor
    
    extractor = MolecularFeatureExtractor()
    features, labels, feature_names = extractor.process_dataset(
        input_csv='data/raw/antioxidant_dataset.csv',
        output_dir='data/processed'
    )
    
    print("✓ 特征提取完成")
    return features, labels

def run_traditional_training():
    """运行传统模型训练"""
    print("\n" + "="*60)
    print("步骤 3/5: 训练传统机器学习模型")
    print("="*60)
    
    from src.traditional_models import main as train_traditional
    
    train_traditional()
    
    print("✓ 传统模型训练完成")

def run_deep_learning_training():
    """运行深度学习训练"""
    print("\n" + "="*60)
    print("步骤 4/5: 训练深度学习模型")
    print("="*60)
    
    try:
        from src.deep_models import main as train_deep
        train_deep()
        print("✓ 深度学习模型训练完成")
    except Exception as e:
        print(f"深度学习训练出错（可能需要GPU或额外依赖）: {e}")
        print("跳过深度学习模型训练")

def run_quercetin_screening():
    """运行槲皮素筛选"""
    print("\n" + "="*60)
    print("步骤 5/5: 槲皮素筛选")
    print("="*60)
    
    from src.quercetin_screening import main as screen_quercetin
    
    screen_quercetin()
    
    print("✓ 槲皮素筛选完成")

def run_visualization():
    """运行可视化"""
    print("\n" + "="*60)
    print("生成可视化")
    print("="*60)
    
    from src.visualization import main as create_viz
    
    create_viz()
    
    print("✓ 可视化完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='抗氧化分子筛选完整管道',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行完整管道
  python run_pipeline.py --all
  
  # 只运行数据收集和特征提取
  python run_pipeline.py --collect --extract
  
  # 只运行模型训练
  python run_pipeline.py --train-traditional --train-deep
  
  # 只运行筛选
  python run_pipeline.py --screen
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='运行完整管道（所有步骤）')
    parser.add_argument('--collect', action='store_true',
                       help='运行数据收集')
    parser.add_argument('--extract', action='store_true',
                       help='运行特征提取')
    parser.add_argument('--train-traditional', action='store_true',
                       help='训练传统机器学习模型')
    parser.add_argument('--train-deep', action='store_true',
                       help='训练深度学习模型')
    parser.add_argument('--screen', action='store_true',
                       help='运行槲皮素筛选')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化')
    parser.add_argument('--quick', action='store_true',
                       help='快速运行（使用较少的epoch和数据）')
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，显示帮助
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("="*60)
    print("抗氧化分子筛选系统")
    print("参考深度学习方法，筛选槲皮素类抗氧化分子")
    print("="*60)
    
    try:
        # 运行完整管道
        if args.all:
            run_data_collection()
            run_feature_extraction()
            run_traditional_training()
            run_deep_learning_training()
            run_quercetin_screening()
            run_visualization()
        else:
            # 按需运行各个步骤
            if args.collect:
                run_data_collection()
            
            if args.extract:
                run_feature_extraction()
            
            if args.train_traditional:
                run_traditional_training()
            
            if args.train_deep:
                run_deep_learning_training()
            
            if args.screen:
                run_quercetin_screening()
            
            if args.visualize:
                run_visualization()
        
        print("\n" + "="*60)
        print("✓ 所有任务完成！")
        print("="*60)
        print("\n结果文件位置:")
        print("  - 数据: data/")
        print("  - 模型: models/")
        print("  - 结果: results/")
        print("  - 槲皮素筛选: results/quercetin/")
        print("\n请查看 notebooks/analysis.ipynb 进行详细分析")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

