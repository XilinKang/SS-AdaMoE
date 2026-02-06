import torch
import os
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor, HGBDataset
from torch_geometric.data import Data
import traceback


def test_acm_dataset():
    """
    专门测试ACM数据集加载问题的函数
    """
    print("=" * 60)
    print("ACM数据集加载测试")
    print("=" * 60)

    # 1. 首先检查基本环境
    print("1. 检查PyTorch和PyG环境:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")

    # 2. 测试HGBDataset的导入和基本功能
    print("\n2. 测试HGBDataset导入:")
    try:
        from torch_geometric.datasets import HGBDataset
        print("   ✓ HGBDataset导入成功")

        # 检查可用的数据集
        print("   HGBDataset支持的数据集:")
        # 注意：实际中可能需要查看文档，这里列出常见的数据集
        supported_datasets = ['acm', 'DBLP', 'IMDB', 'Freebase']
        for ds in supported_datasets:
            print(f"     - {ds}")

    except ImportError as e:
        print(f"   ✗ HGBDataset导入失败: {e}")
        return False

    # 3. 尝试加载ACM数据集
    print("\n3. 尝试加载ACM数据集:")
    root_path = './datasets'
    os.makedirs(root_path, exist_ok=True)

    try:
        dataset = HGBDataset(root=root_path, name='acm')
        print("   ✓ ACM数据集加载成功")
        print(f"   数据集包含 {len(dataset)} 个图")

        if len(dataset) > 0:
            data = dataset[0]
            print(f"   图数据信息:")
            print(f"     - 节点数: {data.num_nodes}")
            print(f"     - 边数: {data.num_edges}")
            print(f"     - 特征维度: {data.num_features}")

            # 检查是否是异质图
            if hasattr(data, 'num_node_types'):
                print(f"     - 节点类型数: {data.num_node_types}")
            if hasattr(data, 'num_edge_types'):
                print(f"     - 边类型数: {data.num_edge_types}")

        return True

    except Exception as e:
        print(f"   ✗ ACM数据集加载失败: {e}")
        print("\n   详细错误信息:")
        traceback.print_exc()

        # 4. 提供解决方案建议
        print("\n4. 可能的解决方案:")
        print("   a) 检查网络连接，ACM数据集需要从网络下载")
        print("   b) 尝试手动下载数据集:")
        print("      访问: https://drive.google.com/file/d/1xbJ4QE9pcDJ0cALv7dYhHDCPITX2Iddz/view")
        print("      下载后放置到: ./datasets/ACM/raw/ 目录下")
        print("   c) 检查PyG版本兼容性")
        print("   d) 尝试使用其他数据集名称，如 'acm' 或 'ACM3025'")

        return False


def test_all_datasets():
    """
    测试所有数据集的加载情况，用于对比分析
    """
    print("=" * 60)
    print("所有数据集加载测试对比")
    print("=" * 60)

    datasets_to_test = ['Cora', 'CiteSeer', 'PubMed', 'acm']
    root_path = './datasets'

    for name in datasets_to_test:
        print(f"\n测试数据集: {name}")
        print("-" * 30)

        try:
            if name in ['Cora', 'CiteSeer', 'PubMed']:
                dataset = Planetoid(root=root_path, name=name)
            elif name == 'acm':
                dataset = HGBDataset(root=root_path, name='acm')
            else:
                print("   未知的数据集类型")
                continue

            print(f"   ✓ 加载成功")
            print(f"   包含 {len(dataset)} 个图")

            if len(dataset) > 0:
                data = dataset[0]
                print(f"   节点数: {data.num_nodes}")
                print(f"   边数: {data.num_edges}")
                if hasattr(data, 'num_features'):
                    print(f"   特征维度: {data.num_features}")

        except Exception as e:
            print(f"   ✗ 加载失败: {e}")


if __name__ == '__main__':
    # 运行ACM专用测试
    acm_success = test_acm_dataset()

    # 运行所有数据集的对比测试
    test_all_datasets()

    print("\n" + "=" * 60)
    if acm_success:
        print("ACM数据集测试完成 - 加载成功!")
    else:
        print("ACM数据集测试完成 - 需要进一步排查问题")
    print("=" * 60)