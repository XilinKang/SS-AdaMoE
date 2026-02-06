import torch
from torch_geometric.datasets import HGBDataset
import os
import traceback


def validate_coraml_dataset():
    print("=" * 60)
    print("开始验证CoraML数据集")
    print("=" * 60)

    # 1. 检查raw目录内容
    raw_path = "./datasets/CoraML/raw"
    print(f"检查raw目录: {raw_path}")

    required_files = ["Cora.cites", "Cora.content"]
    missing_files = []

    for file in required_files:
        file_path = os.path.join(raw_path, file)
        if os.path.exists(file_path):
            print(f"✅ 找到文件: {file}")
        else:
            print(f"❌ 缺少文件: {file}")
            missing_files.append(file)

    if missing_files:
        print(f"\n警告: 缺少 {len(missing_files)} 个关键文件")
        print("请手动下载并放置以下文件:")
        for file in missing_files:
            print(f"  - {file}")
        print("下载地址: https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz")
        print("解压后将文件放入上述raw目录")
        return False

    # 2. 尝试加载数据集
    print("\n尝试加载CoraML数据集...")
    try:
        # 设置环境变量，确保使用本地文件
        os.environ["TORCH_HOME"] = "./datasets"

        # 尝试加载数据集
        dataset = HGBDataset(root='./datasets', name='CoraML')
        print("✅ 数据集加载成功!")

        # 获取图数据
        data = dataset[0]
        print(f"数据集包含 {len(dataset)} 个图")
        print(f"节点数: {data.num_nodes}")
        print(f"边数: {data.num_edges}")
        print(f"特征维度: {data.num_features}")

        # 检查是否有标签
        if hasattr(data, 'y') and data.y is not None:
            print(f"标签数: {data.y.unique().size(0)}")
        else:
            print("⚠️ 未找到标签信息")

        return True

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()

        print("\n建议解决方案:")
        print("1. 手动下载数据集: https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz")
        print("2. 解压后将以下文件放入 ./datasets/CoraML/raw/ 目录:")
        print("   - Cora.cites")
        print("   - Cora.content")
        print("3. 删除可能存在的空文件或损坏文件")
        print("4. 重新运行此验证脚本")

        return False


if __name__ == "__main__":
    success = validate_coraml_dataset()

    print("\n" + "=" * 60)
    if success:
        print("✅ 验证通过! CoraML数据集完整且可用")
    else:
        print("❌ 验证失败! 请按照建议解决问题")
    print("=" * 60)