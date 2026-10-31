# envs —— 每个实验的 conda 环境导出

各模块的依赖栈差异很大，**按模块隔离环境**，避免版本冲突：

| 实验 | 建议环境名 | Python | 关键依赖 |
|---|:---:|:---:|---|
| 01 目标检测 | `pcd01` | 3.7 | tensorflow 2.3.0 + keras 2.4.3 |
| 02 目标跟踪 | `pcd02` | 3.8 | **opencv-contrib-python** ≥4.5.2 |
| 03 视频摘要 | `pcd03` | 3.8 | torch + torchvision |
| 04 视频检索 | `pcd04` | 3.8 | torch + faiss-cpu + hnswlib |

---

## 创建与导出

```bash
# 以实验 01 为例
conda create -n pcd01 python=3.7 -y
conda activate pcd01
pip install -r experiments/01_ObjectDetection_YOLOv3/requirements.txt

# 导出环境（提交时用）
conda env export --no-builds > envs/pcd01.yml
# 或只导出显式安装的包，更精简
conda env export --from-history > envs/pcd01_from_history.yml
# 纯 pip 依赖的另一种记录方式
pip freeze > envs/pcd01_pip.txt
```

## 从导出文件恢复环境

```bash
conda env create -f envs/pcd01.yml
conda activate pcd01
```

---

## 注意事项

- `conda env export` 默认会带上 `prefix:` 一行（本机绝对路径）。
  **提交前建议删掉这一行**，否则别人无法创建同名环境。
- 加 `--no-builds` 可以去掉平台相关的 build 串，提高跨机器可复现性。
- 导出文件里会记录 `cudatoolkit` / `cudnn` 等 GPU 相关包；
  若接收方没有 GPU，需手动把这些行去掉。
- 建议把环境导出文件**纳入版本控制**（本仓库 `.gitignore` 未忽略 `envs/`），
  它体积很小且对复现很有价值。
