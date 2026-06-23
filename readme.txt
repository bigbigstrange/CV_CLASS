================================================================================
  2026 课程实训 — 精密零部件表面缺陷检测
  方案 A：SegDecNet (KSDD) + SuperSimpleNet (KSDD2) 三阶段训练
================================================================================

【项目结构】
  2026课程实训/
  ├── readme.txt                 ← 本文件
  ├── .gitignore
  └── code/
      ├── requirements.txt       依赖
      ├── config/                训练/数据集配置
      ├── data/                  数据加载与预处理
      ├── engine/                训练器、损失、指标
      ├── models/                SegDecNet + SuperSimpleNet
      ├── scripts/
      │   ├── train.py           统一训练入口
      │   ├── evaluate.py        正式评估（KSDD2 + KSDD）
      │   ├── inspect_datasets.py  检查数据集是否就绪
      │   └── verify_preprocessor.py  预处理 smoke test
      ├── Datasets/
      │   ├── kolektor缺陷数据集/   KSDD（需自行下载，不上传 GitHub）
      │   ├── KolektorSDD2/         KSDD2（需自行下载，不上传 GitHub）
      │   └── KSDD-splits/          官方 3-fold 划分（可上传）
      └── outputs/checkpoints/   训练权重（本地生成，默认不上传 GitHub）
          ├── phase1/            SegDecNet
          ├── pretrain/          SSN 预训练
          └── finetune/          SSN 微调

【硬件建议】
  - GPU：NVIDIA，显存 ≥ 8 GB（batch-size 4）；OOM 时改用 batch-size 2
  - 磁盘：数据集约 2–5 GB；checkpoint 每个约 100–200 MB

================================================================================
一、环境搭建
================================================================================

1. 进入项目根目录（所有命令均在此目录执行）

   cd F:\WORK\2026课程实训

2. 创建 Conda 环境（推荐 Python 3.10）

   conda create -n defect-det python=3.10 -y
   conda activate defect-det

3. 安装 PyTorch（按你的 CUDA 版本选一条，见 https://pytorch.org）

   # 示例：CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # 无 GPU / 仅 CPU
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

4. 安装其余依赖

   pip install -r code/requirements.txt

5. 验证 GPU（可选）

   python -c "import torch; print('cuda:', torch.cuda.is_available())"

================================================================================
二、数据集准备
================================================================================

将数据集放到以下路径（目录名必须一致）：

  code/Datasets/kolektor缺陷数据集/     ← KSDD，50 个 kos 文件夹，Part*.jpg + Part*_label.bmp
  code/Datasets/KolektorSDD2/           ← KSDD2，train/ 与 test/ 子目录，*.png + *_GT.png
  code/Datasets/KSDD-splits/            ← 已随仓库提供 split.pyb 等

KSDD  官方下载：https://www.vicos.si/Downloads/KolektorSDD
KSDD2 官方下载：https://www.vicos.si/Downloads/KolektorSDD2

检查数据集是否就绪：

   python code/scripts/inspect_datasets.py

检查预处理是否正常：

   python code/scripts/verify_preprocessor.py

================================================================================
三、完整训练流程（方案 A，fold 0）
================================================================================

说明：
  - 所有 python 命令均在项目根目录执行
  - --fresh 表示从头训练；不加 --fresh 则从 outputs/checkpoints/<stage>/last.pt 续训
  - Ctrl+C 中断训练会自动保存 last.pt，重新运行相同命令即可续训
  - pretrain / finetune 按 Val AP-det 保存 best.pt；评估默认 Recall 导向（hybrid_max + F2）

--------------------------------------------------------------------------------
阶段 1：Phase1 — SegDecNet @ KSDD（fold 0，约 20 epoch）
--------------------------------------------------------------------------------

   python code/scripts/train.py --stage phase1 --fold 0 --epochs 20 --batch-size 2 --fresh

产出：code/outputs/checkpoints/phase1/best.pt

--------------------------------------------------------------------------------
阶段 2：Pretrain — SuperSimpleNet @ KSDD2（建议 100 epoch）
--------------------------------------------------------------------------------

   python code/scripts/train.py --stage pretrain --epochs 100 --batch-size 4 --fresh

若显存不足：

   python code/scripts/train.py --stage pretrain --epochs 100 --batch-size 2 --fresh

续训（不加 --fresh，例如已训到 82 epoch 想训满 100）：

   python code/scripts/train.py --stage pretrain --epochs 100 --batch-size 4

产出：code/outputs/checkpoints/pretrain/best.pt

--------------------------------------------------------------------------------
阶段 3：Finetune — SSN 混合 KSDD2 + KSDD（KSDD 过采样 6×，15 epoch 起）
--------------------------------------------------------------------------------

   python code/scripts/train.py --stage finetune --fold 0 --epochs 15 --batch-size 4 --fresh

续训（例如从 15 epoch 续到 30）：

   python code/scripts/train.py --stage finetune --fold 0 --epochs 30 --batch-size 4

OOM 时：

   python code/scripts/train.py --stage finetune --fold 0 --epochs 30 --batch-size 2

产出：code/outputs/checkpoints/finetune/best.pt

================================================================================
四、正式评估
================================================================================

默认评估（Recall 导向 + 与旧 baseline 对比）：

   python code/scripts/evaluate.py --fold 0 --compare-baseline

指定 checkpoint：

   python code/scripts/evaluate.py --fold 0 ^
     --ssn-checkpoint code/outputs/checkpoints/finetune/best.pt ^
     --segdec-checkpoint code/outputs/checkpoints/phase1/best.pt ^
     --compare-baseline

恢复论文对标评估（decision head + F1 阈值）：

   python code/scripts/evaluate.py --fold 0 ^
     --threshold-criterion f1 ^
     --image-score-mode decision

Recall≥90% 导向阈值：

   python code/scripts/evaluate.py --fold 0 --threshold-criterion recall_at_target

保存 JSON 结果：

   python code/scripts/evaluate.py --fold 0 --compare-baseline --output code/outputs/eval_results.json

--------------------------------------------------------------------------------
当前最佳结果参考（finetune/best.pt epoch 28，fold 0）
--------------------------------------------------------------------------------

  KSDD2（SSN）  AP-det ≈ 95.91%   I-AUROC ≈ 98.96%   AP-loc ≈ 99.58%
  KSDD（SegDec） AP     = 100%     F1 ≈ 100%          IoU ≈ 0.75

  Recall 导向（hybrid_max + F2）：
    Recall ≈ 91.82%（较 decision+F1 的 85.45% 提升约 6.4 pp）
    AP-det 保持 95.91% 不变

================================================================================
五、常用参数速查
================================================================================

  train.py
    --stage       phase1 | pretrain | finetune
    --fold        KSDD 折数，默认 0
    --epochs      总目标 epoch（续训时填最终目标，非增量）
    --batch-size  批大小，8GB 显存 OOM 时用 2
    --fresh       从头训练，不读 last.pt
    --resume-from 显式指定 checkpoint 路径
    --device      cuda（默认）或 cpu

  evaluate.py
    --fold                  KSDD fold，默认 0
    --ssn-checkpoint        SSN 权重路径
    --segdec-checkpoint     SegDecNet 权重路径
    --threshold-criterion   f1 | f2 | recall_at_target
    --image-score-mode      decision | seg_max | seg_topk_mean | hybrid_max
    --target-fpr            Recall@FPR 的目标 FPR，默认 0.01
    --compare-baseline      同时输出旧 decision+f1 结果对比
    --output                保存 JSON

================================================================================
六、上传到 GitHub
================================================================================

说明：
  - 仓库远程地址（已配置）：https://github.com/bigbigstrange/CV_CLASS.git
  - .gitignore 已排除：大型数据集、压缩包、*.pt 权重、缓存、密钥
  - 上传内容：代码 + KSDD-splits + history.json；不上传原始图像与 checkpoint

--------------------------------------------------------------------------------
6.1 首次上传（本地已有 git，远程已有 origin）
--------------------------------------------------------------------------------

   cd F:\WORK\2026课程实训

   # 确认忽略规则生效（datasets / .pt 不应出现在列表中）
   git status

   # 添加代码与配置
   git add .gitignore readme.txt code/

   # 查看将要提交的文件（确认没有大体积数据集）
   git status

   # 提交
   git commit -m "Add defect detection pipeline: SegDecNet + SuperSimpleNet scheme A"

   # 推送到 GitHub（首次推送当前分支）
   git push -u origin main

若远程 main 已有 Initial commit 且推送被拒绝，先拉再推：

   git pull origin main --rebase
   git push -u origin main

--------------------------------------------------------------------------------
6.2 新建 GitHub 仓库后首次关联（若需换远程地址）
--------------------------------------------------------------------------------

   cd F:\WORK\2026课程实训

   git init
   git branch -M main
   git remote add origin https://github.com/<你的用户名>/<仓库名>.git

   git add .gitignore readme.txt code/
   git commit -m "Initial commit: surface defect detection project"
   git push -u origin main

--------------------------------------------------------------------------------
6.3 后续更新代码
--------------------------------------------------------------------------------

   git add code/
   git commit -m "描述本次修改内容"
   git push

--------------------------------------------------------------------------------
6.4 可选：单独上传训练 history（不含 .pt）
--------------------------------------------------------------------------------

   git add code/outputs/checkpoints/**/history.json
   git commit -m "Add training history logs"
   git push

--------------------------------------------------------------------------------
6.5 可选：用 Git LFS 上传 checkpoint（需 LFS 配额）
--------------------------------------------------------------------------------

   git lfs install
   git lfs track "code/outputs/checkpoints/**/*.pt"
   git add .gitattributes
   git add code/outputs/checkpoints/
   git commit -m "Add model checkpoints via Git LFS"
   git push

--------------------------------------------------------------------------------
6.6 克隆仓库后他人如何复现
--------------------------------------------------------------------------------

   git clone https://github.com/bigbigstrange/CV_CLASS.git
   cd CV_CLASS

   # 按「一、环境搭建」安装依赖
   pip install -r code/requirements.txt

   # 自行下载 KSDD / KSDD2 放到 code/Datasets/ 对应目录
   python code/scripts/inspect_datasets.py

   # 从头训练，或向维护者索取 checkpoint 放到 outputs/checkpoints/

================================================================================
七、故障排查
================================================================================

  问题：CUDA out of memory
  解决：--batch-size 2；关闭占显存的其他程序；可选：
        set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  问题：Checkpoint not found
  解决：确认上一阶段已训练完成；finetune 需 pretrain/best.pt；evaluate 会自动 fallback

  问题：数据集路径错误
  解决：运行 inspect_datasets.py；确认目录名与 paths.py 一致

  问题：git push 过大 / 超时
  解决：确认 .gitignore 包含 Datasets 与 *.pt；git rm -r --cached <误加的大目录>

  问题：续训 epoch 不对
  解决：--epochs 填「总目标 epoch」，不是「再训几个」；不加 --fresh

================================================================================
================================================================================
