# YOLOv11_DeepSort_Human_Detection

### _CCTV 영상에서 사람 감지 및 추적!_

## 결과

<div align="center">
<img src="https://github.com/aidevveloper/YOLOv11_DeepSort_Human_Detection/blob/main/assets/demo.gif?raw=true" width="760" height="868">
</div>

### ⚡ 설치
```bash
conda create -n YOLOv11 python=3.9
conda activate YOLOv11
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install PyYAML
pip install scipy
pip install tqdm
```

### 학습

* `main.py`에서 데이터셋 경로 설정
* **처음부터 학습**: `python main.py --train --epochs 600 --batch-size 8`

### 파인튜닝 (전이학습)
```bash
python main.py --finetune \
    --pretrained-path weights/coco/best.pt \
    --freeze 3 \
    --finetune-epochs 100 \
    --batch-size 8
```

* `--freeze 10`: 백본 레이어 10개 동결 (빠른 학습)
* `--optimizer AdamW`: AdamW 또는 SGD 선택 가능

### 테스트/검증
```bash
python main.py --test
```

### 📂 데이터셋 구조
```
├── CCTV_CocoFormat
    ├── images
        ├── train
            ├── 0001.jpg
            ├── 0002.jpg
        ├── val
            ├── 0001.jpg
            ├── 0002.jpg
    ├── labels
        ├── train
            ├── 0001.txt
            ├── 0002.txt
        ├── val
            ├── 0001.txt
            ├── 0002.txt
```

### 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--train` | 처음부터 학습 | - |
| `--finetune` | 사전학습 모델로 파인튜닝 | - |
| `--test` | 모델 평가 | - |
| `--epochs` | 학습 에포크 | 600 |
| `--batch-size` | 배치 크기 | 8 |
| `--freeze` | 동결할 레이어 수 | 0 |
| `--optimizer` | SGD / AdamW | AdamW |

### 출력 파일
```
weights/
├── best.pt              # 최고 성능 모델 (학습)
├── last.pt              # 마지막 체크포인트 (학습)
└── finetune/
    ├── ft_best.pt       # 최고 성능 모델 (파인튜닝)
    └── ft_last.pt       # 마지막 체크포인트 (파인튜닝)
```

⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!

#### 🔗 참고

* https://github.com/ultralytics/ultralytics
* https://github.com/jahongir7174/DeepSort