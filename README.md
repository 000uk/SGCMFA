## SGCH-Net: Skeleton-Guided Cascaded Hierarchical Network
<img width="932" height="246" alt="image" src="https://github.com/user-attachments/assets/3ab28105-f550-4c3f-8c3e-8c30b2946228" />

본 프로젝트는 합성 데이터 증강과 Query-Based Dual-Stream Transformer를 결합하여 조명, 배경, 사용자 변화에 강건한 핸드 제스처 인식 모델을 제안한다.

### Motivation
핸드 제스처 인식은 환경 변화에 민감하며, 대규모 실제 데이터 수집이 어렵다는 한계를 가진다.
합성 데이터는 데이터 부족을 완화할 수 있지만 RGB 단일 모달리티 모델은 도메인 격차와 배경 노이즈로 인해 일반화 성능이 쉽게 붕괴된다.
이에 손의 구조(Skeleton)를 먼저 파악하고, 이를 단서로 시각 정보(RGB)를 능동적으로 찾는 인간의 인지 과정을 모방한 계층적 퓨전 네트워크를 제안한다.

### Core Contributions
- 3D 합성 데이터 기반 대규모 증강 파이프라인 구축
- Skeleton-Guided Cascaded Hierarchical Network (SGCH-Net) 제안
- 스켈레톤 구조 정보를 활용한 배경 노이즈 억제
- RGB–Skeleton 상호 보완적 특징 융합 구조

### Training Strategy
- Blender + Isaac Sim 기반 합성 데이터 생성
- 카메라 시점·조명·배경 무작위화를 통한 강건성 확보
- 손목 원점 이동 및 스케일 정규화를 통해 Transformer 학습 안정성 확보
- Temporal augmentation으로 속도 변화 대응

### Experimental Results
Jester Test Set (Real-world)
| Model | Modality | Accuracy |
|------|----------|----------|
| X3D-S | RGB | 36.10% |
| ST-GCN | Skeleton | 78.20% |
| ST-GCN-CMFA | RGB + Skeleton | 94.44% |
| **SGCH-Net (Proposed)** | **RGB + Skeleton** | **95.31%** |
