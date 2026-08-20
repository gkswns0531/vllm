# NVIDIA GPU Tensor Core 내부 아키텍처 분석

## 목차

1. [SM 내부 연산 유닛 구조](#1-sm-내부-연산-유닛-구조)
2. [Tensor Core 기본 연산: 4×4×4 MMA](#2-tensor-core-기본-연산-4×4×4-mma)
3. [핵심 질문: 정밀도별 물리 분리 vs 논리적 재구성](#3-핵심-질문-정밀도별-물리-분리-vs-논리적-재구성)
4. [근거 1: Throughput 비율이 정확히 2의 거듭제곱](#4-근거-1-throughput-비율이-정확히-2의-거듭제곱)
5. [근거 2: Latency가 정밀도와 무관하게 거의 동일](#5-근거-2-latency가-정밀도와-무관하게-거의-동일)
6. [근거 3: SASS 명령어가 정밀도 그룹별로 통합](#6-근거-3-sass-명령어가-정밀도-그룹별로-통합)
7. [근거 4: FP16과 BF16이 동일 Throughput](#7-근거-4-fp16과-bf16이-동일-throughput)
8. [근거 5: NVDLA의 공개된 설계](#8-근거-5-nvdla의-공개된-설계)
9. [곱셈기 분할(Sub-word Parallelism) 원리](#9-곱셈기-분할sub-word-parallelism-원리)
10. [누적기(Accumulator) 내부 동작](#10-누적기accumulator-내부-동작)
11. [Warp 스레드와 Tensor Core의 협력 모델](#11-warp-스레드와-tensor-core의-협력-모델)
12. [세대별 Tensor Core 진화](#12-세대별-tensor-core-진화)
13. [Hopper의 Transformer Engine](#13-hopper의-transformer-engine)
14. [FP8 표준과 하드웨어 지원](#14-fp8-표준과-하드웨어-지원)
15. [물리적으로 공유되는 것 vs 포맷별 전용 로직](#15-물리적으로-공유되는-것-vs-포맷별-전용-로직)
16. [L4 (Ada Lovelace) vs H100 (Hopper) 아키텍처 비교](#16-l4-ada-lovelace-vs-h100-hopper-아키텍처-비교)
17. [확인된 사실 vs 강한 추론 vs 추측](#17-확인된-사실-vs-강한-추론-vs-추측)
18. [참고문헌](#18-참고문헌)

---

## 1. SM 내부 연산 유닛 구조

NVIDIA GPU의 SM(Streaming Multiprocessor) 안에는 두 종류의 연산 유닛이 **물리적으로 분리**되어 있다.

```
SM (Streaming Multiprocessor)
┌──────────────────────────────────────────────┐
│  공유 인프라: 레지스터 파일, L1 캐시, Warp 스케줄러   │
│                                              │
│  ┌─────────────────┐  ┌─────────────────┐    │
│  │  CUDA Cores      │  │  Tensor Cores    │    │
│  │  (스칼라 연산)     │  │  (행렬 연산 전용)  │    │
│  │                  │  │                  │    │
│  │  FP32 유닛        │  │  MMA 연산 유닛    │    │
│  │  FP64 유닛        │  │  (FP8~FP64,      │    │
│  │  INT32 유닛       │  │   INT4~INT8)     │    │
│  └─────────────────┘  └─────────────────┘    │
└──────────────────────────────────────────────┘
```

- **CUDA Cores**: 스칼라 덧셈, 곱셈 등 범용 연산. FP32/FP64/INT32 유닛은 물리적으로 별도.
- **Tensor Cores**: 행렬 곱셈-누적(MMA) 전용. 여러 정밀도를 **하나의 재구성 가능한 하드웨어**에서 처리.

이 둘은 SM 인프라(레지스터 파일, 캐시, 스케줄러)를 공유하지만, 실행 유닛 자체는 물리적으로 별도이다.

**출처**: NVIDIA 공식 아키텍처 백서 (Volta, Ampere, Hopper)

---

## 2. Tensor Core 기본 연산: 4×4×4 MMA

하나의 Tensor Core가 **1 클럭**에 수행하는 기본 연산:

```
D[4×4] = A[4×4] × B[4×4] + C[4×4]

  A (FP16)      B (FP16)      C (FP32)      D (FP32)
  4×4            4×4            4×4            4×4
┌─┬─┬─┬─┐   ┌─┬─┬─┬─┐   ┌─┬─┬─┬─┐   ┌─┬─┬─┬─┐
│ │ │ │ │   │ │ │ │ │   │ │ │ │ │   │ │ │ │ │
├─┼─┼─┼─┤ × ├─┼─┼─┼─┤ + ├─┼─┼─┼─┤ = ├─┼─┼─┼─┤
│ │ │ │ │   │ │ │ │ │   │ │ │ │ │   │ │ │ │ │
├─┼─┼─┼─┤   ├─┼─┼─┼─┤   ├─┼─┼─┼─┤   ├─┼─┼─┼─┤
│ │ │ │ │   │ │ │ │ │   │ │ │ │ │   │ │ │ │ │
├─┼─┼─┼─┤   ├─┼─┼─┼─┤   ├─┼─┼─┼─┤   ├─┼─┼─┼─┤
│ │ │ │ │   │ │ │ │ │   │ │ │ │ │   │ │ │ │ │
└─┴─┴─┴─┘   └─┴─┴─┴─┘   └─┴─┴─┴─┘   └─┴─┴─┴─┘

= 출력 16개 원소 × 4 FMA = 64 FMA 연산/클럭/Tensor Core
```

이것은 NVIDIA가 Volta 백서에서 공식 확인한 사실이다.

**출처**: NVIDIA Tesla V100 GPU Architecture Whitepaper

---

## 3. 핵심 질문: 정밀도별 물리 분리 vs 논리적 재구성

### 결론

Tensor Core 내부에서 FP8, FP16, BF16, INT8 등 서로 다른 정밀도는 **완전히 분리된 전용 회로가 아니라, 하나의 재구성 가능한(reconfigurable) 곱셈기 배열을 공유**한다. 낮은 정밀도는 같은 곱셈기를 **분할(subdivide)** 하여 더 많은 연산을 병렬 처리한다.

NVIDIA는 Tensor Core의 게이트/트랜지스터 레벨 설계를 공개한 적이 없다. 이 결론은 아래의 다섯 가지 독립적 근거로부터 도출된 **강한 추론**이다.

---

## 4. 근거 1: Throughput 비율이 정확히 2의 거듭제곱

### A100 (Ampere, 3세대)

| 포맷 | Bit 폭 | Throughput | FP16 대비 |
|------|--------|-----------|----------|
| FP64 | 64-bit | 19.5 TFLOPS | 1/16x |
| TF32 | 32-bit (내부 19-bit) | 156 TFLOPS | 1/2x |
| FP16 | 16-bit | 312 TFLOPS | 1x |
| BF16 | 16-bit | 312 TFLOPS | 1x |
| INT8 | 8-bit | 624 TOPS | 2x |
| INT4 | 4-bit | 1,248 TOPS | 4x |

### H100 (Hopper, 4세대)

| 포맷 | Throughput | FP16 대비 |
|------|-----------|----------|
| FP16 | 989 TFLOPS | 1x |
| FP8 | 1,979 TFLOPS | 2x |

### Blackwell (5세대)

| 포맷 | Throughput | FP16 대비 |
|------|-----------|----------|
| FP16 | ~1,925 TFLOPS | 1x |
| FP8 | ~3,851 TFLOPS | 2x |
| FP4 | ~7,703 TFLOPS | 4x |

**모든 세대에서 비트 폭이 절반이 되면 throughput이 정확히 2배가 된다.**

만약 각 정밀도에 **완전히 별도의 물리 회로**가 있다면, 이렇게 깨끗한 정수 배율이 나올 이유가 없다. 이 패턴은 **같은 물리적 곱셈기 배열을 비트 폭에 따라 분할하여 병렬화하는 구조**(sub-word parallelism)에서 나타나는 전형적인 특성이다.

**출처**: NVIDIA 공식 데이터시트, Hopper/Blackwell 백서

---

## 5. 근거 2: Latency가 정밀도와 무관하게 거의 동일

### Blackwell 마이크로벤치마크 결과

| 포맷 | MMA 명령어 Latency |
|------|------------------|
| FP16 | 11.2 cycles |
| FP8 | 11.8 cycles |
| FP6 | 12.3 cycles |
| FP4 | 12.6 cycles |
| INT8 | 11.9 cycles |

FP4의 throughput은 FP64 대비 **177배** 차이가 나지만, latency는 **1.27배** 차이밖에 나지 않는다.

연구자들의 결론:

> "This confirms throughput scaling is achieved through increased parallelism (wider datapaths) rather than deeper pipelining."
> (처리량 증가는 더 깊은 파이프라인이 아닌, 더 넓은 데이터 경로를 통한 병렬성 증가로 달성된다.)

별도의 물리 회로라면 파이프라인 깊이와 회로 특성이 달라 latency 차이가 더 클 것이다. 거의 동일한 latency는 **같은 파이프라인을 통과하되, 한 사이클에 처리하는 원소 수만 다른** 구조를 강하게 시사한다.

**출처**: Microbenchmarking NVIDIA's Blackwell Architecture (arXiv:2512.02189)

---

## 6. 근거 3: SASS 명령어가 정밀도 그룹별로 통합

Blackwell에서 역어셈블된 SASS(네이티브 하드웨어) 명령어:

| SASS 명령어 | 처리하는 포맷 |
|------------|-------------|
| **HMMA** | FP16, BF16, TF32 |
| **QMMA** | FP8, FP6 |
| **OMMA** | FP4 (block scaling 포함) |
| **IMMA** | INT8, INT4 |
| **DMMA** | FP64 |

FP16, BF16, TF32가 **동일한 SASS 명령어**(HMMA)를 사용한다는 것은, 하드웨어 디코더가 이들을 **같은 실행 유닛**으로 라우팅한다는 강한 증거이다. FP8과 FP6도 QMMA를 공유한다.

SemiAnalysis의 분석:

> "FP8 and FP6 have the same theoretical throughput, so we believe they share physical circuits in Tensor Cores."

**출처**: Dissecting the NVIDIA Blackwell Architecture (arXiv:2507.10789), SemiAnalysis

---

## 7. 근거 4: FP16과 BF16이 동일 Throughput

```
FP16:  1bit 부호 | 5bit 지수 | 10bit 가수
BF16:  1bit 부호 | 8bit 지수 |  7bit 가수
```

BF16의 가수(mantissa)는 7비트로 FP16의 10비트보다 작다. 만약 BF16 전용 곱셈기가 별도로 있다면:

- 7bit × 7bit 곱셈기는 10bit × 10bit보다 **면적이 약 절반** (곱셈기 면적 ∝ bit²)
- 같은 실리콘 면적에 더 많은 BF16 곱셈기를 넣을 수 있으므로, BF16이 FP16보다 **더 높은** throughput을 가져야 함

하지만 실제로는 **모든 세대에서 FP16 = BF16 throughput**이다.

이는 BF16이 FP16의 10bit 곱셈기를 그대로 사용하되, 7bit 가수를 **3bit 제로패딩**하여 10bit로 맞춰 넣는 구조를 강하게 시사한다.

**출처**: NVIDIA Ampere Architecture In-Depth (공식 블로그)

---

## 8. 근거 5: NVDLA의 공개된 설계

NVIDIA의 오픈소스 Deep Learning Accelerator(NVDLA)는 이 원리를 직접 보여준다:

> "NVDLA decomposes an FP16 unit into two INT8 units spatially."
> (FP16 유닛을 공간적으로 두 개의 INT8 유닛으로 분해한다.)

FP16 가수 곱셈에 사용되는 두 개의 INT8 곱셈기 블록이, INT8 모드에서는 독립적인 병렬 곱셈기로 동작하여 **2x throughput**을 달성한다.

NVDLA는 Tensor Core 자체는 아니지만, NVIDIA가 설계한 동일 원리의 하드웨어이며, Tensor Core에 유사한 기법이 적용되었을 가능성이 매우 높다.

**출처**: NVDLA Hardware Architectural Specification (nvdla.org)

---

## 9. 곱셈기 분할(Sub-word Parallelism) 원리

### 부동소수점 곱셈기의 구조

```
부동소수점 곱셈 (A × B):

1. 부호: XOR(A.sign, B.sign)          ← 게이트 1개 (무시할 수준)
2. 지수: A.exp + B.exp - bias          ← 단순 덧셈기
3. 가수: A.mantissa × B.mantissa       ← 전체 면적의 60~80% 차지
4. 정규화: leading zero 탐지 + 시프트    ← 나머지
```

**가수 곱셈이 면적과 전력의 대부분을 차지**한다. 그리고 가수 곱셈은 본질적으로 **부호 없는 정수 곱셈**이다.

### 분할 원리

```
FP16 모드 (10bit 가수):

  ┌──────────────────────┐
  │  10bit × 10bit 곱셈기  │  → 1개 곱셈 결과
  └──────────────────────┘

FP8-E4M3 모드 (3bit 가수, implicit bit 포함 4bit):

  ┌──────────┬──────────┐
  │ 4bit×4bit │ 4bit×4bit │  → 2개 곱셈 결과 (동시)
  └──────────┴──────────┘
  (같은 물리적 곱셈기를 분할)

FP4 모드 (1~2bit 가수):

  ┌─────┬─────┬─────┬─────┐
  │2b×2b│2b×2b│2b×2b│2b×2b│  → 4개 곱셈 결과 (동시)
  └─────┴─────┴─────┴─────┘
```

곱셈기 면적은 bit² 에 비례하므로:

- 10bit × 10bit = 100 단위 면적
- 4bit × 4bit × 2 = 32 단위 면적 (여유 있음)
- 2bit × 2bit × 4 = 16 단위 면적 (더욱 여유)

실제로는 carry propagation 등의 오버헤드로 완벽한 면적 비례는 아니지만, 큰 곱셈기를 분할해서 작은 곱셈을 병렬로 돌리는 것은 충분히 가능하다.

### 학술적 검증: 실제 제작된 칩

**Intel VLSI 2018**: 14nm CMOS로 제작된 재구성 가능 행렬곱 가속기:

- INT16/FP16 모드: 각 노드가 1×1 곱셈
- **INT8 모드: 같은 노드가 2×2 곱셈기로 재구성** → 4배 throughput
- 동일 물리 게이트, 제어 신호만 변경

**GLSVLSI 2023**: 단일 유닛이 FP8-E4M3, FP8-E5M2, FP16, FP32, INT8을 모두 처리:

- FP8 모드: 4개 병렬 곱셈
- FP16 모드: 4개 병렬 곱셈 (bit 폭 감소)
- INT8 모드: 8개 병렬 곱셈
- FP32 모드: 1개 곱셈

**출처**: Intel VLSI 2018 (IEEE), GLSVLSI 2023 (ACM), ADiP (arXiv:2510.10623)

---

## 10. 누적기(Accumulator) 내부 동작

### 표준 FP32보다 넓은 내부 정밀도

마이크로벤치마킹으로 밝혀진 사실 (Fasi & Higham, 2020; Khattak & Mikaitis, 2024):

```
IEEE FP32:       1bit 부호 | 8bit 지수 | 23bit 가수

Tensor Core 내부 누적기:
  Volta:         23 fractional bits (FP32과 동일)
  Turing:        24 fractional bits (FP32보다 1bit 넓음)
  Ampere:        24 fractional bits
  Ada/Hopper:    25 fractional bits (FP16/BF16 모드)
                 13 fractional bits (FP8 모드!)  ← 주목
  Blackwell:     25 fractional bits (모든 모드)
```

**주목할 점**: Ada Lovelace와 Hopper에서 FP8 모드의 누적기는 13 fractional bits로, FP16 모드(25bits)보다 **절반 이하**이다. 이는 FP8 경로의 누적 부분에 일부 **별도 로직 또는 다른 동작 모드**가 존재할 가능성을 시사한다. Blackwell에서는 25bits로 통일되어 개선되었다.

### 내적 계산의 세부 동작

```
4-element dot product: d = a₁b₁ + a₂b₂ + a₃b₃ + a₄b₄ + c

1. 곱셈: aᵢ × bᵢ → 정확한 곱 (반올림 없음, 내부 포맷에 정확히 표현)
2. 정렬: 5개 피연산자를 절대값이 큰 순서로 정렬 (하드웨어 비교기)
3. 누적: 큰 값부터 순서대로 덧셈
   - 중간 반올림: round-towards-zero (IEEE 기본값 round-to-nearest가 아님!)
   - 중간 정규화: 없음 (최종 결과에서만 정규화)
4. 최종 정규화: leading-zero-count + 시프트 + round-to-nearest-even
5. FP32로 출력
```

이 동작은 일반 FP32 연산과 다르기 때문에, 동일한 행렬곱을 CUDA Core와 Tensor Core로 계산하면 미세한 수치 차이가 발생할 수 있다.

### 세대별 누적 블록 크기

| 세대 | Block FMA 크기 | 추가 정렬 비트 |
|------|--------------|-------------|
| Volta | 4 | 0 |
| Ampere (FP16) | 8 | 1 |
| Hopper (FP16) | 16 | 2 |
| Hopper (FP8) | 16 | 2 |

Hopper에서 FP8과 FP16이 **동일한 Block FMA 크기(16)와 정렬 비트(2)**를 사용한다는 것은, 같은 누적 파이프라인을 공유한다는 추가 증거이다.

**출처**: Fasi & Higham (PeerJ CS, 2021), Khattak & Mikaitis (arXiv:2512.07004), MMA-Sim (arXiv:2511.10909)

---

## 11. Warp 스레드와 Tensor Core의 협력 모델

Tensor Core는 혼자 동작하지 않는다. 여러 스레드가 행렬 조각(fragment)을 레지스터에 나눠 들고 협력한다.

### 세대별 협력 단위

```
Volta:   Quadpair (8 스레드)   → HMMA.884 명령어
Ampere:  Warp (32 스레드)      → mma.sync.m16n8k16
Hopper:  Warp Group (128 스레드 = 4 Warp) → wgmma.mma_async.m64nNk16
```

### Ampere에서의 데이터 공급 (mma.sync)

```
Warp (32 스레드)
┌──────────────────────────────┐
│ T0  T1  T2  ... T31           │
│                               │
│ 각 스레드의 레지스터:             │
│  A fragment: FP16 × 8개       │
│  B fragment: FP16 × 4개       │
│  C/D accumulator: FP32 × 4개  │
│                               │
│    mma.sync 명령어 발행         │
│         ↓                     │
│    32개 스레드 전부 동기 대기      │
│         ↓                     │
│    Tensor Core 연산 수행        │
│         ↓                     │
│    결과 → 각 스레드 레지스터       │
└──────────────────────────────┘
```

### Hopper의 비동기 모델 (wgmma)

```
Ampere (동기 - mma.sync):
  명령 → [TC 연산] → 결과 → [대기] → 명령 → [TC 연산] → 결과
  ████████░░░░░░░████████░░░░░░░  (연산과 대기가 번갈아)

Hopper (비동기 - wgmma.mma_async):
  명령 → [TC 연산 + 동시에 메모리 접근 + 다음 명령 준비]
  ████████████████████████████████  (연산과 메모리 접근이 겹침)
```

| 속성 | mma.sync (Ampere) | wgmma (Hopper) |
|-----|-------------------|----------------|
| 스레드 수 | 32 (1 Warp) | 128 (4 Warps) |
| 실행 모델 | 동기 (blocking) | 비동기 (non-blocking) |
| A 행렬 소스 | 레지스터 | 레지스터 또는 shared memory |
| B 행렬 소스 | 레지스터 | shared memory만 |
| 피크 활용률 | ~63% | ~95%+ |

Hopper에서 레거시 `mma.sync`를 쓰면 피크의 **~63%**만 나오고, `wgmma`를 써야 **95%+**가 나온다.

**출처**: Dissecting the NVIDIA Hopper Architecture (arXiv:2501.12084), CUTLASS Tutorial: WGMMA on Hopper (Colfax Research)

---

## 12. 세대별 Tensor Core 진화

| | Volta (1세대) | Turing (2세대) | Ampere (3세대) | Hopper (4세대) | Blackwell (5세대) |
|---|---|---|---|---|---|
| 출시 | 2017 | 2018 | 2020 | 2022 | 2024 |
| 대표 GPU | V100 | T4 | A100 | H100 | B200 |
| TC 수/SM | 8 | 8 | 4 | 4 | 4 |
| FMA/TC/클럭 | 64 | 64 | 256 | 512 | 1024 |
| FMA/SM/클럭 | 512 | 512 | 1,024 | 2,048 | 4,096 |
| **지원 포맷** | FP16 | +INT8,4,1 | +BF16,TF32,FP64 | +FP8 | +FP6,FP4 |
| 실행 모델 | sync (8t) | sync (8t) | sync (32t) | async (128t) | async (128t) |
| Sparsity | X | X | 2:4 구조적 | 2:4 구조적 | 2:4 구조적 |

**참고**: Ampere에서 TC 수가 8→4로 줄고 개별 TC 성능이 4배가 된 것은, TC당 곱셈기 배열을 더 크게 만든 결과이다. SemiAnalysis:

> "Over generations, NVIDIA scaled the Tensor Core size more aggressively than the number of Tensor Cores."

**참고**: Hopper에서 INT4, INT1 지원이 **제거**되었다. Hopper의 INT4 MMA 명령어는 Tensor Core가 아닌 CUDA Core(IMAD 명령어)에서 실행된다.

**출처**: NVIDIA 공식 백서, SemiAnalysis, Hot Chips 발표

---

## 13. Hopper의 Transformer Engine

Tensor Core와 연계되는 **하드웨어+소프트웨어 시스템**:

```
┌─ SM 내부 ──────────────────────────────┐
│                                        │
│  Tensor Core                           │
│    FP8 × FP8 → FP32 결과               │
│         │                              │
│         ▼                              │
│  ┌─ Transformer Engine HW ──────────┐  │
│  │  출력값의 통계(amax) 수집          │  │
│  │  → 다음 레이어의 FP8 scale 계산    │  │
│  └──────────────────────────────────┘  │
│         │                              │
│         ▼                              │
│  다음 레이어에서 scale 적용하여          │
│  BF16 → FP8 동적 양자화                │
└────────────────────────────────────────┘
```

### Scaling 전략

| 전략 | 설명 | 용도 |
|------|------|------|
| **Delayed Scaling** | 이전 N 반복의 amax 이력으로 scale 결정. 추가 데이터 순회 불필요 | 학습 |
| **Current Scaling** | 현재 데이터에서 즉시 scale 계산. 정확하지만 추가 패스 필요 | 추론 |
| **Per-Block Scaling (MXFP8)** | 32개 값마다 별도 scale. Blackwell에서 네이티브 지원 | 차세대 |

### FP8 포맷 할당 (Hybrid Recipe)

- **E4M3** (4bit 지수, 3bit 가수): 순전파 (weight, activation) → 정밀도 우선
- **E5M2** (5bit 지수, 2bit 가수): 역전파 (gradient) → 동적 범위 우선
- 누적: 항상 **FP32** (WGMMA에서는 FP16도 가능)

**출처**: NVIDIA H100 Transformer Engine Blog, Transformer Engine FP8 Primer

---

## 14. FP8 표준과 하드웨어 지원

### FP8 표준화

- **2022년 9월**: NVIDIA, Arm, Intel이 공동으로 "FP8 Formats for Deep Learning" 발표 (arXiv:2209.05433)
- 두 가지 포맷 정의:
    - **E4M3**: 범위 ±448, 정밀도 높음
    - **E5M2**: 범위 ±57,344, 동적 범위 넓음

### GPU별 FP8 네이티브 지원

| GPU | 아키텍처 | Compute Capability | FP8 네이티브 |
|-----|---------|-------------------|-------------|
| A100 | Ampere | 8.0 | X (2020년 출시, FP8 표준 이전) |
| L4 | Ada Lovelace | 8.9 | O |
| RTX 4090 | Ada Lovelace | 8.9 | O |
| H100 | Hopper | 9.0 | O |
| B200 | Blackwell | 10.0 | O |

A100에 FP8이 없는 이유: A100(2020년)은 FP8 포맷 표준화(2022년) 이전에 설계/제조 완료. 곱셈기 배열에 FP8 포맷을 해석하고 분할 연산하는 **제어 로직과 포맷 변환 회로**가 설계에 포함되지 않았음. 칩은 제조 후 회로 변경 불가.

**출처**: NVIDIA FP8 Specification Blog, arXiv:2209.05433

---

## 15. 물리적으로 공유되는 것 vs 포맷별 전용 로직

### 포맷별 전용 로직이 필요한 이유: 부동소수점 연산의 단계별 분해

부동소수점 곱셈 A × B는 다음 단계로 수행된다. 이 중 **곱셈기(Step 3)만 포맷에 무관**하고, 나머지는 포맷의 비트 배치와 규칙이 달라 별도 회로가 필요하다.

```
부동소수점 수의 구조:  (-1)^S × 2^(E-bias) × 1.M

예시: FP16으로 3.5 × 2.25 를 계산

A = 3.5  → 부호=0, 지수=10000₂(=16), 가수=1.1100000000
B = 2.25 → 부호=0, 지수=10000₂(=16), 가수=1.0010000000

Step 1: 부호 결정   → XOR(0, 0) = 0 (양수)            ← 포맷 무관
Step 2: 지수 덧셈   → 16 + 16 - 15(bias) = 17         ← 포맷별 다름
Step 3: 가수 곱셈   → 1.110 × 1.001 = 1.111110        ← 정수 곱셈, 포맷 무관
Step 4: 정규화      → 이미 1.xxx 형태이므로 불필요       ← 포맷별 다름
Step 5: 반올림      → 10bit 가수로 맞춤                 ← 포맷별 다름
Step 6: 특수값 확인  → NaN? Inf? denormal?              ← 포맷별 다름

결과: (-1)^0 × 2^(17-15) × 1.1111100000 = 7.875  ✓
```

#### (1) 지수 처리 (Exponent Handling)

지수 덧셈 `E_result = E_A + E_B - bias` 에서 비트 폭과 bias가 포맷마다 다르다.

```
FP8-E4M3:   4bit 덧셈기, bias = 7    → 4bit carry chain + 7 뺄셈 회로
FP8-E5M2:   5bit 덧셈기, bias = 15   → 5bit carry chain + 15 뺄셈 회로
FP16:       5bit 덧셈기, bias = 15   → 5bit carry chain + 15 뺄셈 회로
BF16:       8bit 덧셈기, bias = 127  → 8bit carry chain + 127 뺄셈 회로
```

같은 수학적 값 3.5도 포맷에 따라 지수 인코딩이 다르다:

```
FP16에서 3.5:  E = 1+15 = 16 = 10000₂    (5bit)
BF16에서 3.5:  E = 1+127 = 128 = 10000000₂ (8bit)
FP8-E4M3:     E = 1+7 = 8 = 1000₂        (4bit)
```

4bit 덧셈기와 8bit 덧셈기는 carry propagation 체인 길이가 다르고, bias 하드코딩도 다르다.

#### (2) 정렬 시프터 (Alignment Shifter)

누적(덧셈)을 위해 두 수의 소수점 위치를 맞추는 시프터. 지수 차이만큼 가수를 시프트한다.

```
예시: 100.0 + 0.125

100.0  = 1.1001 × 2^6
0.125  = 1.0000 × 2^(-3)
지수 차이 = 9 → 0.125의 가수를 9칸 오른쪽 시프트

  1.1001000000 × 2^6    (100.0)
+ 0.0000000001 × 2^6    (0.125, 9칸 시프트됨)
──────────────────────
  1.1001000001 × 2^6    (100.125)
```

시프터의 최대 폭이 지수 범위에 의해 결정되므로 포맷마다 다르다:

```
FP8-E4M3:  범위 0~15  → 4단 배럴 시프터 (16:1 MUX)
FP16:      범위 0~31  → 5단 배럴 시프터 (32:1 MUX)
BF16:      범위 0~255 → 8단 배럴 시프터 (256:1 MUX)
```

실제로는 가수 폭이 좁으면 유효 비트가 빠르게 사라지므로, 필요 시프터 크기는 가수 폭 + 누적기 폭에 의해 제한된다.

#### (3) 정규화 (Normalization) — 면적 비중 가장 큼 (~21%)

곱셈/덧셈 결과가 `1.xxx` 형태가 아닐 수 있어 정규화가 필요하다.

```
오버플로우 케이스:
  1.111 × 2^5 + 1.001 × 2^5 = 11.000 × 2^5
  → 가수 1칸 오른쪽 시프트, 지수 +1 → 1.1000 × 2^6

언더플로우 케이스:
  1.000 × 2^3 - 0.111 × 2^3 = 0.001 × 2^3
  → LZC(Leading Zero Count) = 2
  → 가수 2칸 왼쪽 시프트, 지수 -2 → 1.000 × 2^1
```

포맷별 차이:

```
LZC 탐지 범위:
  FP8-E4M3:  3bit 가수  → 최대 3개 leading zero 탐지
  BF16:      7bit 가수  → 최대 7개 leading zero 탐지
  FP16:      10bit 가수 → 최대 10개 leading zero 탐지

지수 조정 시 오버플로우/언더플로우 범위:
  FP8-E4M3:  지수 0~15   → 좁은 범위
  BF16:      지수 0~255  → 넓은 범위
  → 오버플로우 탐지 비교기의 크기가 다름

출력 가수 절단 위치:
  FP8:  3bit에서 자름
  BF16: 7bit에서 자름
  FP16: 10bit에서 자름
```

BF16 행렬 엔진 합성 결과, 정규화 로직만으로 PE 면적의 **~21%**를 차지한다 (arXiv:2408.11997).

#### (4) 반올림 (Rounding)

곱셈 결과(2n-bit)를 출력 가수 폭(n-bit)으로 축소할 때, Guard/Round/Sticky bit의 물리적 위치가 다르다.

```
FP16: 10bit × 10bit = 20bit 곱
  → 반올림 판단 위치: 11번째(guard), 12번째(round), 나머지(sticky)

FP8-E4M3: 3bit × 3bit = 6bit 곱
  → 반올림 판단 위치: 4번째(guard), 5번째(round), 나머지(sticky)
```

단, Tensor Core 내부에서는 정확한 곱을 확장 포맷(23~25 fractional bits)으로 유지하고 최종 출력 시에만 FP32로 반올림하므로, 중간 반올림 회로는 공유될 수 있다.

#### (5) 특수값 탐지 (Special Value Detection) — 포맷별 차이가 가장 큼

```
FP16 (IEEE 754 호환):
  Zero:     E=00000, M=0000000000         → ±0
  Denormal: E=00000, M≠0                  → implicit bit = 0 (0.M 형태)
  Inf:      E=11111, M=0000000000         → ±∞
  NaN:      E=11111, M≠0                  → qNaN, sNaN
  탐지: 5bit all-zero + 5bit all-one + 10bit zero 비교기

BF16:
  구조 동일하나 비트 폭 다름:
  탐지: 8bit all-zero + 8bit all-one + 7bit zero 비교기

FP8-E4M3 (IEEE 비호환):
  Zero:     E=0000, M=000                 → ±0
  Denormal: E=0000, M≠0
  Inf:      없음!                          → Inf를 포기하여 동적 범위 확장
  NaN:      S.1111.111 만                  → 가수 패턴 1개 (비트 패턴 2개)
  S.1111.110 = ±448은 정상 값!             → FP16에서 같은 패턴이면 Inf
  탐지: 4bit + 3bit 비교기, Inf 탐지 불필요

FP8-E5M2 (IEEE 호환):
  Inf:      E=11111, M=00                 → ±∞
  NaN:      E=11111, M≠0                  → 6개 비트 패턴
  탐지: 5bit + 2bit 비교기
```

특수값을 만나면 곱셈 경로를 우회해야 한다:

```
A × B 에서:
  A 또는 B가 NaN  → 결과 = NaN (곱셈 생략)
  A=Inf, B=0      → 결과 = NaN (0×∞ 미정의)
  A=Inf, B=정상    → 결과 = Inf (곱셈 생략, 부호만 결정)
  A=0, B=정상      → 결과 = 0 (곱셈 생략)
  A=denormal       → 가수의 implicit bit이 0 (0.M 형태로 처리)

E4M3은 Inf가 없으므로:
  → "A=Inf" 케이스 자체가 존재하지 않음 → Inf 판단 회로 불필요
  → S.1111.110 이 정상 값으로 처리되어야 함
```

이 판단 로직이 포맷마다 다른 조합 논리 회로로 구현된다.

#### (6) 포맷 변환 (Format Conversion)

입력 포맷 → 내부 표현, 내부 → 출력 변환이 필요하다.

```
입력 변환:
  FP16 → 지수 5bit에서 bias 15 보정, 가수 10bit + implicit 1 = 11bit
  BF16 → 지수 8bit에서 bias 127 보정, 가수 7bit + zero-pad 3bit → 10bit 곱셈기에 맞춤
  FP8  → 지수 4bit에서 bias 7 보정, 가수 3bit → 분할 모드에서 4bit 곱셈기로 입력

출력 변환:
  내부 25 fractional bits → FP32 23bit로 반올림
```

각 포맷의 bias 보정 회로와 가수 폭 패딩/트렁케이션 회로가 다르다.

#### 포맷별 전용 로직 요약

| 로직 | 하는 일 | 포맷별로 다른 이유 | 면적 비중 |
|------|--------|-----------------|---------|
| **지수 처리** | E_A + E_B - bias | 비트 폭(4/5/8bit)과 bias(7/15/127)가 다름 | 작음 |
| **정렬 시프터** | 덧셈 전 소수점 위치 맞춤 | 지수 범위 → 최대 시프트 양이 다름 | 중간 |
| **정규화** | 결과를 1.xxx 형태로 복원 | LZC 범위, 지수 조정 범위, 절단 위치가 다름 | **가장 큼 (~21%)** |
| **반올림** | 넓은 곱을 출력 폭으로 축소 | Guard/Round/Sticky bit 위치가 다름 | 작음 |
| **특수값 탐지** | NaN, Inf, Zero, Denormal 판별 | 포맷마다 정의가 다름 (E4M3은 Inf 없음 등) | 작음 |
| **포맷 변환** | 입출력 포맷 ↔ 내부 표현 변환 | bias 보정, 가수 폭 패딩/트렁케이션 | 작음 |

**핵심**: 곱셈기(가수 × 가수)는 본질적으로 정수 곱셈이라 포맷에 무관하다. 하지만 그 곱셈을 올바르게 수행하기 위한 **전처리(지수, 정렬, 변환)와 후처리(정규화, 반올림, 특수값)**는 포맷의 비트 배치와 규칙이 다르기 때문에 별도의 물리 회로가 필요하다. 이 부분이 전체 면적의 약 15~25%를 차지한다.

---

### 공유되는 부분 (전체 면적의 ~70-80%)

| 구성요소 | 설명 |
|---------|------|
| **곱셈기 배열** | 가수(mantissa) 정수 곱셈을 수행하는 Wallace tree 또는 유사 구조. 전체 면적의 60-80%. 정밀도에 따라 분할/결합하여 재사용 |
| **누적 트리** | Carry-save adder(CSA) 기반 FP32 누적기. 모든 모드에서 공유 |
| **데이터 라우팅** | 레지스터 파일, shared memory 연결, 명령어 디코드 파이프라인 |
| **제어 로직** | 명령어에 따라 동작 모드를 선택하는 상태 머신 |

### 포맷별 전용 로직이 필요한 부분 (전체 면적의 ~20-30%)

| 구성요소 | 이유 |
|---------|------|
| **지수 처리** | FP8-E4M3은 4bit 지수(bias 7), FP16은 5bit(bias 15), BF16은 8bit(bias 127). 각각 다른 덧셈기/bias 회로 필요 |
| **정렬 시프터** | 지수 범위에 따라 시프트 양이 다름 |
| **정규화/반올림** | 포맷별 출력 정규화 로직 |
| **특수값 처리** | NaN, Inf, denormal 탐지가 포맷마다 다름 (E4M3은 NaN 가수 패턴 1개/Inf 없음, E5M2는 IEEE 호환) |
| **포맷 변환기** | 입출력 시 FP8↔FP32, FP16↔FP32 등의 변환 |

### 시각적 정리

```
Tensor Core 내부 (추정 모델)
┌────────────────────────────────────────────┐
│                                            │
│  ┌─ 포맷별 전용 로직 (~20-30%) ──────────┐  │
│  │  지수 처리 / 정렬 / 정규화 / 특수값    │  │
│  │  [FP16/BF16/TF32] [FP8 E4M3/E5M2]    │  │
│  │  [INT8]           [FP64]              │  │
│  └───────────────────┬───────────────────┘  │
│                      ↓                      │
│  ┌─ 공유 곱셈기 배열 (~60-70%) ──────────┐  │
│  │                                       │  │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┐   │  │
│  │  │4b │4b │4b │4b │4b │4b │4b │4b │   │  │
│  │  └───┴───┴───┴───┴───┴───┴───┴───┘   │  │
│  │                                       │  │
│  │  FP16: [4b+4b+4b = ~10b] × 1연산     │  │
│  │  FP8:  [4b] × 2연산 (병렬)            │  │
│  │  FP4:  [4b÷2] × 4연산 (병렬)          │  │
│  └───────────────────┬───────────────────┘  │
│                      ↓                      │
│  ┌─ 공유 누적기 (~10%) ─────────────────┐   │
│  │  FP32 (내부적으로 23~25 fractional bits)│  │
│  └──────────────────────────────────────┘   │
│                                            │
└────────────────────────────────────────────┘
```

**출처**: MLSys 2021 (Rethinking FP Overheads), ARITH 2024 (Fused FP8 Dot Product), NVDLA HW Spec

---

## 15-1. 면적 비율 크로스체크: 곱셈기 vs 포맷별 로직

위 섹션에서 제시한 "곱셈기 ~60-70%, 포맷별 로직 ~20-30%" 비율의 근거를 독립적으로 검증한 결과이다.

### 검증 1: 곱셈기가 면적의 대부분을 차지하는가?

| 출처 | 측정 방법 | 결과 |
|------|----------|------|
| Luo et al. (ScienceDirect) | FP 곱셈기 전력 분석 | 가수 곱셈 블록이 **전체 전력의 80% 이상** (전력 ∝ 면적) |
| arXiv:2408.11997 | BF16 행렬 엔진 합성 결과 | 정규화 로직만 PE 면적의 **~21%** → 나머지 ~79%는 곱셈기+누적기+기타 |
| Variable Point (arXiv:2512.00186) | 고정소수점 설계 합성 | 곱셈기가 전체 면적의 **66%** |
| MDPI Electronics (2022) | 근사 FP 곱셈기 연구 | 가수 곱셈기 대상 최적화로 면적 **82% 절감** 가능 → 곱셈기가 면적 지배 |

**결론**: "곱셈기 60-80%"는 **근거 있음**. 독립형 FP 곱셈기에서는 80%에 가깝고, 전체 FMA/MAC 유닛(누적기 포함)에서는 50-70% 수준. 문서에서 제시한 60-70%는 합리적 범위 내.

### 검증 2: 포맷별 로직(지수/정렬/정규화)이 20-30%인가?

| 출처 | 측정 | 결과 |
|------|------|------|
| arXiv:2408.11997 | BF16 행렬 엔진 | 정규화 로직 = PE 면적의 **~21%** |
| MLSys 2021 (arXiv:2101.11748) | DNN 가속기 FP 오버헤드 분석 | 정렬 정밀도 58→8bit 축소 시 타일 면적 **17% 감소**, 누적 정밀도 추가 축소 시 **39% 감소** |
| VLSI 교과서 | 이론적 | 4~8bit 지수 덧셈기는 n-bit 곱셈기(n²) 대비 무시할 수준 |

**결론**: "포맷별 로직 20-30%"는 **근거 있음**. 정규화만 ~21%이고, 지수 처리/특수값 탐지는 추가로 소량. 단, 전체 FMA 유닛에서 누적기(adder tree)도 상당 면적을 차지하므로, "공유 부분 70-80%"보다는 **"공유 부분 60-75%"**가 더 보수적으로 정확.

### 검증 3: 곱셈기 면적 ∝ n² 인가?

**확인됨** (디지털 설계 교과서 수준의 사실):

- n-bit × n-bit 곱셈기는 n² 개의 부분곱(partial product)을 생성
- Wallace tree, Booth encoding 등의 최적화도 점근적 복잡도는 O(n²)
- Booth radix-4는 부분곱을 ~n²/4로 줄이지만 여전히 O(n²)

**출처**: Cornell ECE4740, Concordia COEN 6501, MDPI Applied Sciences (2025)

### 검증 4: 새 포맷 추가의 실리콘 비용

| 출처 | 설계 | 재구성 오버헤드 |
|------|------|---------------|
| ADiP (arXiv:2510.10623) | 적응형 정밀도 시스톨릭 배열 | 고정 정밀도 대비 **26-41%** 면적 증가 |
| FEDP (arXiv:2512.00053) | 다중 포맷 Dot Product 유닛 | HardFloat 대비 **40-55% LUT 절감** (공유 효과) |
| FFP8 (NVIDIA 참조) | 유연 FP8 포맷 지원 | 면적/지연 오버헤드 **<5%** |

**결론**: "곱셈기를 공유하므로 새 포맷 추가 비용이 상대적으로 작다"는 **맞지만**, "최소(minimal)"라는 표현은 맥락에 따라 다름. 단일 포맷 추가는 <5%~몇% 수준일 수 있으나, 완전 재구성 가능한(reconfigurable) 다중 포맷 지원은 26-41% 오버헤드. A100→H100에서 FP8 하나를 추가한 것은 전자에 가까울 것.

### 검증 5: E4M3 특수값 표현

FP8 Formats for Deep Learning (arXiv:2209.05433) 및 ONNX FP8 사양 확인:

| 포맷 | NaN | Inf | 비고 |
|------|-----|-----|------|
| **E4M3** | 가수 패턴 1개 (`S.1111.111`), 부호비트 포함 2개 비트 패턴 | **없음** (Inf를 포기하여 동적 범위 확장) | IEEE 비호환 |
| **E5M2** | 가수 패턴 3개 (`S.11111.{01,10,11}`), 부호 포함 6개 비트 패턴 | 있음 (`S.11111.00`) | IEEE 호환 |

원본 문서의 "E4M3은 NaN이 1개"는 소폭 부정확했음 → "가수 패턴 1개 (비트 패턴은 부호 포함 2개), Inf 없음"으로 이미 수정됨.

### 면적 비율 요약 (보정 후)

```
┌─────────────────────────────────────────────────┐
│  Tensor Core FMA 유닛 면적 구성 (추정)             │
│                                                  │
│  ┌────────────────────────────────┐              │
│  │  공유 곱셈기 배열    50~70%      │ ← 확인됨     │
│  └────────────────────────────────┘              │
│  ┌────────────────────────┐                      │
│  │  공유 누적기     10~20%  │ ← 포맷 간 공유       │
│  └────────────────────────┘                      │
│  ┌─────────────────┐                             │
│  │ 포맷별 로직 15~25% │ ← 지수/정렬/정규화/특수값   │
│  └─────────────────┘                             │
│  ┌──────┐                                        │
│  │기타 5%│ ← 파이프라인 레지스터, 제어 로직 등       │
│  └──────┘                                        │
│                                                  │
│  공유 부분 합계: ~65-85%                           │
│  포맷별 전용 부분: ~15-25%                         │
│  (정확한 비율은 설계/세대에 따라 상이)               │
└─────────────────────────────────────────────────┘
```

**핵심은 변하지 않는다**: 면적의 대부분을 차지하는 곱셈기가 공유되므로, 새로운 정밀도 추가의 실리콘 비용은 전체 면적 대비 상대적으로 작다. 단, 0은 아니기 때문에 설계 시점에 포함되어야 한다.

**출처**: Luo et al. (ScienceDirect), arXiv:2408.11997, arXiv:2101.11748, arXiv:2510.10623, arXiv:2512.00053, arXiv:2209.05433, ONNX FP8 Spec, Cornell ECE4740, MDPI Applied Sciences 2025

---

## 16. L4 (Ada Lovelace) vs H100 (Hopper) 아키텍처 비교

L4(SM89, Ada Lovelace)과 H100(SM90, Hopper)은 모두 FP8을 지원하지만, 내부 아키텍처는 근본적으로 다르다. 이 차이가 소프트웨어(CUDA 커널, 프레임워크)에 미치는 영향을 분석한다.

### 16.1 물리적 하드웨어 차이: Hopper 전용 기능 5가지

Ada Lovelace(SM89)는 본질적으로 **Ampere(SM80) + FP8 Tensor Core + RT Core 개선**이다. 프로그래밍 모델은 Ampere와 동일하다. 반면 Hopper(SM90)는 5가지 새로운 물리적 하드웨어를 추가했다.

| 하드웨어 기능 | L4 (SM89) | H100 (SM90) | 설명 |
|-------------|-----------|-------------|------|
| **TMA (Tensor Memory Accelerator)** | 없음 | 있음 | 1D~5D 텐서의 비동기 대량 복사를 하드웨어가 처리. 주소 계산, out-of-bound 체크 자동화 |
| **WGMMA (Warp Group MMA)** | 없음 | 있음 | 128스레드(4 Warp)가 비동기로 행렬곱. Operand B를 shared memory에서 직접 읽음 |
| **Thread Block Cluster** | 없음 | 있음 | 최대 16개 thread block을 같은 GPC에 co-scheduling 보장 |
| **DSMEM (Distributed Shared Memory)** | 없음 | 있음 | Cluster 내 SM간 shared memory 직접 접근 (~7x faster vs global memory) |
| **Shared Memory 228KB** | 100KB | 228KB | 2.28배 차이. WGMMA의 persistent kernel에 필수 |

이 5가지는 **소프트웨어로 에뮬레이션이 불가능**하다. 해당 PTX 명령어를 L4에서 실행하면 illegal instruction 에러가 발생한다.

### 16.2 MMA 명령어 수준의 차이

```
L4 (SM89): mma.sync 기반
┌────────────────────────────────────────────────┐
│  32 스레드 (1 Warp) 동기 실행                     │
│                                                │
│  PTX:  mma.sync.aligned.m16n8k32.f32.e4m3.e4m3 │
│  SASS: HMMA                                     │
│                                                │
│  A 입력: 레지스터에서                              │
│  B 입력: 레지스터에서                              │
│  출력 타일: 16×8                                  │
│                                                │
│  [TC 연산] → [대기] → [데이터 로드] → [TC 연산]    │
│  ████░░░░░░████░░░░░░████  (~63% 활용률)          │
└────────────────────────────────────────────────┘

H100 (SM90): wgmma 기반
┌────────────────────────────────────────────────┐
│  128 스레드 (4 Warps) 비동기 실행                  │
│                                                │
│  PTX:  wgmma.mma_async.m64nNk32.f32.e4m3.e4m3  │
│  SASS: HGMMA / IGMMA                            │
│                                                │
│  A 입력: 레지스터 또는 shared memory               │
│  B 입력: shared memory (직접 읽기)                 │
│  출력 타일: 64×N (N=8~256)                        │
│                                                │
│  [TC 연산 + TMA 데이터 로드 + 다음 명령 준비]       │
│  ████████████████████████  (~95%+ 활용률)          │
└────────────────────────────────────────────────┘
```

WGMMA의 핵심 장점:

1. **레지스터 압력 감소**: B 행렬을 shared memory에서 직접 읽어 레지스터 사용량 절감
2. **비동기 실행**: TC 연산과 메모리 접근이 겹침 (compute-memory overlap)
3. **넓은 타일**: 64×256 vs 16×8 → 스케줄링 오버헤드 감소
4. **TMA 연동**: 데이터 이동을 하드웨어가 자동 처리

### 16.3 PTX 명령어 지원 비교

| PTX 명령어 | L4 (SM89) | H100 (SM90) | 용도 |
|-----------|-----------|-------------|------|
| `mma.sync.aligned` | O | O (하위호환) | Warp-level 행렬곱 |
| `wgmma.mma_async` | **X** | O | Warp-group 비동기 행렬곱 |
| `wgmma.fence/commit/wait` | **X** | O | WGMMA 동기화 |
| `cp.async` | O | O | 비동기 복사 (global→shared) |
| `cp.async.bulk` | **X** | O | TMA 대량 복사 |
| `cp.async.bulk.tensor` | **X** | O | TMA 다차원 텐서 복사 |
| `cluster.sync` | **X** | O | Cluster 동기화 |
| `mbarrier.*` | 부분 지원 | 전체 지원 | 비동기 배리어 |
| `stmatrix` | 제한적 | O | Shared memory에 행렬 저장 |
| `setmaxnreg` | **X** | O | 동적 레지스터 수 조정 |

### 16.4 FP8 상세 비교: 같은 "FP8 지원"이지만 구현이 완전히 다름

| 속성 | L4 (SM89) | H100 (SM90) |
|------|-----------|-------------|
| FP8 E4M3/E5M2 지원 | O | O |
| FP8 MMA 명령어 | `mma.sync.m16n8k32` | `wgmma.mma_async.m64nNk32` |
| 참여 스레드 | 32 (1 Warp) | 128 (4 Warps) |
| 출력 타일 크기 | 16×8 | 64×N (N=8~256) |
| B 행렬 소스 | 레지스터 | shared memory (직접) |
| 내부 누적기 정밀도 | ~13 fractional bits (FP22) | ~13 fractional bits (FP22) |
| FP8 Dense TFLOPS | ~242 | ~1,979 |
| FP8 Sparse TFLOPS | 485 | 3,958 |
| 메모리 대역폭 | 300 GB/s | 3,350 GB/s (SXM) |
| FP8 최소 CUDA 버전 | CUDA 12.4 | CUDA 12.0 |

**주목**: 내부 누적기 정밀도는 둘 다 동일하게 ~13 fractional bits (FP22 수준)이다. 이 제한은 Blackwell(SM100)에서 25 fractional bits로 개선되었다.

### 16.5 소프트웨어 영향: 물리적 불가 vs 소프트웨어 갭 vs 둘 다 존재

vLLM 코드베이스에서 확인된 실제 분기점을 기반으로 세 가지로 분류한다.

#### (1) 물리적으로 불가능 — 하드웨어가 없어서 소프트웨어로 해결 불가

| 기능 | L4 (SM89) 동작 | 물리적 원인 | vLLM 코드 위치 |
|------|---------------|-----------|---------------|
| **FlashAttention-3** | FA2로 fallback | TMA + WGMMA 없음 | `vllm/v1/attention/backends/fa_utils.py:58` |
| **FP8 Attention** | BF16 FA2로 fallback | FA3 전용 (SM90+ 필수) | `vllm/v1/attention/backends/fa_utils.py:98-102` |
| **Block-quantized FP8 GEMM** | per-tensor FP8로 fallback | CUTLASS 3.x 전용 (TMA 필요) | `csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu:144-157` |
| **DeepGEMM** | CUTLASS 2.x FP8로 fallback | TMA 기반 설계 | `vllm/model_executor/layers/quantization/utils/fp8_utils.py:364` |
| **CUTLASS MoE Grouped GEMM** | Triton fused MoE로 fallback | SM90+ 전용 | `csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu:159-173` |
| **W4A8 혼합정밀도** | 미지원 | SM90 전용 커널 | `vllm/model_executor/layers/quantization/kernels/mixed_precision/cutlass.py:28-37` |
| **MLA (Multi-Latent Attention)** | 미지원 | FA3 + SM90 필수 | `vllm/v1/attention/backends/mla/flashattn_mla.py:67-69` |
| **Sink Tokens** | 미지원 | SM90+ 필수 | `vllm/v1/attention/backends/flash_attn.py:178-182` |
| **Shared Memory >100KB 커널** | 실행 불가 | 물리적 용량 100KB | 하드웨어 제약 |

#### (2) 지원 가능하지만 안 하는 것 — 소프트웨어 갭

| 기능 | 상태 | 이유 |
|------|------|------|
| SM89 FP8 GEMM (CUDA <12.4) | CUDA 12.4에서 해결됨 | NVCC가 SM89 FP8 코드 생성 미지원이었음 |
| SM89 Block FP8 (CUTLASS 2.x) | 이론적으로 가능하나 미구현 | TMA 없이 성능이 너무 나빠 구현 가치 없음 |
| SM89 Triton FP8 최적화 | 성능 문제 존재 | Triton issue #5583에 보고됨 |

#### (3) 둘 다 지원하지만 구현이 완전히 다른 것

| 기능 | L4 경로 | H100 경로 | 코드 위치 |
|------|--------|----------|----------|
| **FP8 per-tensor GEMM** | CUTLASS 2.x `cutlass_scaled_mm_sm89` | CUTLASS 3.x `cutlass_scaled_mm_sm90` | `csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu` |
| **Attention 백엔드 우선순위** | `FLASH_ATTN > FLASHINFER > TRITON` | `FLASHINFER > FLASH_ATTN > TRITON` | `vllm/platforms/cuda.py:44-82` |
| **FP8 CUTLASS 가속 모드** | `OpMultiplyAddFastAccum` (일반), `OpMultiplyAdd` (소규모 M) | `padded_cutlass()` (Hopper 전용) | `vllm/model_executor/layers/quantization/utils/fp8_utils.py:458-475` |

### 16.6 SM89 FP8 CUTLASS 2.x 커널의 특수 처리

L4에서 FP8 GEMM은 CUTLASS 2.x의 SM89 전용 디스패치 경로를 사용한다:

```
vllm/csrc/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_fp8_dispatch.cuh

- 아키텍처: cutlass::arch::Sm89 (enable_sm89_to_sm90 플래그 사용)
- MMA 명령어 타일: GemmShape<16, 8, 32> (Warp-level mma.sync)
- 기본 누적 모드: OpMultiplyAddFastAccum (빠르지만 ~FP22 정밀도)
- 소규모 M 감지 시: OpMultiplyAdd (느리지만 더 정확)
```

반면 H100의 CUTLASS 3.x 경로는:

```
- 아키텍처: SM90 (TMA + WGMMA 네이티브)
- MMA 명령어 타일: m64nNk32 (Warp-group wgmma)
- TMA로 데이터 비동기 프리페치
- Warp specialization: producer/consumer 패턴
```

### 16.7 FP8 누적기 정밀도: 공통 제한사항

```
       L4 (SM89)           H100 (SM90)
       ┌─────────┐         ┌─────────┐
FP8 →  │ mma.sync │   FP8 → │  wgmma  │
       │         │         │         │
       │ 누적기:  │         │ 누적기:  │
       │ ~FP22   │         │ ~FP22   │
       │ (13bit) │         │ (13bit) │  ← 둘 다 동일하게 제한됨
       └────┬────┘         └────┬────┘
            ↓                   ↓
       FP32 출력             FP32 출력    ← 표면적으로는 FP32이지만
       (실제 ~FP22)         (실제 ~FP22)     13 fractional bits만 유효

       vs Blackwell (SM100):
       ┌─────────┐
FP8 →  │  wgmma  │
       │ 누적기:  │
       │ ~FP32   │
       │ (25bit) │  ← Blackwell에서 개선됨
       └────┬────┘
            ↓
       FP32 출력 (실제 FP32 수준)
```

NVIDIA는 Ada Lovelace(SM89)에서 정밀도가 중요한 경우 FP8 MMA 대신 **FP16/BF16 MMA로 FP8 데이터를 처리할 것을 권장**한다. Throughput은 떨어지지만 누적 정밀도가 25 fractional bits로 높아진다.

### 16.8 요약: 왜 같은 "FP8 지원"인데 이렇게 다른가?

```
GPU 세대 진화:

Ampere (SM80, 2020)     Ada Lovelace (SM89, 2022)     Hopper (SM90, 2022)
┌──────────────┐        ┌──────────────┐              ┌──────────────┐
│ FP16, BF16   │        │ FP16, BF16   │              │ FP16, BF16   │
│ TF32, INT8   │──+FP8──│ TF32, INT8   │              │ TF32, INT8   │
│ mma.sync     │        │ mma.sync     │   완전 새설계  │ FP8          │
│ cp.async     │        │ cp.async     │──────────────│ wgmma        │
│              │        │              │              │ TMA          │
│              │        │              │              │ Cluster      │
│              │        │              │              │ DSMEM        │
└──────────────┘        └──────────────┘              └──────────────┘
        │                      │                            │
        └── 프로그래밍 모델 동일 ──┘    프로그래밍 모델 완전히 다름 ──┘
```

- Ada Lovelace(L4)는 **Ampere + FP8 추가**이다. 프로그래밍 모델은 Ampere와 동일.
- Hopper(H100)는 **완전히 새로운 프로그래밍 모델**. TMA, WGMMA, DSMEM, Cluster가 연동되어 동작.
- 이 차이 때문에 같은 FP8 연산이라도 **완전히 다른 커널 코드**로 작성되어야 한다.
- vLLM은 SM 버전을 감지하여 적절한 커널 경로로 자동 분기한다.

**출처**: NVIDIA Hopper Architecture In-Depth, NVIDIA Ada Lovelace Architecture, CUTLASS 3.x Documentation, vLLM 소스 코드 분석, FlashAttention-3 논문 (arXiv:2407.08608), NVIDIA PTX ISA Documentation, NVIDIA Ada/Hopper Tuning Guide

---

## 17. 확인된 사실 vs 강한 추론 vs 추측

### 확인된 사실 (NVIDIA 공식 또는 학술적 측정)

| 사실 | 출처 |
|------|------|
| 4×4×4 기본 연산, TC 수/SM, 지원 dtype, TFLOPS | NVIDIA 백서 |
| FP16 = BF16 throughput (모든 세대) | NVIDIA 공식 |
| TF32은 10-bit 가수 = FP16과 동일 | NVIDIA 공식 |
| Throughput 비율이 정확히 2의 거듭제곱 | NVIDIA 데이터시트 |
| 정밀도와 무관하게 거의 동일한 latency | 독립 마이크로벤치마크 |
| FP16/BF16/TF32가 HMMA, FP8/FP6가 QMMA 공유 | 독립 디스어셈블리 |
| 내부 누적기가 FP32보다 넓음 (23~25 fractional bits) | Fasi & Higham (2021) |
| 중간 덧셈이 round-towards-zero 사용 | Fasi & Higham (2021) |
| FP16 곱셈은 정확함 (반올림 없음) | Fasi & Higham (2021) |
| mma.sync는 Hopper에서 ~63% 피크, wgmma는 95%+ | 독립 벤치마크 |
| Ada/Hopper에서 FP8 누적기는 13 fractional bits (FP16의 25보다 적음) | MMA-Sim (2025) |
| WGMMA는 128스레드 비동기, mma.sync는 32스레드 동기 | NVIDIA PTX 문서 |
| Hopper에서 INT4는 TC가 아닌 CUDA Core(IMAD)에서 실행 | 독립 벤치마크 |

### 강한 추론 (다수의 독립적 증거가 수렴, 높은 신뢰도)

| 추론 | 근거 | 신뢰도 |
|------|------|--------|
| BF16과 FP16은 동일 곱셈기 사용 | 동일 throughput + 동일 SASS + 동일 latency | ~95% |
| TF32는 FP16 곱셈기 재사용 | 동일 10-bit 가수 + 1/2x throughput | ~90% |
| 낮은 정밀도 = 곱셈기 분할 (sub-word parallelism) | 2x/4x 비율 + 일정한 latency + NVDLA 선례 | ~85% |
| INT8은 FP 가수 곱셈기를 정수 모드로 재사용 | 동일 throughput 비율 + Intel 특허 + 가수곱셈=정수곱셈 원리 | ~80% |
| TC 내부는 Dot Product Unit 그리드 구조 | FEDP 학술 구현 + 일정 latency + 4×4 granularity | ~75% |

### 추측 (가능하지만 직접 증거 불충분)

| 추측 | 비고 | 신뢰도 |
|------|------|--------|
| FP64는 다수 사이클에 걸쳐 좁은 곱셈기를 반복 사용 | 1/16x throughput에서 유추. 별도 FP64 곱셈기일 수도 있음 | ~40% |
| Sparsity는 곱셈기 전단의 전처리 단계 | sparse/dense 동일 latency에서 유추 | ~60% |
| 시스톨릭 배열이 아닌 공간적 병렬 배열 | 학술 문헌에서 합의 없음 | ~50% |
| FP8의 줄어든 누적 정밀도(13bit)는 별도 누적 로직 때문 | 동일 하드웨어의 truncated 모드일 수도 있음 | ~50% |

### 아직 미확인 / 미공개

- Tensor Core의 게이트/트랜지스터 레벨 설계도
- 정확한 곱셈기 유닛 수와 파이프라인 깊이
- Tensor Core 간 자원 공유 여부
- 절대값 순서 누적의 물리적 구현 방식 (비교기 네트워크?)
- 다이 사진에서 개별 TC 해상도 식별 불가 (Ada Lovelace 기준 TC 면적 ≈ 0.04mm²)

---

## 18. 참고문헌

### NVIDIA 공식

- [NVIDIA Tesla V100 GPU Architecture Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA Hopper Architecture In-Depth Blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA Ampere Architecture In-Depth Blog](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [Accelerating AI Training with TF32 Tensor Cores](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)
- [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [NVIDIA, Arm, and Intel Publish FP8 Specification](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/)
- [H100 Transformer Engine Blog](https://blogs.nvidia.com/blog/h100-transformer-engine/)
- [Transformer Engine FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [NVDLA Hardware Architectural Specification](https://nvdla.org/hw/v1/hwarch.html)
- [CuTe MMA Atom Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html)
- [NVIDIA PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [GTC 2019: Tensor Core Performance, The Ultimate Guide](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf)
- [NVIDIA Ada Lovelace Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [CUTLASS 3.x SM90 Architecture Documentation](https://github.com/NVIDIA/cutlass)
- [CUTLASS Tutorial: Mastering TMA (Colfax Research)](https://research.colfax-intl.com/tutorial-hopper-tma/)

### 학술 논문 (마이크로벤치마킹)

- [Sun et al. - Dissecting Tensor Cores via Microbenchmarks (2022)](https://arxiv.org/abs/2206.02874)
- [Fasi & Higham - Numerical behavior of NVIDIA tensor cores (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7959640/)
- [Khattak & Mikaitis - Accurate Models of NVIDIA Tensor Cores (2024)](https://arxiv.org/abs/2512.07004)
- [MMA-Sim: Bit-Accurate Reference Model of Tensor Cores (2025)](https://arxiv.org/abs/2511.10909)
- [Microbenchmarking NVIDIA's Blackwell Architecture (2024)](https://arxiv.org/abs/2512.02189)
- [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks (2025)](https://arxiv.org/abs/2507.10789)
- [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking (2025)](https://arxiv.org/abs/2501.12084)
- [Benchmarking and Dissecting the NVIDIA Hopper GPU Architecture (2024)](https://arxiv.org/abs/2402.13499)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (2024)](https://arxiv.org/abs/2407.08608)

### 학술 논문 (하드웨어 설계)

- [Rethinking Floating Point Overheads for Mixed Precision DNN Accelerators (MLSys 2021)](https://arxiv.org/abs/2101.11748)
- [Fused FP8 4-Way Dot Product with Scaling and FP32 Accumulation (ARITH 2024)](https://www.ac.uma.es/arith2024/papers/)
- [A Configurable Mixed-Precision FEDP Unit for GPGPU Tensor Computation (2024)](https://arxiv.org/abs/2512.00053)
- [Low-Cost Multiple-Precision Multiplication Unit Design For Deep Learning (GLSVLSI 2023)](https://dl.acm.org/doi/10.1145/3583781.3590269)
- [ADiP: Adaptive Precision Systolic Array (2025)](https://arxiv.org/abs/2510.10623)
- [Intel 2.9 TOPS/W Reconfigurable Dense/Sparse Matrix-Multiply Accelerator (VLSI 2018)](https://ieeexplore.ieee.org/document/8502333/)
- [A Computational Model for Tensor Core Units (2019)](https://arxiv.org/abs/1908.06649)

### 산업 분석

- [SemiAnalysis: NVIDIA Tensor Core Evolution From Volta To Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
- [Computer Enhance: Zen, CUDA, and Tensor Cores, Part I: The Silicon](https://www.computerenhance.com/p/zen-cuda-and-tensor-cores-part-i)
- [Modal GPU Glossary: What is a Tensor Core?](https://modal.com/gpu-glossary/device-hardware/tensor-core)
- [CUTLASS Tutorial: WGMMA on Hopper (Colfax Research)](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)

### 특허

- [US20220083500A1 - Flexible Accelerator for Tensor Workloads](https://patents.google.com/patent/US20220083500A1/en)
- [EP4242838A2 - Shared mantissa/integer multiplication circuitry (Intel)](https://patents.google.com/patent/EP4242838A2/en)
- [US20170372202A1 - Tensor Processing Using Low Precision Format](https://patents.google.com/patent/US20170372202A1/en)

---

*작성일: 2026-02-06*
*면책: NVIDIA는 Tensor Core의 트랜지스터/게이트 레벨 설계를 공개한 적이 없습니다. 본 문서의 내부 구조 설명은 공식 문서, 마이크로벤치마킹 논문, 특허, 산업 분석을 종합한 추론입니다. 확인 수준은 각 섹션에 명시되어 있습니다.*
