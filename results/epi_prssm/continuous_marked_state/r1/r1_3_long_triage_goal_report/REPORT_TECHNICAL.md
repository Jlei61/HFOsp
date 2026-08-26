# 长序列 T1 分诊与最小 H3：技术报告

## 验收

- T1 status: `COMPLETE`；H3 status: `COMPLETE`。
- scientific verdict: `H3_NOT_RUN_NO_PATIENT_MET_STATE_AND_INDEPENDENT_SUPPORT`。
- formal test opened: false；sealed opened: false。
- fixed subjects: yuquan_hanyuxuan, yuquan_chenziyang, yuquan_chengshuai。
- H3 scheduled jobs: 0；可解释 groups: 0。
- full module tests: 67；failures/errors: 0/0。

## T1 分层

- yuquan_hanyuxuan: target alignment 3/3；persistent 胜 memoryless 1/3；correct-time 胜 wrong-time 3/3；persistent joint 中位 +0.00023321 （timing +0.0056592，mark -0.0049346）；correct−wrong 中位 -6.4442e-05；端点中位 subset -0.0013015，continuation -0.0037134，STOP +0.012162，size -0.01222；起点/终点 distinct payload 3/3。
- yuquan_chenziyang: target alignment 0/3；persistent 胜 memoryless 0/3；correct-time 胜 wrong-time 1/3；persistent joint 中位 +3.3924e-05 （timing -0.00020469，mark +0.00022413）；correct−wrong 中位 +1.2063e-05；端点中位 subset +5.4698e-05，continuation +0.0010528，STOP +0.0062059，size -0.0071182；起点/终点 distinct payload 3/3。
- yuquan_chengshuai: target alignment 1/3；persistent 胜 memoryless 1/3；correct-time 胜 wrong-time 1/3；persistent joint 中位 +0 （timing +0，mark +0）；correct−wrong 中位 +0；端点中位 subset +0，continuation +0，STOP +0，size +0；起点/终点 distinct payload 3/3。

## 支持度

- yuquan_chengshuai: 选择 N=1000；完整对比实际需要 2000 events；TRAIN/validation 不重叠窗 8/3；validation 名义/完整支持时长中位 0.61/1.15 h。
- yuquan_chenziyang: 没有任何 N 同时达到 TRAIN/validation 各至少 3 个不重叠完整对比支持窗。
- yuquan_hanyuxuan: 没有任何 N 同时达到 TRAIN/validation 各至少 3 个不重叠完整对比支持窗。

```json
{
  "yuquan_chengshuai": {
    "candidate_windows": [
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 2000,
        "scale_events": 1000,
        "train": {
          "median_full_instrument_hours": 1.6063158333301544,
          "median_real_exposure_hours": 0.7066652777459886,
          "nonoverlapping_full_windows": 8,
          "nonoverlapping_real_exposure_windows": 15,
          "windows": 14545
        },
        "validation": {
          "median_full_instrument_hours": 1.1503625000185436,
          "median_real_exposure_hours": 0.6085438888602787,
          "nonoverlapping_full_windows": 3,
          "nonoverlapping_real_exposure_windows": 6,
          "windows": 5515
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 3000,
        "scale_events": 2000,
        "train": {
          "median_full_instrument_hours": 2.7191105555825765,
          "median_real_exposure_hours": 1.7963327777385711,
          "nonoverlapping_full_windows": 5,
          "nonoverlapping_real_exposure_windows": 7,
          "windows": 13545
        },
        "validation": {
          "median_full_instrument_hours": 1.5734675000111262,
          "median_real_exposure_hours": 1.1503625000185436,
          "nonoverlapping_full_windows": 2,
          "nonoverlapping_real_exposure_windows": 3,
          "windows": 5515
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 4000,
        "scale_events": 3000,
        "train": {
          "median_full_instrument_hours": 3.3959719444645775,
          "median_real_exposure_hours": 2.6211091666751436,
          "nonoverlapping_full_windows": 4,
          "nonoverlapping_real_exposure_windows": 5,
          "windows": 12545
        },
        "validation": {
          "median_full_instrument_hours": 2.053313333325916,
          "median_real_exposure_hours": 1.5734675000111262,
          "nonoverlapping_full_windows": 2,
          "nonoverlapping_real_exposure_windows": 2,
          "windows": 5515
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 5000,
        "scale_events": 4000,
        "train": {
          "median_full_instrument_hours": 4.121185277766652,
          "median_real_exposure_hours": 3.3692136110862094,
          "nonoverlapping_full_windows": 3,
          "nonoverlapping_real_exposure_windows": 3,
          "windows": 11545
        },
        "validation": {
          "median_full_instrument_hours": 2.478151666654481,
          "median_real_exposure_hours": 2.053313333325916,
          "nonoverlapping_full_windows": 2,
          "nonoverlapping_real_exposure_windows": 2,
          "windows": 5515
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 6000,
        "scale_events": 5000,
        "train": {
          "median_full_instrument_hours": 5.454617499974039,
          "median_real_exposure_hours": 3.883247777753406,
          "nonoverlapping_full_windows": 2,
          "nonoverlapping_real_exposure_windows": 3,
          "windows": 10545
        },
        "validation": {
          "median_full_instrument_hours": 3.0103202777438693,
          "median_real_exposure_hours": 2.478151666654481,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 2,
          "windows": 5515
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 11000,
        "scale_events": 10000,
        "train": {
          "median_full_instrument_hours": 12.192251944409476,
          "median_real_exposure_hours": 10.229129722184606,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 5545
        },
        "validation": {
          "median_full_instrument_hours": 6.027375000052982,
          "median_real_exposure_hours": 5.2812141666147445,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 5515
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 16000,
        "scale_events": 15000,
        "train": {
          "median_full_instrument_hours": 15.501746666696336,
          "median_real_exposure_hours": 14.620290833380487,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 545
        },
        "validation": {
          "median_full_instrument_hours": 13.849678333335453,
          "median_real_exposure_hours": 11.72529166665342,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 5515
        }
      }
    ],
    "chosen": {
      "causal_delay_events": 1000,
      "full_instrument_support_events": 2000,
      "scale_events": 1000,
      "train": {
        "median_full_instrument_hours": 1.6063158333301544,
        "median_real_exposure_hours": 0.7066652777459886,
        "nonoverlapping_full_windows": 8,
        "nonoverlapping_real_exposure_windows": 15,
        "windows": 14545
      },
      "validation": {
        "median_full_instrument_hours": 1.1503625000185436,
        "median_real_exposure_hours": 0.6085438888602787,
        "nonoverlapping_full_windows": 3,
        "nonoverlapping_real_exposure_windows": 6,
        "windows": 5515
      }
    },
    "design": "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/r1_2/cache/yuquan_chengshuai/full_design.npz",
    "design_sha256": "42706f2d6a001eeec12b9fde6df3f670bd3eeb4591968800411e0032a6ca34d9",
    "minimum_nonoverlapping_each_split": 3,
    "subject": "yuquan_chengshuai"
  },
  "yuquan_chenziyang": {
    "candidate_windows": [
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 2000,
        "scale_events": 1000,
        "train": {
          "median_full_instrument_hours": 3.134868333339691,
          "median_real_exposure_hours": 1.6009038889076974,
          "nonoverlapping_full_windows": 2,
          "nonoverlapping_real_exposure_windows": 4,
          "windows": 3764
        },
        "validation": {
          "median_full_instrument_hours": 3.06656972222858,
          "median_real_exposure_hours": 1.3208358333508174,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 2,
          "windows": 1281
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 3000,
        "scale_events": 2000,
        "train": {
          "median_full_instrument_hours": 4.67112583335903,
          "median_real_exposure_hours": 3.159036388893922,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 2,
          "windows": 2764
        },
        "validation": {
          "median_full_instrument_hours": 4.706470555596882,
          "median_real_exposure_hours": 3.06656972222858,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 1281
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 4000,
        "scale_events": 3000,
        "train": {
          "median_full_instrument_hours": 6.079302083386315,
          "median_real_exposure_hours": 4.756900972227255,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 1764
        },
        "validation": {
          "median_full_instrument_hours": 6.165169722239177,
          "median_real_exposure_hours": 4.706470555596882,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 1281
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 5000,
        "scale_events": 4000,
        "train": {
          "median_full_instrument_hours": 8.49535041666693,
          "median_real_exposure_hours": 6.08599111109972,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 764
        },
        "validation": {
          "median_full_instrument_hours": 7.396485555569331,
          "median_real_exposure_hours": 6.165169722239177,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 1281
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 6000,
        "scale_events": 5000,
        "train": {
          "median_full_instrument_hours": null,
          "median_real_exposure_hours": null,
          "nonoverlapping_full_windows": 0,
          "nonoverlapping_real_exposure_windows": 0,
          "windows": 0
        },
        "validation": {
          "median_full_instrument_hours": 9.466561944484711,
          "median_real_exposure_hours": 7.405159999993113,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 1045
        }
      }
    ],
    "chosen": null,
    "design": "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/r1_2/cache/yuquan_chenziyang/full_design.npz",
    "design_sha256": "31980187c7a767dd2f5630c3efabd164e67b10f0cc197a5a97c1a07637aec5da",
    "minimum_nonoverlapping_each_split": 3,
    "subject": "yuquan_chenziyang"
  },
  "yuquan_hanyuxuan": {
    "candidate_windows": [
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 2000,
        "scale_events": 1000,
        "train": {
          "median_full_instrument_hours": 10.265804444419013,
          "median_real_exposure_hours": 4.517599722213215,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 2,
          "windows": 1279
        },
        "validation": {
          "median_full_instrument_hours": 9.579695833325387,
          "median_real_exposure_hours": 5.707590555581781,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 2,
          "windows": 1094
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 3000,
        "scale_events": 2000,
        "train": {
          "median_full_instrument_hours": 14.804884444408946,
          "median_real_exposure_hours": 10.522804166674614,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 279
        },
        "validation": {
          "median_full_instrument_hours": 15.692597083350023,
          "median_real_exposure_hours": 9.579695833325387,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 1094
        }
      },
      {
        "causal_delay_events": 1000,
        "full_instrument_support_events": 4000,
        "scale_events": 3000,
        "train": {
          "median_full_instrument_hours": null,
          "median_real_exposure_hours": null,
          "nonoverlapping_full_windows": 0,
          "nonoverlapping_real_exposure_windows": 0,
          "windows": 0
        },
        "validation": {
          "median_full_instrument_hours": 19.306802777780426,
          "median_real_exposure_hours": 14.760038611094156,
          "nonoverlapping_full_windows": 1,
          "nonoverlapping_real_exposure_windows": 1,
          "windows": 373
        }
      }
    ],
    "chosen": null,
    "design": "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/r1_2/cache/yuquan_hanyuxuan/full_design.npz",
    "design_sha256": "ecafc39ffbc38c6a23f9630b1ecf5b12f249b2090f9686a2fe0f191cace4c450",
    "minimum_nonoverlapping_each_split": 3,
    "subject": "yuquan_hanyuxuan"
  }
}
```

## H3 对比

- 没有患者同时满足可用跨窗口状态和 TRAIN/validation 各至少 3 个不重叠长窗口，因此本轮没有为了凑结果而运行新的人体 H3。

## 方法更正

- R1.3 H3 入口读取真实 persistent−memoryless 符号，不再自动写 `True`；
- 主对比仅为 real−intercept-matched 与 real−causal-delayed；
- participation exposure 先去除总 load，再用 TRAIN-only 条件残差的两个 PCA 分量；
- boxcar 支持多维 exposure，并在 TRAIN-only decoder space 拟合；
- H3 运行前要求 TRAIN/validation 各至少 3 个不重叠整窗。

## 复现入口

- T1 summary: `/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/r1_3_long_t1_triage/summary.json`
- H3 support: `/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/r1_3_long_h3_followup/support_audit.json`
- H3 summary: `/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/r1_3_long_h3_followup/summary.json`
- machine audit: `/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/r1_3_long_triage_goal_report/machine_audit.json`
