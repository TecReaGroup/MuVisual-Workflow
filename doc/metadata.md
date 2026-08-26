# 音乐元数据

工作流会为每首歌曲生成一个歌曲级元数据文件：

```text
歌名/
└── 歌名_meta.json
```

节拍、调性和速度等信息只写入该 JSON 文件，不再由 `music_metadata` 步骤写入或改写 MIDI 文件。不同乐器的 workflow 会合并更新同一个文件，并保留已生成的其他乐器信息。

## 示例

```json
{
  "audio": "歌名.mp3",
  "beats": [0.48, 0.98, 1.48],
  "downbeats": [0.48],
  "instruments": {
    "piano": {
      "key": "C major",
      "bpm": 120.5,
      "delay": 0.032,
      "alignment": {
        "positive_count": 20,
        "negative_count": 15,
        "sample_count": 35,
        "error": 0.18,
        "average_error": 0.005142857142857143
      }
    }
  }
}
```

## 字段

- `audio`：歌曲目录中的原始音频文件名。
- `beats`：节拍时间点数组，单位为秒。
- `downbeats`：小节强拍时间点数组，单位为秒。
- `instruments`：按乐器名保存的识别结果，例如 `piano`、`drum`、`guitar`、`bass`。
- `key`：识别出的调性，例如 `C major`、`A minor`；无法判断时为 `Unknown`。
- `bpm`：该乐器 MIDI 识别出的全局速度，单位为每分钟拍数。
- `delay`：半拍网格相对时间零点的偏移量，单位为秒。
- `alignment.positive_count`：采样音符中更接近正向网格的数量。
- `alignment.negative_count`：采样音符中更接近镜像网格的数量。
- `alignment.sample_count`：参与网格对齐计算的音符数量。
- `alignment.error`：所有采样音符的网格对齐误差总和，单位为秒。
- `alignment.average_error`：平均网格对齐误差，单位为秒。
