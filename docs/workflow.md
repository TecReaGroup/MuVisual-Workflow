# Workflow

读取音频标签，复制原始音频
→ 对整首歌运行一次 Beat This
→ 分离所有 stems
→ 仅对配置的乐器执行 noise gate
→ 仅对配置的乐器执行 Audio-to-MIDI 转录（其他的乐器文件夹只放入分离后的音频）
→ 对 midi 文件执行 normalizer

## File

```txts
data/output/
└── 歌名_专辑名/
    ├── 歌名_专辑名.mp3
    ├── 歌名_专辑名_beat.json
    ├── piano/
    │   ├── 歌名_专辑名_piano.mp3
    │   └── 歌名_专辑名_piano.mid
    ├── other/
    │   ├── 歌名_专辑名_other.mp3
    │   └── 歌名_专辑名_other.mid
    ├── vocals/
    │   └── 歌名_专辑名_vocals.mp3
    ├── bass/
    │   └── 歌名_专辑名_bass.mp3
    ├── drums/
    │   └── 歌名_专辑名_drums.mp3
    └── guitar/
        └── 歌名_专辑名_guitar.mp3
```
