# modal

编写 modal 相关代码支持（可以参考 temp/muscriptor-deploy 相关代码），从而使得 文件处理 api 化，api 使用相关内容需要追加写入 docs/modal.md 文件中

输入 一个 audio 文件，处理完返回压缩包文件，解压后格式为

```txt
歌名_专辑名/
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

## vps

vps 解压后覆盖性放到对应位置

## api

Modal 入口位于 `src/muvisual_workflow/modal_app.py`，以 Python 模块方式部署。

首次使用时安装依赖并完成 Modal 登录：

```bash
make install
make modal-setup
```

部署 API：

```bash
make modal-deploy
```

批量处理本地 `data/input` 中的音频：

```bash
make modal
```

该命令按文件顺序执行，每个音频发起一次独立的 Modal GPU 调用。每次调用返回
一个 ZIP，客户端校验压缩包内容后，将其中的 `歌名_专辑名/` 覆盖解压到
`data/output`。单个文件失败不会阻止后续文件继续处理，全部结束后会汇总失败。

部署完成后，Modal 会输出 `process-audio` HTTPS 地址。接口使用 L40S GPU，
模型文件缓存在持久化的 `muvisual-model-cache` Volume 中；第一次请求需要下载
模型，因此耗时会更长。

请求使用 `multipart/form-data`，文件字段名必须为 `file`。音频文件必须包含
`title` 和 `album` 标签，它们将组成压缩包内的目录名和文件名前缀。

```bash
curl -L -X POST "https://YOUR_MODAL_ENDPOINT" \
  -F "file=@./song.flac" \
  -o "song.zip"
```

支持 `.wav`、`.flac`、`.mp3`、`.ogg`、`.opus`、`.m4a`、`.aiff` 和
`.ac3`。成功响应的 Content-Type 为 `application/zip`。处理超过 Modal 单次
HTTP 等待窗口时会产生 303 结果跳转，因此 curl 必须使用 `-L`。

- HTTP 400：上传文件为空。
- HTTP 415：文件扩展名不受支持。
- HTTP 422：音频标签无效或流水线处理失败。
