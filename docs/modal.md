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

部署后，将 `process-audio` 的完整 HTTPS URL 配置为 Node.js 后端环境变量：

```env
MODAL_URL=https://YOUR_MODAL_ENDPOINT
```

下面的 Node.js 22+ 函数上传本地音频，并将返回的 ZIP 流式保存到指定路径：

```js
import { createWriteStream, openAsBlob } from "node:fs";
import { mkdir } from "node:fs/promises";
import { basename, dirname } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

export async function processAudio(inputPath, outputZipPath) {
  const form = new FormData();
  const audio = await openAsBlob(inputPath);
  form.set("file", audio, basename(inputPath));

  const headers = {};
  if (process.env.MODAL_KEY && process.env.MODAL_SECRET) {
    headers["Modal-Key"] = process.env.MODAL_KEY;
    headers["Modal-Secret"] = process.env.MODAL_SECRET;
  }

  const response = await fetch(process.env.MODAL_URL, {
    method: "POST",
    headers,
    body: form,
    redirect: "follow",
    signal: AbortSignal.timeout(60 * 60 * 1000),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Modal 请求失败 (${response.status}): ${detail}`);
  }
  if (!response.body) {
    throw new Error("Modal 返回了空响应");
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/zip")) {
    throw new Error(`Modal 返回的不是 ZIP: ${contentType || "unknown"}`);
  }

  await mkdir(dirname(outputZipPath), { recursive: true });
  await pipeline(
    Readable.fromWeb(response.body),
    createWriteStream(outputZipPath),
  );
  return outputZipPath;
}

await processAudio("./song.m4a", "./data/output/result.zip");
```

上传字段名必须为 `file`，不要手动设置 `Content-Type`，`FormData` 会自动生成
multipart boundary。接口支持 `.wav`、`.flac`、`.mp3`、`.ogg`、`.opus`、
`.m4a`、`.aiff`、`.ac3`，音频必须包含非空的 `title` 和 `album` 标签。

当前 Modal 端点没有启用 Proxy Auth，只需要配置 `MODAL_URL`。如果以后在
`fastapi_endpoint` 上启用 `requires_proxy_auth=True`，再配置 `MODAL_KEY` 和
`MODAL_SECRET`；代码会通过 `Modal-Key`、`Modal-Secret` Header 自动携带凭据。
