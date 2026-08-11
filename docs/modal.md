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

### 部署

Modal 入口位于 `src/muvisual_workflow/modal_app.py`，以 Python 模块方式部署。

```bash
make install
make modal-setup
make modal-deploy
```

部署成功后终端会输出 `process-audio` 的 HTTPS URL。后续示例中的
`$MODAL_URL` 或 `MODAL_URL` 均表示这个完整 URL。

接口使用 L40S GPU，模型缓存在持久化的 `muvisual-model-cache` Volume 中。
第一次请求需要下载模型，耗时会明显长于后续请求。

### 接口约定

```txt
POST $MODAL_URL
Content-Type: multipart/form-data
```

请求体只有一个字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | File | 是 | 单个音频文件，字段名必须为 `file` |

支持的文件扩展名：`.wav`、`.flac`、`.mp3`、`.ogg`、`.opus`、`.m4a`、
`.aiff`、`.ac3`。文件必须包含非空的 `title` 和 `album` 音频标签，服务端使用
这两个标签生成 `歌名_专辑名` 输出目录。

不要手动设置请求的 `Content-Type`。浏览器、curl 或 HTTP SDK 需要自行生成
`multipart/form-data` 的 boundary。

成功响应：

```txt
HTTP/1.1 200 OK
Content-Type: application/zip
Content-Disposition: attachment; filename="muvisual-output.zip"; filename*=UTF-8''...
```

响应体是 ZIP 二进制数据，不能按 JSON 或文本读取。ZIP 根目录为
`歌名_专辑名/`，其内容与本文开头的目录结构一致。

失败响应使用 FastAPI JSON 格式：

```json
{
  "detail": "错误原因"
}
```

| 状态码 | 含义 |
| --- | --- |
| `400` | 上传文件为空 |
| `415` | 文件扩展名不受支持 |
| `422` | 缺少 `file` 字段、音频标签无效或流水线处理失败 |
| `500` | Modal 或服务端发生未处理错误 |

### Web 接入方式

当前接口是同步的长耗时接口。Modal Web Function 单次 HTTP 等待窗口为 150 秒，
超过后会返回 303 并通过结果 URL 继续等待。该跳转机制不兼容跨域 CORS 请求，
而且当前端点没有开放浏览器跨域访问，因此生产 Web 页面不要直接请求 Modal URL。

推荐调用链：

```txt
浏览器 -> Web 应用同源后端 -> Modal process-audio -> ZIP -> 浏览器下载
```

Web 后端代理需要：

1. 接收浏览器提交的 `multipart/form-data`。
2. 将单个 `file` 转发给 Modal，保持原始文件名和 MIME 类型。
3. 启用 HTTP 重定向跟随，并将请求超时设置为整个音频处理可接受的时长。
4. 流式转发 Modal 返回的 ZIP、`Content-Type` 和 `Content-Disposition`。
5. 不要把 ZIP 转换为 JSON、Base64 或字符串。

浏览器调用同源代理并下载 ZIP：

```js
async function processAndDownload(file) {
  const form = new FormData();
  form.append("file", file, file.name);

  const response = await fetch("/api/muvisual/process", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail ?? `处理失败：HTTP ${response.status}`);
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/zip")) {
    throw new Error(`响应不是 ZIP：${contentType || "unknown"}`);
  }

  const disposition = response.headers.get("content-disposition") ?? "";
  const encodedName = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  const filename = encodedName
    ? decodeURIComponent(encodedName)
    : "muvisual-output.zip";

  const blob = await response.blob();
  if (blob.size === 0) {
    throw new Error("下载的 ZIP 为空");
  }

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
```

### curl 验证

Linux/macOS：

```bash
export MODAL_URL="https://YOUR_MODAL_ENDPOINT"
curl --fail-with-body --show-error --location \
  --dump-header response.headers \
  --form "file=@./song.flac" \
  --output result.zip \
  --write-out "HTTP %{http_code}, %{content_type}, %{size_download} bytes\n" \
  "$MODAL_URL"

unzip -t result.zip
unzip -l result.zip
```

Windows PowerShell 应显式使用 `curl.exe`，避免调用 PowerShell 的 curl 别名：

```powershell
$env:MODAL_URL = "https://YOUR_MODAL_ENDPOINT"
curl.exe --fail-with-body --show-error --location `
  --dump-header response.headers `
  --form "file=@.\song.flac" `
  --output result.zip `
  --write-out "HTTP %{http_code}, %{content_type}, %{size_download} bytes\n" `
  $env:MODAL_URL

tar -tf result.zip
Expand-Archive -LiteralPath result.zip -DestinationPath data\output -Force
```

验证成功时应满足：HTTP 状态码为 `200`、Content-Type 包含
`application/zip`、下载字节数大于零，且 `tar -tf` 或 `unzip -t` 没有报告
ZIP 损坏。

### 本地批量调用

不经过 Web URL，也可以逐个调用 Modal GPU Function 处理 `data/input`：

```bash
make modal
```

每个音频对应一次独立的 Modal 调用。客户端收到 ZIP 后进行路径校验，再将
`歌名_专辑名/` 覆盖解压到 `data/output`。单个文件失败不会阻止后续文件继续
处理，全部结束后会汇总失败。
