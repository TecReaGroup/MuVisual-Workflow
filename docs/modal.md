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

浏览器不要直接调用 Modal。由 Web 的 Node API 验证用户身份、上传文件，并把
Modal 返回的 ZIP 原样转发给浏览器：

```txt
浏览器 -> Node API -> Modal -> Node API -> ZIP 下载
```

### Node API

部署后，将 `process-audio` 的完整 HTTPS URL 配置为 Node 服务端环境变量
`MODAL_URL`。不要使用 `NEXT_PUBLIC_` 等会暴露给浏览器的变量名。

下面使用 Node 20+ Web API 写法。`requireUser` 表示 Web 项目现有的 Session、JWT
或 OAuth 校验逻辑；未登录时必须在调用 Modal 前返回 `401`。

```js
export async function POST(request) {
  const user = await requireUser(request);
  if (!user) {
    return Response.json({ detail: "Unauthorized" }, { status: 401 });
  }

  const incoming = await request.formData();
  const file = incoming.get("file");
  if (!(file instanceof File) || file.size === 0) {
    return Response.json({ detail: "file is required" }, { status: 400 });
  }

  const form = new FormData();
  form.set("file", file, file.name);

  const modalResponse = await fetch(process.env.MODAL_URL, {
    method: "POST",
    body: form,
    redirect: "follow",
    signal: AbortSignal.timeout(60 * 60 * 1000),
  });

  const headers = new Headers();
  headers.set(
    "Content-Type",
    modalResponse.headers.get("Content-Type") ?? "application/octet-stream",
  );
  const disposition = modalResponse.headers.get("Content-Disposition");
  if (disposition) headers.set("Content-Disposition", disposition);

  return new Response(modalResponse.body, {
    status: modalResponse.status,
    headers,
  });
}
```

不要手动设置上传请求的 `Content-Type`，`FormData` 会生成正确的 multipart
boundary。Node API 也不要把 ZIP 转换成 JSON、Base64 或字符串。

当前 Modal 端点没有启用 Proxy Auth，调用它不需要 Modal 密钥。因此必须将
`MODAL_URL` 保存在服务端，并由 Node API 完成用户身份验证和访问控制。

### 浏览器上传和下载

浏览器只调用同源 Node API，现有 Session Cookie 会随同源请求自动携带；如果
Web 使用 Bearer Token，则在请求中添加对应的 `Authorization` Header。

```js
async function processAndDownload(file) {
  const form = new FormData();
  form.set("file", file, file.name);

  const response = await fetch("/api/muvisual/process", {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail ?? `处理失败：HTTP ${response.status}`);
  }

  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const encodedName = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  const filename = encodedName
    ? decodeURIComponent(encodedName)
    : "muvisual-output.zip";

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
```

上传字段名必须为 `file`。支持 `.wav`、`.flac`、`.mp3`、`.ogg`、`.opus`、
`.m4a`、`.aiff`、`.ac3`，音频必须包含非空的 `title` 和 `album` 标签。成功时
返回 `application/zip`；失败时返回 `{ "detail": "错误原因" }`。
