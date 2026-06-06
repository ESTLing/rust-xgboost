# Rust XGBoost Binding — 长期维护策略设计

> 版本 0.x，紧跟 XGBoost 最新版，API 对标 Python 核心层厚度

---

## 一、API 设计原则

**核心方针：Python core API（非 sklearn 层）的 Rust 翻译，不擅自加戏。**

| 问题 | 决策 |
|------|------|
| 参数怎么传？ | `&[(&str, &str)]` 平铺 key-value，和 Python `dict` 一样 |
| 参数 struct 要吗？ | 不。零参数 struct，用户照着 XGBoost 文档直接写 string |
| enum 要吗？ | 不。`"gbtree"`、`"binary:logistic"` 都是 string |
| Interval/range 校验要吗？ | 不。参数传错 XGBoost runtime 自己报，Rust 层不兜底 |
| 训练入口 | `Booster::train()` 自由函数，直接收 `&[(&str, &str)]` |
| 功能覆盖范围 | 对标 Python 核心 API：DMatrix、Booster、train、predict、save/load |

**API 形态：**

```rust
use xgb::{DMatrix, Booster};

let mut dtrain = DMatrix::from_dense(&data, num_rows)?;
dtrain.set_labels(&labels)?;

// 参数就是 key-value，对着 XGBoost 文档抄
let bst = Booster::train(
    &[
        ("max_depth", "6"),
        ("eta", "0.3"),
        ("objective", "binary:logistic"),
        ("eval_metric", "logloss"),
    ],
    &dtrain,
    10,
    Some(&[(&dtest, "test")]),
)?;

let preds = bst.predict(&dtest)?;
bst.save("model.json")?;
```

---

## 二、FFI 层 (`xgboost-sys`) 改造

**参照 [aryehlev/xgboost-rust](https://github.com/aryehlev/xgboost-rust/blob/main/build.rs)，从 PyPI wheel 下载共享库。**

### build.rs 新流程

```
1. docs.rs 安全阀 → CARGO_CFG_DOCSRS / DOCS_RS 环境检测，跳过联网
2. PyPI JSON API 动态查询 wheel URL（https://pypi.org/pypi/xgboost/{version}/json）
3. GitHub raw 下载 c_api.h + base.h（按版本拼接 URL，SHA256 校验）
4. bindgen 生成绑定（只导出 XGB*/XGD* 函数 + 核心类型）
5. PyPI wheel 下载 + zip 解压取出 .so/.dylib/.dll
6. 设置 rpath（macOS: @loader_path, Linux: $ORIGIN）
7. cargo:rerun-if-changed=build.rs
```

### 关键细节

**PyPI JSON API（不硬编码 URL）：**
- 请求 `https://pypi.org/pypi/xgboost/{version}/json` 获取所有 wheel 文件名和下载地址
- 按 `filename.contains(platform_keyword)` 匹配：Linux `manylinux`、Windows `win_amd64`、macOS `macosx`
- 好处：PyPI manylinux tag 升级（如 `manylinux2014` → `manylinux_2_28`）URL 自动跟上

**docs.rs 安全阀（必须）：**
- `build.rs` 开头检测 `CARGO_CFG_DOCSRS` 或 `DOCS_RS` 环境变量
- 命中则 `return`，跳过所有网络操作
- `Cargo.toml` 配置 `[package.metadata.docs.rs].rustdoc-args = ["--cfg", "docsrs"]`
- 不处理 → docs.rs 沙盒无网络 → 文档发布永远失败

**依赖（build-dependencies，全部纯 Rust）：**
- `ureq` — 轻量 HTTP，用于 PyPI JSON API + wheel 下载
- `zip` — wheel 解压
- `serde` + `serde-json` — 解析 PyPI JSON 响应
- 用户不需要安装 Python、pip、cmake、ninja 等任何系统工具

**平台匹配（不硬编码文件名）：**

| OS | Arch | 匹配关键字 |
|----|------|-----------|
| Linux x86_64 / aarch64 | `manylinux` | |
| macOS x86_64 / aarch64 | `macosx` | |
| Windows x86_64 | `win_amd64` | |

### 版本控制

- 默认 XGBoost 版本：`3.2.0`
- `XGBOOST_VERSION` 环境变量覆盖默认版本
- SHA256 校验映射表维护在 build.rs 中，每个支持的 XGBoost 版本各一份
- **bindings.rs 提交到仓库**，用户不需要 clang/llvm

### Windows 支持

- Windows 无 rpath 概念，dll 复制到 target 目录即可
- 本地构建不提供 Windows 支持（pre-built wheel only）

---

## 三、版本号与发布

| 项目 | 策略 |
|------|------|
| 起始版本 | `0.1.0`，表示不稳定 API |
| 版本管理 | `[workspace.package]` 统一管理 `xgb` 和 `xgboost-sys` |
| CHANGELOG | 删除旧的，从 `0.1.0` 开始，格式用 [Keep a Changelog](https://keepachangelog.com/) |
| 发布节奏 | XGBoost 新版本发布后 1-2 天内跟进 |
| 发布物 | crates.io (`xgb` + `xgboost_lib-sys`) |

---

## 四、unsafe 安全边界

**目标：从 safe Rust 不可能触发未定义行为。**

| 必须修 | 位置 | 做法 |
|--------|------|------|
| 🔴 | `Booster::drop` `xgb_call!(XGBoosterFree).unwrap()` | 静默处理，`error!()` 打 log |
| 🔴 | `DMatrix::drop` `xgb_call!(XGDMatrixFree).unwrap()` | 同上 |
| 🔴 | `assert!(!out_result.is_null())` | `assert!` 在 release 被优化掉，改为 if-is-null-return-Err |
| 🔴 | `CStr::from_ptr(...).to_str().unwrap()` | 非 UTF-8 转 `Err`，不 panic |
| 🟡 | `DMatrix::handle` 当前是 `pub(super)` | 考虑收紧可见性 |

**不需要的：**
- ~~FFI 调用加 `// SAFETY:`~~ → 这就是正常调用，不需要注释
- `from_raw_parts` / 裸指针解引用等有所有权/生命周期细节的地方，按实际情况适当说明

---

## 五、CI/CD

### 触发条件
`push` (main) + `pull_request`

### 矩阵

```
├── Linux (ubuntu-latest)
│   ├── cargo build --verbose
│   ├── cargo test --verbose
│   ├── cargo clippy -- -D warnings
│   ├── cargo fmt --check
│   └── cargo doc --no-deps --document-private-items
│
├── macOS (macos-latest, arm64)
│   └── cargo build + test
│
├── Windows (windows-latest, x86_64)
│   └── cargo build + test
│
└── Quality Gate
    └── cargo-deny check (license / advisory / source)
```

### 清理

- 删除 `.travis.yml`（已被 GitHub Actions 取代）
- 删除 `.github/workflows/linux_arm64.yml`（并入主 Linux job 或用 cross-compile）

---

## 六、文档

| 媒介 | 内容 |
|------|------|
| `lib.rs` doc comments | 主文档来源 → docs.rs |
| README | 快速上手 + 安装说明（pre-built / local build）+ XGBoost 版本对应表 |
| `examples/` | 保持现有 4 个 + 考虑补充 save/load、cv |
| CHANGELOG | 每次 release 更新 |

**不需要：** 独立文档网站、贡献指南

---

## 七、实施路线

### Phase 1 — 安全底线（优先）
- Drop 去 unwrap
- assert → if-is-null-return-Err
- UTF-8 panic → error

### Phase 2 — build.rs 重写
- PyPI wheel 下载 + zip 解压
- GitHub raw 下载 header + SHA256 校验
- 删除本地 `xgboost-sys/lib/` 预编译库

### Phase 3 — API 简化
- 删除 `src/parameters/` 整个目录
- 删除 `derive_builder`、`indexmap` 依赖
- `Booster::train()` 直接收 `&[(&str, &str)]`
- `Booster::new()` 直接收 `&[(&str, &str)]`

### Phase 4 — 工程化
- 版本切 0.1.0
- `[workspace.package]` 统一版本
- 重写 CHANGELOG
- 清理 CI
