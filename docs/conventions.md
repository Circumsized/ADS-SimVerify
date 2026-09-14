# 项目约定

## 命名

Rust 类型、trait、枚举和变体使用 PascalCase；函数、方法、字段、局部变量和模块使用 snake_case；常量和静态量使用 SCREAMING_SNAKE_CASE。缩写按普通单词处理，例如 `Url`、`btree_map`。

Python 遵循 PEP 8，函数和变量使用 snake_case，类使用 PascalCase。公开接口的名称、数据字段和节点端口属于稳定契约，不得为风格调整而重命名。

## 注释

注释应解释必要的设计理由、边界条件或安全约束，不重复代码表面含义。生产代码不保留越狱提示、营销话术、伪引用或无关历史叙述。代码与标识符使用英文；中文仅在确需面向操作者的注释或文档中出现。

## 可移植性

不得写入开发机绝对路径，尤其是 `/home/zhz` 和 `/run/media/zhz`。机器相关路径通过环境变量或仓库相对路径配置；`.env.example` 记录所需变量及本地默认值。CI 只依赖可声明安装的工具和运行时资源。

## 行为保持红线

重构不得改变节点 id、端口名、数据流拓扑、环境变量名、Arrow 字段名、CSV 表头、contracts JSON 键、acados 外部符号、模型文件及哈希、冻结验收指标。`contracts/**` 和 `assets/**` 是冻结证据与资源，R6 不修改。

`sensor_fusion` 已实现但当前未接入主链，属于已知记录，不作为本轮重构的修复范围。

## 验证

Windows 或不含 Isaac Sim 的环境可运行：

```bash
python -m unittest discover -s scripts -p "test_*.py"
python scripts/check_code_style.py
python scripts/check_repo_hygiene.py
cargo fmt --all -- --check
```

Linux 权威验证还应执行：

```bash
cargo fetch --locked
cargo build --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
dora run dora_dataflow.yaml
```

运行时验证应复核 Phase 6 的 12/12 到达、零碰撞、零求解失败及 33.06 ms 控制链 p95，并复核 Phase 7 的 17.07 ms 控制 p95。真机控制仍须通过独立的硬件准入门禁。
