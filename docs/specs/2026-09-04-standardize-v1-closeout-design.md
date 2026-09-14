# Standardize-v1 收口设计（2026-09-04）

## 目标

把当前 `refactor/standardize-v1` 分支上未提交的标准化改动收束成可信基线：
门禁可绿、构建可移植、运行时安全边界 fail-closed、CI 分层明确。

## 冻结边界（不可破坏）

- 节点 id、端口名、Arrow 字段名、环境变量名、拓扑语义不变。
- `contracts/**`、`assets/**`、`model/**` 及模型哈希不变。
- Phase 6（12/12、零碰撞、p95 33.06 ms）与 Phase 7（17.07 ms）指标不变。
- `real_vehicle_control_allowed=false` 保持；真机六门禁不被触碰。
- 根 `dora_dataflow.yaml` 不改名为生产/真机拓扑。

## 已确认证据

- `python -m unittest discover -s scripts` 有 5 个错误：
  `test_repo_hygiene.py::load_checker` 用 `spec_from_file_location`
  动态导入含 `@dataclass` 的模块未注册 `sys.modules`，dataclass PEP 563 解析崩溃。
- `python scripts/check_code_style.py` 失败：
  `brain/spiced_rl_trainer/utils/freeze_phase0_baseline.py` 与
  `core-control/build.rs` 含 `/home/zhz`；`expert_logger_node.py`、
  `keyboard_teleop.py`、`teleop_sender.py` 含 emoji/禁词；`dora_dataflow.yaml`
  节点 id 集合缺 `telemetry_dashboard`。
- `cargo check --locked` 失败：`Cargo.lock` 缺少 `core-perception` 新增
  `cuda` feature 引入的依赖锁定，需要 `cargo update --workspace` 重新生成。
- `cargo fmt` 工具链组件缺失（rustfmt 未安装），本地按可用性分层执行。
- Dora `send_output` 返回 `NodeResult<()>`；当前 `fast_brain_node.rs`
  所有安全输出均 `let _ =` 丢弃，通道断开时控制任务无感知。
- 执行器 `isaac_dora_node.py` 收 `control_cmd` 仅检查长度 2，
  未做 finite/限幅；freshness 以仿真 tick 计量，非墙钟。
- 慢脑 `slow_brain_node.rs` XFeat Arrow 解码链式 `unwrap()`，
  workspace `panic=abort` 下畸形帧会终止进程。
- `build.rs` 把 `rerun-if-env-changed` 写成 `key=value` 格式（Cargo 只认 `key`），
  且对所有目标注入 GNU 链接参数（破坏 MSVC）。

## 实施切片

1. **脚本门禁**：修 `load_checker` 注册 `sys.modules`（`@dataclass` + PEP 563
   动态导入崩溃）；`check_repo_hygiene` 的 `git_lines` 失败转为 `other_errors`
   并 guard README 读取（非 Git 目录产出结构化报告而非异常）；风格检查：清理
   真实文件中的 `/home/zhz`、禁词、emoji；`code_without_strings_and_comments`
   增加 `.yaml` 的 `#` 注释剥离（此前 yaml 走 Rust 分支未剥 `#`，注释中文被误判
   CJK-in-code）；新增 `test_real_repository_passes_style_gate` 真实仓库零错误回归。
   `dora_dataflow.yaml` 的 `telemetry_dashboard` 节点：committed 版存在、被继承的
   未提交工作删除，导致 `validate_contracts.py` 与冻结 `data_contracts.json` 的 6 条
   边不一致（stale edge）。因 `contracts/**` 冻结且"不改拓扑"是红线，收口选择
   **恢复该节点**（仅把禁用的绝对路径换成 `${UV_CACHE_DIR}`），`check_code_style`
   的 `NODE_IDS` 相应保留 `telemetry_dashboard`。
2. **构建可移植**：`Cargo.lock` 重新锁定；`build.rs` 修 rerun-if-env-changed
   格式；GNU 专属 `rustc-link-arg` 用 `CARGO_CFG_TARGET_ENV=gnu` 门控；
   缺 acados 目录时不再无条件 link（警告 + 跳过链接，feature 化 wheel 能力
   留待后续，本轮不新增 feature 名以避免 API 变更）。
3. **输出安全**：`fast_brain_node.rs` 收敛单一 `send_control_cmd()`，
   返回 `Result`；发送失败时 eprintln 记录并终止控制任务（发送任务 join 后
   主事件循环退出），符合"失去输出能力即失去控制资格"。
4. **执行器输入边界**：从 `isaac_dora_node.py` 提取
   `parse_control_command(cmd_arr)`（长度/finite/限幅，违规返回 None）与
   `is_control_fresh(last_valid_monotonic, now)`（单调时钟），
   节点主循环调用；`simulation-env/test_isaac_control_gate.py` 单测。
5. **慢脑 Arrow 边界**：提取 `decode_xfeat_struct(&StructArray) -> Result<Vec<SparseFeature>, String>`
   纯函数（校验列存在、类型、x/y/score/descriptor 长度一致、descriptor 固定宽度 64、
   keypoint 与 descriptor 数值有限），调用处 `match` 处理：坏帧 `eprintln` + `continue`
   丢弃，不 panic（工作区 `panic=abort`）。附 5 个单元测试（合法/缺列/长度不符/非有限/
   宽度不符），用 arrow 54 稳定的 `FixedSizeListArray::new` 与 `StructArray::from` 构造。
   已知降级：为匹配原文件已验证可编译的 API 面（原文件未导入 `Array` trait 却调用
   `.len()`），未使用 `is_null()` 显式拒绝 null 槽（null 经 `value()` 读作 0.0，
   不 panic，但不会被判为无效）；`state.nav_route[current_target_index]` 的越界风险
   为既有代码，超出本切片范围，记为残留风险。
6. **CI 与总验证**：`core-control` 因链接 acados C 求解器、CI 无 acados，保持
   `cargo check --all-targets`（不升级为 clippy/test，避免引入无法本地验证的
   `-D warnings` 红灯），并加注释说明 wheel-capable 构建与运行测试是 Linux+acados
   宿主的分层门禁；Python `python-hygiene` job 增加 `simulation-env` 无依赖单测发现
   （`test_isaac_control_gate.py`）。真实仓库零错误回归由切片 1 的
   `test_real_repository_passes_style_gate` 承担。

## 验收标准

- Windows 本机：Python 单测全绿、`check_code_style.py` 零错误、
  `check_repo_hygiene.py` ok、`cargo check -p core-perception -p core-decision`
  通过、新单测通过。
- 无法在 Windows 执行项（Isaac、acados、cargo test -p core-control 的
  wheel 依赖部分、Linux rustfmt/clippy）逐项列出不执行原因。
- `git diff --check` 干净；冻结契约验证不回归。
- 自审 + 对抗复审无未处置的重要发现。

## 不做

- 不替换/恢复 PIDNet 模型；不新增 feature 名称；不改变端口语义；
  不做真机能力；不做 USD 场景优化。

## 对抗复核补充（fresh-context，实现后）

对三个改动的 Rust 文件做了只读对抗复核,发现并修复了会让新 CI
(`clippy --all-targets -D warnings` / `cargo test`) 变红的问题:

1. `slow_brain` 测试 `decode_rejects_length_mismatch` 会在构造阶段 panic
   (`StructArray::from` 强制子列等长),改为可达的 `decode_rejects_missing_descriptor`。
2. `build.rs` 的 `&PathBuf` 触发 `clippy::ptr_arg`,改为 `&Path`。
3. `slow_brain` 启动横幅 `print!("...\n")` 触发 `clippy::print_literal`,改 `println!`。
4. `slow_brain` 的 `odom_yaw` 字段只写不读,触发 `dead_code`,删除声明/初始化/写入,
   并把 odometry 守卫从 `>= 3` 收敛为 `>= 2`。
5. `fast_brain` 的 `let control_result = loop{..}; control_result` 有 `let_and_return`
   风险,改为闭包返回类型标注 `async move -> eyre::Result<()>` 并让带标签 loop 作尾表达式。

复核确认无问题的点:`&mut JoinHandle: Future`(Unpin)、带标签 loop 类型推断、
`break` 目标(宏调用点均在 `for k` 之外)、不导入 `Array` 时 `.len()/.value()/.values()/.value_length()`
可解析(与 `core-control/src/lib.rs` 既有先例一致)、`FixedSizeListArray::new`/`StructArray::from`
签名匹配 arrow 54.2、`struct_column_f32` 生命周期必需且 `needless_lifetimes` 不触发。

## 残留风险（本机不可验证,交 CI/后续）

- 本机 Windows 工具链缺 `llvm-config`/`libclang`/`dlltool`,`core-*` 无法本地编译,
  Rust 改动仅经静态复核 + 对抗复核,最终以 Linux CI 的 `clippy/test` 为准。
- `slow_brain` 中 `state.nav_route[state.current_target_index]` 越界为既有风险,
  非本切片引入,未处理。
- `decode_xfeat_struct` 的列/descriptor 长度校验对合法 `StructArray` 不可达
  (StructArray 不变式保证等长),保留为防御性代码。
- `verify_nexus.rs:14` 的 `print!` 同类 `print_literal` 属 core-control(CI 仅 `check`,
  不跑 clippy),未纳入本轮改动。

## 扩大复核（core-perception / core-decision / core-control 继承改动）

对继承的中文→英文重命名 crate 做了 fresh-context 对抗复核,并交叉核对 ort/opencv/dora
官方文档。确定修复:

1. **`cargo fmt --all`**:继承重命名把多行撑破 `max_width=100` 且 use tree 未排序,
   `format` CI job 必红;rustfmt 权威格式化后 `cargo fmt --all -- --check` 干净。
2. **`xfeat_engine.rs` `needless_lifetimes`**:`named_tensor<'output,'session>` 的
   `'session` 只出现一次 → 改 `SessionOutputs<'_>`(core-perception 跑 `clippy -D warnings`)。
3. **`xfeat_engine.rs` 真实模型测试**:原 `assert!(model.exists())` + `new().unwrap()`
   在 CPU-only CI 上因 `with_execution_providers([CUDA, CPU])` 注册 CUDA 失败而 panic;
   改为 session 初始化失败时优雅 skip(对齐 pidnet 策略),不改生产 EP 语义。
4. **自纠**:实现期一度写 `tokio::spawn(async move -> eyre::Result<()> {...})`,
   异步块不支持返回类型标注(rustfmt 解析报错暴露),回退为
   `let control_result: eyre::Result<()> = 'control_loop: loop {...}; control_result`
   并加 `#[allow(clippy::let_and_return)]`。

复核确认为**假阳性/噪声**(不改):`sim_echo_node.rs` 的 `send_output_bytes` 经 Dora 官方
文档核实为合法方法;`Event::Stop(_)` 元组模式与三 crate 既有写法一致。

 **残留外部 API 假设(交 Linux CI 编译仲裁,本机无法确定,盲改有破坏可工作代码风险)**:
 - ort `try_extract_tensor` 返回 `(&Shape, &[T])`:`matcher.rs` 用 `shape.num_elements()`、
   `xfeat/pidnet` 标注 `&[i64]` 并 `shape[i]/shape.len()`——仅当 `Shape: Deref<Target=[i64]>`
   且固有 `num_elements()` 时三者同时成立;`matcher.rs` 的 `outputs.values()` 同理需
   `SessionOutputs` 提供该方法。
 - `matcher.rs` 的 `find_fundamental_mat`/`find_homography_ext`/`mask.at::<u8>` opencv-rust 签名。
 - `perception_node.rs` 的 `metadata.timestamp()` dora 字段/方法名。
 - 少量中文错误文案残留(pidnet/ipm),非阻断,属作者面向操作者的语言选择。

## 外部 API 疑虑澄清(定论,已清除)

结合"原仓库可在实机(Isaac+acados+CUDA 宿主)编译运行"这一事实 + ort 官方源码 + git diff,
上述"残留外部 API 假设"经核查全部为**误报**,无需改动:

- **ort `Shape`**:官方源码 `src/value/impl_tensor/shape.rs` 显示 `Shape` 包装
  `SmallVec<[i64;4]>` 且提供固有 `num_elements()`;`try_extract_tensor` 返回 `(&Shape, &[T])`。
  故 `matcher.rs` 的 `shape.num_elements()` 与 `xfeat/pidnet` 的 `&[i64]` 标注/索引经
  `Shape: Deref<Target=[i64]>` 强制转换**同时成立**,不存在"必有一处编译失败"。
- **git diff 证据**:`try_extract_tensor`、`num_elements`、`find_fundamental_mat`、
  `find_homography_ext`、`outputs.values()` 均为**未改动的上下文行**(无 `+/-`),即 committed
  原样;committed 代码在实机可编译运行 → 这些 opencv-rust / dora / ort 调用签名本就正确。
  `metadata.timestamp()` 未出现在 diff 中 → 同样未改动、实机已验证。
- **`send_output_bytes`**:Dora 官方文档列为合法方法(`send_output_bytes(id, params, len, &[u8])`)。

净结论:继承的重命名只改标识符、未触碰外部 API 调用面;实机可运行 + 文档核对共同证明
core-perception 的外部 API 使用正确。真正的编译验证仍应在 Linux CI 跑一次以坐实,但本机
可确定的范围内已无未决阻断项。
