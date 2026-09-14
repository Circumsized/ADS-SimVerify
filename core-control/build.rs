// core-control/build.rs
//
// 该脚本负责把 Phase 5 验证主链所依赖的 C 动态库注入到 Rust 可执行文件:
//   * acados 主库 (`libacados.so`)                    -> 通用 acados 运行时;
//   * NMPC 自动生成的求解器 (`libacados_ocp_solver_diff_drive_car.so`)
//                                                    -> 当前 20 Hz 差速车 NMPC。
//
// 行为契约:
//   * 路径完全来自 `CARGO_MANIFEST_DIR` 或环境变量,
//     禁止硬编码任何开发机专属目录;
//   * acados / solver 目录不存在时仅打印 warning 而不注入链接指令,
//     因此 `cargo check` 与不含 wheel 能力的构建仍能完成;
//   * 检测到 C 求解器源文件变动时,自动触发 Rust 重新链接;
//   * 仅在 GNU/ELF 目标上注入 `-rpath` 与 `--disable-new-dtags`,
//     避免 RUNPATH 切断 `libacados.so -> libqpOASES_e.so` 的次级依赖,
//     同时不污染 MSVC 等非 GNU 目标的链接参数。
use std::env;
use std::path::{Path, PathBuf};

const ACADOS_LIB_NAME: &str = "acados";
const SOLVER_LIB_NAME: &str = "acados_ocp_solver_diff_drive_car";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect(
        "CARGO_MANIFEST_DIR is set by Cargo for build scripts; missing is a toolchain bug.",
    ));
    let workspace_root = manifest_dir
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.clone());

    // GNU 专属链接参数只在 ELF/glibc 目标注入,避免 MSVC 等目标失败。
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let is_gnu = target_env == "gnu";

    let acados_lib_dir = if let Ok(source) = env::var("ACADOS_SOURCE_DIR") {
        PathBuf::from(source).join("lib")
    } else {
        workspace_root.join("simulation-env/acados/lib")
    };

    let solver_lib_dir = if let Ok(generated) = env::var("ACADOS_GENERATED_CODE_DIR") {
        PathBuf::from(generated)
    } else {
        workspace_root.join("simulation-env/c_generated_code")
    };

    if inject_link_search_path(&acados_lib_dir, "acados", is_gnu) {
        println!("cargo:rustc-link-lib=dylib={ACADOS_LIB_NAME}");
    }
    if inject_link_search_path(&solver_lib_dir, "generated acados OCP solver", is_gnu) {
        println!("cargo:rustc-link-lib=dylib={SOLVER_LIB_NAME}");
    }

    let c_solver_source = solver_lib_dir.join("acados_solver_diff_drive_car.c");
    if c_solver_source.exists() {
        println!("cargo:rerun-if-changed={}", c_solver_source.display());
    }
    // 任何环境变量改动都要求重新评估链接路径,避免不同主机误用旧缓存。
    // Cargo 的 rerun-if-env-changed 只接受变量名,不接受 `key=value`。
    for key in [
        "ACADOS_SOURCE_DIR",
        "ACADOS_GENERATED_CODE_DIR",
        "ISAAC_SIM_PYTHON_SH",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
}

/// 注入库搜索路径与 GNU 专属链接参数。返回 `true` 表示目录存在、
/// 调用方应继续注入对应的 `rustc-link-lib`;目录缺失时仅告警并返回 `false`。
fn inject_link_search_path(directory: &Path, label: &str, is_gnu: bool) -> bool {
    if !directory.exists() {
        println!(
            "cargo:warning=[build.rs] {label} library directory not found: {}. \
             Run the Phase 5 solver generator under simulation-env/ before \
             expecting a wheel-capable binary.",
            directory.display()
        );
        return false;
    }

    println!("cargo:rustc-link-search=native={}", directory.display());
    if is_gnu {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", directory.display());
        // 关闭 new-dtags, 保留经典 RPATH 的依赖传递性, 否则
        // libacados.so 找不到次级依赖 libqpOASES_e.so 等。
        println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    }
    true
}
