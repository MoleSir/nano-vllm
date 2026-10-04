use std::path::{Path, PathBuf};
use std::process::Command;

/// Locate the CUDA root: prefer env vars, else derive from `nvcc` in PATH.
fn cuda_root() -> Option<PathBuf> {
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(v) = std::env::var(var)
            && !v.is_empty()
        {
            return Some(PathBuf::from(v));
        }
    }
    let out = Command::new("which").arg("nvcc").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let nvcc = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    nvcc.parent()?.parent().map(Path::to_path_buf)
}

/// CUDA arch flags. `CUDA_ARCH` (e.g. `90`) wins; otherwise detect all GPUs.
fn cuda_arch_flags() -> Vec<String> {
    if let Ok(a) = std::env::var("CUDA_ARCH") {
        let a = a.trim();
        if !a.is_empty() {
            return vec![
                format!("-gencode=arch=compute_{a},code=sm_{a}"),
                format!("-gencode=arch=compute_{a},code=compute_{a}"),
            ];
        }
    }

    let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }

    let mut seen = std::collections::BTreeSet::new();
    let mut flags = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let num: String = line.trim().chars().filter(|c| c.is_ascii_digit()).collect();
        if num.is_empty() || !seen.insert(num.clone()) {
            continue;
        }
        flags.push(format!("-gencode=arch=compute_{num},code=sm_{num}"));
        flags.push(format!("-gencode=arch=compute_{num},code=compute_{num}"));
    }
    flags
}

fn main() {
    let sources = ["src/kernels/store_kvcache.cu", "src/kernels/flash_attn.cu", "src/kernels/paged_attn.cu"];
    for src in sources {
        println!("cargo:rerun-if-changed={src}");
    }
    println!("cargo:rerun-if-changed=src/kernels/paged_attn_common.cuh");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let root = cuda_root();
    let nvcc = root
        .as_ref()
        .map(|r| r.join("bin").join("nvcc"))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("nvcc"));

    let arch = cuda_arch_flags();
    let mut objects = Vec::new();

    // 1. Compile every .cu to a .o with nvcc.
    for src in sources {
        let stem = Path::new(src).file_stem().unwrap().to_string_lossy().to_string();
        let obj = out_dir.join(format!("{stem}.o"));
        let mut args: Vec<String> = [
            "-c",
            src,
            "-o",
            obj.to_str().unwrap(),
            "-std=c++17",
            "-O3",
            "-Xcompiler",
            "-fPIC",
            "--use_fast_math",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        args.extend(arch.iter().cloned());

        let out = Command::new(&nvcc)
            .args(&args)
            .output()
            .expect("failed to run nvcc");
        assert!(
            out.status.success(),
            "nvcc failed for {src}:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        objects.push(obj);
    }

    // 2. Archive them into a static lib.
    let lib = out_dir.join("libnanovllm_kernels.a");
    let mut ar_args = vec!["rcs".to_string(), lib.to_str().unwrap().to_string()];
    ar_args.extend(objects.iter().map(|o| o.to_str().unwrap().to_string()));
    let out = Command::new("ar")
        .args(&ar_args)
        .output()
        .expect("failed to run ar");
    assert!(
        out.status.success(),
        "ar failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // 3. Link instructions.
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=nanovllm_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // 4. cudart library dir + embedded rpath.
    if let Some(root) = &root {
        for sub in ["lib64", "lib", "targets/x86_64-linux/lib"] {
            let dir = root.join(sub);
            if dir.is_dir() {
                println!("cargo:rustc-link-search=native={}", dir.display());
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
            }
        }
    }
}
