use std::env;
use std::fs;
use std::path::PathBuf;


fn rust_variant(name: &str) -> String {
    name.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first, chars.as_str().to_ascii_lowercase()),
                None => String::new(),
            }
        })
        .collect()
}

fn main() {
    println!("cargo:rerun-if-changed=cuda_abi.def");
    let spec = fs::read_to_string("cuda_abi.def").expect("read cuda_abi.def");
    let mut section = "";
    let mut abi_version = None::<u32>;
    let mut config = Vec::<(String, String)>::new();
    let mut persistent = Vec::<String>::new();
    let mut delta = Vec::<String>::new();

    for raw in spec.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        if let Some(value) = line.strip_prefix("abi_version ") {
            assert!(section.is_empty(), "abi_version must precede ABI sections");
            assert!(abi_version.is_none(), "duplicate abi_version");
            abi_version = Some(value.parse().expect("numeric abi_version"));
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = &line[1..line.len()-1];
            continue;
        }
        match section {
            "config" => {
                let mut parts = line.split_whitespace();
                let ty = parts.next().expect("config type");
                let name = parts.next().expect("config name");
                assert!(parts.next().is_none(), "invalid config ABI line: {line}");
                assert!(matches!(ty, "u32" | "f32"), "unsupported ABI type: {ty}");
                config.push((ty.to_owned(), name.to_owned()));
            }
            "persistent" => persistent.push(line.to_owned()),
            "delta" => delta.push(line.to_owned()),
            _ => panic!("ABI entry outside a section: {line}"),
        }
    }
    let abi_version = abi_version.expect("cuda_abi.def must declare abi_version");
    assert!(!config.is_empty());
    assert!(!persistent.is_empty());
    assert!(!delta.is_empty());

    let mut rust = String::new();
    rust.push_str(&format!("const CUDA_ABI_GENERATED_VERSION: u32 = {abi_version};\n\n"));
    rust.push_str("#[repr(C)]\n#[derive(Clone, Copy)]\nstruct CudaConfig {\n");
    for (ty, name) in &config { rust.push_str(&format!("    {name}: {ty},\n")); }
    rust.push_str("}\n\n");
    rust.push_str("#[repr(usize)]\n#[derive(Clone, Copy, Debug, PartialEq, Eq)]\nenum PersistentPointer {\n");
    for (i,name) in persistent.iter().enumerate() { rust.push_str(&format!("    {} = {i},\n", rust_variant(name))); }
    rust.push_str("}\nimpl PersistentPointer { const fn index(self) -> usize { self as usize } }\n");
    rust.push_str(&format!("const PERSISTENT_POINTER_COUNT: usize = {};\n\n", persistent.len()));
    rust.push_str("#[repr(usize)]\n#[derive(Clone, Copy, Debug, PartialEq, Eq)]\nenum BatchDeltaPointer {\n");
    for (i,name) in delta.iter().enumerate() { rust.push_str(&format!("    {} = {i},\n", rust_variant(name))); }
    rust.push_str("}\nimpl BatchDeltaPointer { const fn index(self) -> usize { self as usize } }\n");
    rust.push_str(&format!("const BATCH_DELTA_POINTER_COUNT: usize = {};\n", delta.len()));

    let mut header = String::new();
    header.push_str(&format!("#define LEO_CUDA_ABI_VERSION {abi_version}\n"));
    header.push_str("struct LeoConfig {\n");
    for (ty,name) in &config {
        let c_ty = if ty == "u32" { "unsigned int" } else { "float" };
        header.push_str(&format!("    {c_ty} {name};\n"));
    }
    header.push_str("};\n");
    header.push_str("enum LeoPersistentPointerIndex {\n");
    for (i,name) in persistent.iter().enumerate() { header.push_str(&format!("    LEO_P_{name} = {i},\n")); }
    header.push_str(&format!("    LEO_P_POINTER_COUNT = {}\n}};\n", persistent.len()));
    header.push_str("enum LeoBatchDeltaPointerIndex {\n");
    for (i,name) in delta.iter().enumerate() { header.push_str(&format!("    LEO_D_{name} = {i},\n")); }
    header.push_str(&format!("    LEO_D_POINTER_COUNT = {}\n}};\n", delta.len()));
    header.push_str(&format!("static_assert(sizeof(LeoConfig) == {}, \"LeoConfig ABI size mismatch\");\n", config.len() * 4));

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    fs::write(out.join("cuda_abi_generated.rs"), rust).expect("write Rust CUDA ABI");
    fs::write(out.join("leo_cuda_abi.h"), header).expect("write CUDA ABI header");
}
