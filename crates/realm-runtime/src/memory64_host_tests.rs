//! Unit tests for host function registration

#[cfg(test)]
mod tests {
    use crate::memory64_host::{Memory64Runtime, MemoryLayout};
    use wasmtime::{Config, Engine, Linker, Module, Store};

    #[test]
    #[cfg_attr(target_os = "windows", ignore)] // Skip on Windows CI due to memory limits (8GB allocation)
    fn test_realm_get_model_metadata_registered() {
        // Test that realm_get_model_metadata is registered in the linker
        let mut config = Config::new();
        config.wasm_memory64(true);
        config.wasm_bulk_memory(true);
        let engine = Engine::new(&config).unwrap();
        let mut linker = Linker::new(&engine);

        let layout = MemoryLayout::single(8, "test_storage").unwrap();
        let runtime = Memory64Runtime::new(layout, true);

        // This should register all host functions including realm_get_model_metadata
        runtime.add_to_linker(&mut linker).unwrap();

        // Create a minimal WASM module that imports realm_get_model_metadata
        let wasm_bytes = wat::parse_str(
            r#"
            (module
                (import "env" "realm_get_model_metadata" (func $realm_get_model_metadata (param i32 i32 i32) (result i32)))
                (memory 1)
                (func (export "test") (result i32)
                    i32.const 0
                    i32.const 0
                    i32.const 4096
                    call $realm_get_model_metadata
                )
            )
            "#,
        )
        .unwrap();

        let module = Module::new(&engine, wasm_bytes).unwrap();
        let mut store = Store::new(&engine, ());
        runtime.initialize(&mut store).unwrap();

        // This should succeed if realm_get_model_metadata is registered
        let instance = linker.instantiate(&mut store, &module);
        assert!(
            instance.is_ok(),
            "Failed to instantiate WASM module with realm_get_model_metadata import: {:?}",
            instance.err()
        );
    }

    #[test]
    #[cfg_attr(target_os = "windows", ignore)] // Skip on Windows CI due to memory limits (8GB allocation)
    fn test_all_model_functions_registered() {
        // Test that all model-related host functions are registered
        let mut config = Config::new();
        config.wasm_memory64(true);
        config.wasm_bulk_memory(true);
        let engine = Engine::new(&config).unwrap();
        let mut linker = Linker::new(&engine);

        let layout = MemoryLayout::single(8, "test_storage").unwrap();
        let runtime = Memory64Runtime::new(layout, true);

        runtime.add_to_linker(&mut linker).unwrap();

        // List of expected host functions - create a WASM module that imports all of them
        let expected_functions = vec![
            ("realm_store_model", 3, 1),
            ("realm_get_model_info", 3, 1),
            ("realm_get_model_metadata", 3, 1),
            ("realm_remove_model", 1, 1),
        ];

        for (func_name, param_count, result_count) in expected_functions {
            let params: String = (0..param_count)
                .map(|_| "i32")
                .collect::<Vec<_>>()
                .join(" ");
            let results: String = (0..result_count)
                .map(|_| "i32")
                .collect::<Vec<_>>()
                .join(" ");
            let wasm_bytes = wat::parse_str(format!(
                r#"
                (module
                    (import "env" "{}" (func $f (param {}) (result {})))
                    (memory 1)
                    (func (export "test") (result i32) i32.const 0)
                )
                "#,
                func_name, params, results
            ))
            .unwrap();

            let module = Module::new(&engine, wasm_bytes).unwrap();
            let mut store = Store::new(&engine, ());
            runtime.initialize(&mut store).unwrap();

            let instance = linker.instantiate(&mut store, &module);
            assert!(
                instance.is_ok(),
                "Failed to instantiate WASM module with {} import: {:?}",
                func_name,
                instance.err()
            );
        }
    }
}
