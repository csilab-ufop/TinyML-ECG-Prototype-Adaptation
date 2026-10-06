# Generated ModusToolbox project

The generated `build_workspace/` is intentionally absent from this source package. `make -C firmware/psoc6 bootstrap` creates a CY8CPROTO-063-BLE dual-core project from Infineon's `mtb-example-psoc6-dual-cpu-empty-app` and attaches the published replay application to the CM4. The published `cm4_main.c` and `bootstrap.py` record the integration changes that were present in the measured Release build.

Use `make -C firmware/psoc6 getlibs` to fetch the BSP/middleware, `make -C firmware/psoc6 build` to compile, and `make -C firmware/psoc6 program` to program a connected board. A successful `make -C firmware/psoc6 check` verifies that the generated CM4 project still references the published sources. See [`../README.md`](../README.md) for the measurement scope and UART capture command.
