# FPGA-Based Hardware Accelerator for LSTM Battery State-of-Charge Estimation

## Overview
This repository contains the complete RTL design and validation environment for a highly optimized design of a 16-neuron Long Short Term Memory (LSTM) neural network hardware accelerator. Designed specifically for the Xilinx Zynq-7020 FPGA, this accelerator predicts battery State of Charge (SoC) in real-time.

By translating a Python/Keras pipeline into a datapath using Q8.8 fixed point arithmetic, this design eliminates the inference and power latencies and complexities of software equivalents or floating point IPs.

The pipeline takes 4 sensory inputs - load voltage, load currents, battery temperature and coulomb counted Ah used. 
`ADC (SPI) -> coulomb_counter -> minmax_preprocess -> bnorm_preprocess -> lstmtop (lstmlayer + dense layer, macarray) -> soc_out`

### Architectural Data Flow
The hardware architecture is designed as a sequential datapath that operates entirely on Q8.8 fixed-point arithmetic. The process begins at the periphery, where an external ADC captures real-time voltage, current, and temperature, interfacing with the FPGA via a custom SPI controller (`spicontrol.v`), while a hardware coulomb counter tracks the Ampere-hours (Ah) used. These four raw inputs are fed into the `minmax_preprocess` and `bnorm_preprocess` modules, which scale and normalize the signals into proper Q8.8 format. The normalized inputs then enter the core inference engine, `lstmtop`. To conserve area, the internal `macarray.v` functions as a time-multiplexed matrix engine driven by an 84-cycle Finite State Machine. It reuses 16 dedicated DSP slices across multiple clock cycles to sequentially compute the Forget, Input, Cell Candidate, and Output gates, fetching the required parameters from parallel BRAMs. These gate values are constrained by piecewise-linear `hard_sigmoid` and `hard_tanh` logic before the `lstmlayer.v` updates the internal cell memory and outputs 16 hidden states. Finally, these hidden states flow into a combinational `denselayer.v`, which executes a single-cycle, 16-to-1 parallel MAC reduction—utilizing rigid bit-shifting to maintain decimal alignment—to output the final SoC percentage.

## FPGA Components Used

**1. DSP48E1 slices:** 
*   **Used where:** `macarray.v` (gate MACs), `denselayer.v` (final MAC)
*   **Purpose:** 16-bit × 32-bit signed multiply-accumulate for every LSTM gate and the dense output. 121 slices, 55% utilization.

**2. Block RAM (BRAM)**
*   **Used where:** `W_memory`/`U_memory`/`b_memory` in `macarray.v`; `W_memory`/`b_memory` in `denselayer.v`
*   **Purpose:** Stores Q8.8 fixed-point weights/biases, loaded via `$readmemh`, declared `(* ram_style = "block" *)`.

**3. Look-Up Tables (fabric LUTs)**
*   **Used where:** `hard_sigmoid`/`hard_tanh` functions, all FSM control logic, `minmax_preprocess.v`/`bnorm_preprocess.v` scaling logic, `spicontrol.v`
*   **Purpose:** General combinational logic — 6.7% utilization.

**4. BUFG (global clock buffer)**
*   **Used where:** Clock divider output in `topmodule.v`
*   **Purpose:** Low-skew distribution of the divided 25MHz slow_clk across the design.

**5. SPI / GPIO pins**
*   **Used where:** `spicontrol.v` (`sdin`/`cs_n`/`sclk`), reset button, status LEDs
*   **Purpose:** External ADC interface and board I/O.

## Phases Left
1. VIO Testing
2. Physical implementation on the ZedBoard via external ADC inputs

## Hardware Results

*   **Target device:** Xilinx Zynq-7020 (ZedBoard)
*   **DSP utilization:** 121 slices (55%)
*   **LUT utilization:** 6.7%
*   **Timing closure:** +7.84 ns WNS
*   **Power:** 0.133 W
*   **Arithmetic format:** Q8.8 fixed-point, piecewise-linear (`hard_sigmoid`/`hard_tanh`) activations
*   **RTL vs. Python golden reference:** Bit-exact (0 LSB) across 1050 test vectors (`tb_lstm_system.v`)
*   **Fixed-point pipeline vs. Keras float32:** ~4.87% mean SoC deviation
