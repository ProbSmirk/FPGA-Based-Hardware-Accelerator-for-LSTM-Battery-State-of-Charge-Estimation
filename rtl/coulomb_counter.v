`timescale 1ns / 1ps


module coulomb_counter #(
    parameter DATA_WIDTH   = 16,
    // Sample rate of ADC in Hz
    // Must match SAMPLE_RATE_HZ used in prepare_dataset.py
    // Your data: time column shows ~10s between rows → 0.1Hz
    // But XADC runs faster — set to your actual ADC polling rate
    // For 1Hz: DT_NUMERATOR=1, DT_DENOMINATOR=3600
    // For 10Hz: DT_NUMERATOR=1, DT_DENOMINATOR=36000
    // For 0.1Hz (1 sample per 10 seconds): DT_NUMERATOR=10, DT_DENOMINATOR=3600
    parameter integer SAMPLE_PERIOD_S  = 10,   // seconds per sample (from your data: ~10s)
    parameter integer SECONDS_PER_HOUR = 3600
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        sample_valid,   // pulse: new current sample ready

    input  wire signed [DATA_WIDTH-1:0] i_in,         // current in Q8.8 (Amps)
    output reg  signed [DATA_WIDTH-1:0] ah_out         // Ah_used in Q8.8
);

    // dt/3600 in Q16.16:
    // dt = SAMPLE_PERIOD_S seconds
    // dt/3600 = SAMPLE_PERIOD_S/3600
    // In Q16.16: round(SAMPLE_PERIOD_S/3600 * 65536)
    // For 10s: 10/3600 * 65536 = 182.04 → 182
    localparam signed [31:0] DT_AH_Q16 =
        $rtoi($itor(SAMPLE_PERIOD_S) / $itor(SECONDS_PER_HOUR) * 65536.0 + 0.5);

    // Internal accumulator: Q16.16 gives enough range
    // Max Ah_used in training data = 1774 Ah
    // Q16.16 max = 32767.99 — covers up to 32767 Ah ✓
    reg signed [47:0] accumulator;

    // Previous current sample for trapezoidal rule
    reg signed [DATA_WIDTH-1:0] i_prev;

    always @(posedge clk) begin
        if (rst) begin
            accumulator <= 48'sd0;
            i_prev      <= 16'sd0;
            ah_out      <= 16'sd0;
        end else if (sample_valid) begin
            // Trapezoidal: dAh = (i_prev + i_in) / 2 * dt_ah
            // = (i_prev + i_in) * DT_AH_Q16 / 2
            // i is Q8.8, DT_AH_Q16 is Q16.16
            // product is Q24.24 in 48-bit
            // divide by 2 (arithmetic right shift 1)
            accumulator <= accumulator +
                           (($signed({{32{i_prev[DATA_WIDTH-1]}}, i_prev}) +
                             $signed({{32{i_in[DATA_WIDTH-1]}},   i_in}))
                            * $signed(DT_AH_Q16) >>> 9);
                            // >>>9 = >>>8 (Q8.8→Q16.16 correction) + >>>1 (÷2)

            i_prev <= i_in;

            // Output: take bits [31:16] of the 48-bit accumulator
            // accumulator is Q16.16, [31:16] gives the Q8.8 integer+fraction part
            ah_out <= accumulator[31:16];
        end
    end

endmodule
