`timescale 1ns / 1ps


module coulomb_counter #(
    parameter DATA_WIDTH   = 16
    parameter integer SAMPLE_PERIOD_S  = 10,   // seconds per sample (from data: ~10s)
    parameter integer SECONDS_PER_HOUR = 3600
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        sample_valid,   // pulse: new current sample ready

    input  wire signed [DATA_WIDTH-1:0] i_in,         // current in Q8.8 (Amps)
    output reg  signed [DATA_WIDTH-1:0] ah_out         // Ah_used in Q8.8
);

    // dt/3600 in Q16.16;
    localparam signed [31:0] DT_AH_Q16 =
        $rtoi($itor(SAMPLE_PERIOD_S) / $itor(SECONDS_PER_HOUR) * 65536.0 + 0.5);

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
                            

            i_prev <= i_in;

            // Output: take bits [31:16] of the 48-bit accumulator
            // accumulator is Q16.16, [31:16] gives the Q8.8 integer+fraction part
            ah_out <= accumulator[31:16];
        end
    end

endmodule
