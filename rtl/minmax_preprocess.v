`timescale 1ns / 1ps


module minmax_preprocess #(

    parameter signed [31:0] MM_SCALE_V  = 32'h00000000,  // replace with mm_scale[0]
    parameter signed [31:0] MM_SCALE_I  = 32'h00000000,  // replace with mm_scale[1]
    parameter signed [31:0] MM_SCALE_T  = 32'h00000000,  // replace with mm_scale[2]
    parameter signed [31:0] MM_SCALE_AH = 32'h00000000,  // replace with mm_scale[3]

    parameter signed [31:0] MM_OFFSET_V  = 32'h00000000, // replace with mm_offset[0]
    parameter signed [31:0] MM_OFFSET_I  = 32'h00000000, // replace with mm_offset[1]
    parameter signed [31:0] MM_OFFSET_T  = 32'h00000000, // replace with mm_offset[2]
    parameter signed [31:0] MM_OFFSET_AH = 32'h00000000, // replace with mm_offset[3]

    parameter DATA_WIDTH = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire valid_in,                           // pulse: new ADC sample ready

    // Raw physical-unit inputs in Q8.8
    input  wire signed [DATA_WIDTH-1:0] v_raw,      // voltage (V)
    input  wire signed [DATA_WIDTH-1:0] i_raw,      // current (A)
    input  wire signed [DATA_WIDTH-1:0] t_raw,      // temperature (C)
    input  wire signed [DATA_WIDTH-1:0] ah_raw,     // Ah_used (Ah)

    // Normalised outputs in Q8.8 [0, 1]
    output reg  signed [DATA_WIDTH-1:0] v_norm,
    output reg  signed [DATA_WIDTH-1:0] i_norm,
    output reg  signed [DATA_WIDTH-1:0] t_norm,
    output reg  signed [DATA_WIDTH-1:0] ah_norm,
    output reg  valid_out
);


    function automatic signed [15:0] minmax_apply;
        input signed [15:0] x;
        input signed [31:0] scale;
        input signed [31:0] offset;
        reg signed [47:0] product;
        reg signed [47:0] result;
        reg signed [15:0] out;
        begin
            // x(Q8.8) * scale(Q16.16) = Q24.24 in 48-bit
            product = $signed({{32{x[15]}}, x}) * $signed(scale);
            result = (product >>> 16) + $signed(offset >>> 8);
            // result[15:0] is now Q8.8
            out = result[15:0];
            // Clamp to [0, 1] in Q8.8
            if ($signed(out) < 16'sh0000)
                minmax_apply = 16'sh0000;
            else if ($signed(out) > 16'sh0100)
                minmax_apply = 16'sh0100;
            else
                minmax_apply = out;
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            v_norm    <= 0;
            i_norm    <= 0;
            t_norm    <= 0;
            ah_norm   <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                v_norm  <= minmax_apply(v_raw,  MM_SCALE_V,  MM_OFFSET_V);
                i_norm  <= minmax_apply(i_raw,  MM_SCALE_I,  MM_OFFSET_I);
                t_norm  <= minmax_apply(t_raw,  MM_SCALE_T,  MM_OFFSET_T);
                ah_norm <= minmax_apply(ah_raw, MM_SCALE_AH, MM_OFFSET_AH);
            end
        end
    end

endmodule
