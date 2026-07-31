`timescale 1ns / 1ps


module bnorm_preprocess #(

    parameter signed [31:0] BN_SCALE_V  = 32'h00010000,  // replace with bn_scale[0]
    parameter signed [31:0] BN_SCALE_I  = 32'h00010000,  // replace with bn_scale[1]
    parameter signed [31:0] BN_SCALE_T  = 32'h00010000,  // replace with bn_scale[2]
    parameter signed [31:0] BN_SCALE_AH = 32'h00010000,  // replace with bn_scale[3]

    parameter signed [31:0] BN_OFFSET_V  = 32'h00000000, // replace with bn_offset[0]
    parameter signed [31:0] BN_OFFSET_I  = 32'h00000000, // replace with bn_offset[1]
    parameter signed [31:0] BN_OFFSET_T  = 32'h00000000, // replace with bn_offset[2]
    parameter signed [31:0] BN_OFFSET_AH = 32'h00000000, // replace with bn_offset[3]

    parameter DATA_WIDTH = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire valid_in,

    input  wire signed [DATA_WIDTH-1:0] v_in,
    input  wire signed [DATA_WIDTH-1:0] i_in,
    input  wire signed [DATA_WIDTH-1:0] t_in,
    input  wire signed [DATA_WIDTH-1:0] ah_in,

    output reg  signed [DATA_WIDTH-1:0] v_out,
    output reg  signed [DATA_WIDTH-1:0] i_out,
    output reg  signed [DATA_WIDTH-1:0] t_out,
    output reg  signed [DATA_WIDTH-1:0] ah_out,
    output reg  valid_out
);

    // Same arithmetic as minmax_preprocess but NO clamping
    // (BatchNorm output can be negative and > 1)
    function automatic signed [15:0] bn_apply;
        input signed [15:0] x;
        input signed [31:0] scale;
        input signed [31:0] offset;
        reg signed [47:0] product;
        reg signed [47:0] result;
        begin
            product  = $signed({{32{x[15]}}, x}) * $signed(scale);
            result   = (product >>> 16) + $signed(offset >>> 8);
            bn_apply = result[15:0];
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            v_out     <= 0;
            i_out     <= 0;
            t_out     <= 0;
            ah_out    <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                v_out  <= bn_apply(v_in,  BN_SCALE_V,  BN_OFFSET_V);
                i_out  <= bn_apply(i_in,  BN_SCALE_I,  BN_OFFSET_I);
                t_out  <= bn_apply(t_in,  BN_SCALE_T,  BN_OFFSET_T);
                ah_out <= bn_apply(ah_in, BN_SCALE_AH, BN_OFFSET_AH);
            end
        end
    end

endmodule
