`timescale 1ns / 1ps


module lstmtop #(
    parameter DATA_WIDTH  = 16,
    parameter NUM_NEURONS = 16,
    parameter NUM_INPUTS  = 4      
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        sensor_valid,

    // Four sensor inputs (after preprocessing)
    input  wire signed [DATA_WIDTH-1:0] v_in,
    input  wire signed [DATA_WIDTH-1:0] i_in,
    input  wire signed [DATA_WIDTH-1:0] t_in,
    input  wire signed [DATA_WIDTH-1:0] ah_in,   

    output wire signed [DATA_WIDTH-1:0] soc_out,
    output wire                         soc_valid
);

    // Pack 4 inputs into wide bus for macarray
    // Packing order must match macarray input indexing:
    wire [NUM_INPUTS*DATA_WIDTH-1:0] lstm_x_in;
    assign lstm_x_in = {ah_in, t_in, i_in, v_in};

    // LSTM layer instance
    wire [NUM_NEURONS*DATA_WIDTH-1:0] lstm_h_out;
    wire                              lstm_done;

    lstmlayer #(
        .DATA_WIDTH (DATA_WIDTH),
        .NUM_NEURONS(NUM_NEURONS),
        .NUM_INPUTS (NUM_INPUTS)
    ) u_lstm (
        .clk        (clk),
        .rst        (rst),
        .start_calc (sensor_valid),
        .x_in       (lstm_x_in),
        .h_out      (lstm_h_out),
        .calc_done  (lstm_done)
    );

    // Dense layer instance
    denselayer #(
        .DATA_WIDTH (DATA_WIDTH),
        .NUM_INPUTS (NUM_NEURONS),
        .NUM_OUTPUTS(1)
    ) dense_final (
        .clk       (clk),
        .rst       (rst),
        .start_calc(lstm_done),
        .x_in      (lstm_h_out),
        .y_out     (soc_out),
        .calc_done (soc_valid)
    );

endmodule
