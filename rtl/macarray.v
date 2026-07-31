`timescale 1ns / 1ps


module macarray #(
    parameter DATA_WIDTH  = 16,
    parameter NUM_NEURONS = 16,
    parameter NUM_INPUTS  = 4,
    parameter ACC_WIDTH   = 48
)(
    input  wire                              clk,
    input  wire                              rst,
    input  wire                              enable,
    input  wire [2:0]                        state,
    input  wire [NUM_INPUTS*DATA_WIDTH-1:0]  x_in,
    input  wire [NUM_NEURONS*DATA_WIDTH-1:0] h_in,
    output reg                               mac_done,
    output reg  [NUM_NEURONS*DATA_WIDTH-1:0] f_out,
    output reg  [NUM_NEURONS*DATA_WIDTH-1:0] i_out,
    output reg  [NUM_NEURONS*DATA_WIDTH-1:0] c_out,
    output reg  [NUM_NEURONS*DATA_WIDTH-1:0] o_out
);

    (* ram_style = "block" *) reg signed [31:0] W_memory [0:255];
    (* ram_style = "block" *) reg signed [31:0] U_memory [0:1023];
    (* ram_style = "block" *) reg signed [31:0] b_memory [0:63];

    initial begin
        $readmemh("lstm_W.txt", W_memory);
        $readmemh("lstm_U.txt", U_memory);
        $readmemh("lstm_b.txt", b_memory);
    end

    reg [1:0] gate_idx;      // Only 2 bits needed to store states 0, 1, 2, 3
    wire [5:0] gate_base;    // Dynamically reconstruct the 6-bit value
    assign gate_base = {gate_idx, 4'b0000}; // Append 4 zeros (multiplies by 16)

    localparam STEP_BIAS   = 5'd0;
    localparam STEP_INP0   = 5'd1;
    localparam STEP_INP1   = 5'd2;
    localparam STEP_INP2   = 5'd3;
    localparam STEP_INP3   = 5'd4;
    localparam STEP_REC0   = 5'd5;
    localparam STEP_ACT    = 5'd21;

    reg [4:0]  mac_step;
    reg        mac_running;
    reg [2:0]  state_reg;   // capture state at enable - stable during computation

    // Accumulators for all 16 neurons in parallel
    (* use_dsp = "yes" *)
    reg signed [ACC_WIDTH-1:0] acc [0:NUM_NEURONS-1];

    // act_in: bits [31:16] of accumulator - Q8.8 result
    wire signed [DATA_WIDTH-1:0] act_in [0:NUM_NEURONS-1];

    genvar gi;
    generate
        for (gi = 0; gi < NUM_NEURONS; gi = gi + 1) begin : act_slice
            assign act_in[gi] = acc[gi][31:16];
        end
    endgenerate

    function automatic signed [DATA_WIDTH-1:0] hard_sigmoid;
        input signed [DATA_WIDTH-1:0] x;
        begin
            if      ($signed(x) >  16'sh0200) hard_sigmoid =  16'sh0100;
            else if ($signed(x) < -16'sh0200) hard_sigmoid =  16'sh0000;
            else                              hard_sigmoid = ($signed(x) >>> 2) + 16'sh0080;
        end
    endfunction

    function automatic signed [DATA_WIDTH-1:0] hard_tanh;
        input signed [DATA_WIDTH-1:0] x;
        begin
            if      ($signed(x) >  16'sh0100) hard_tanh =  16'sh0100;
            else if ($signed(x) < -16'sh0100) hard_tanh = -16'sh0100;
            else                              hard_tanh = x;
        end
    endfunction

    integer n;

    always @(posedge clk) begin
        if (rst) begin
            mac_done    <= 1'b0;
            mac_running <= 1'b0;
            mac_step    <= 0;
            state_reg   <= 0;
            gate_idx    <= 0;
            for (n = 0; n < NUM_NEURONS; n = n + 1)
                acc[n] <= 0;
        end else begin
            mac_done <= 1'b0;

            if (enable && !mac_running) begin
                // Capture state, start sequential MAC
                mac_running <= 1'b1;
                mac_step    <= STEP_BIAS;
                state_reg   <= state;
                
                // Set the correct 2-bit gate index to drive the DSP memory offsets safely
                case (state)
                    3'd1:    gate_idx <= 2'd1;  
                    3'd2:    gate_idx <= 2'd0;  
                    3'd3:    gate_idx <= 2'd2;  
                    3'd4:    gate_idx <= 2'd3;  
                    default: gate_idx <= 2'd0;
                endcase
            end

            if (mac_running) begin
                case (mac_step)
                    STEP_BIAS: begin
                        // Load bias for all neurons
                        for (n = 0; n < NUM_NEURONS; n = n + 1)
                            acc[n] <= $signed(b_memory[gate_base + n]) <<< 8;
                        mac_step <= STEP_INP0;
                    end

                    STEP_INP0: begin
                        for (n = 0; n < NUM_NEURONS; n = n + 1)
                            acc[n] <= acc[n] +
                                ($signed(x_in[15:0]) *
                                 $signed(W_memory[0*64 + gate_base + n]));
                        mac_step <= STEP_INP1;
                    end

                    STEP_INP1: begin
                        for (n = 0; n < NUM_NEURONS; n = n + 1)
                            acc[n] <= acc[n] +
                                ($signed(x_in[31:16]) *
                                 $signed(W_memory[1*64 + gate_base + n]));
                        mac_step <= STEP_INP2;
                    end

                    STEP_INP2: begin
                        for (n = 0; n < NUM_NEURONS; n = n + 1)
                            acc[n] <= acc[n] +
                                ($signed(x_in[47:32]) *
                                 $signed(W_memory[2*64 + gate_base + n]));
                        mac_step <= STEP_INP3;
                    end

                    STEP_INP3: begin
                        for (n = 0; n < NUM_NEURONS; n = n + 1)
                            acc[n] <= acc[n] +
                                ($signed(x_in[63:48]) *
                                 $signed(W_memory[3*64 + gate_base + n]));
                        mac_step <= STEP_REC0;
                    end

                    STEP_REC0,
                    5'd6, 5'd7, 5'd8, 5'd9, 5'd10,
                    5'd11, 5'd12, 5'd13, 5'd14, 5'd15,
                    5'd16, 5'd17, 5'd18, 5'd19, 5'd20: begin
                        begin : rec_block
                            reg [4:0] k_idx;
                            k_idx = mac_step - STEP_REC0;

                            for (n = 0; n < NUM_NEURONS; n = n + 1)
                                acc[n] <= acc[n] +
                                    ($signed(h_in[k_idx*DATA_WIDTH +: DATA_WIDTH]) *
                                     $signed(U_memory[k_idx*64 + gate_base + n]));
                        end
                        mac_step <= mac_step + 1;
                    end

                    STEP_ACT: begin
                        mac_done    <= 1'b1;
                        mac_running <= 1'b0;
                        mac_step    <= 0;

                        case (state_reg)
                            3'd1: begin  // FORGET_GATE-sigmoid
                                for (n = 0; n < NUM_NEURONS; n = n + 1)
                                    f_out[n*DATA_WIDTH +: DATA_WIDTH] <= hard_sigmoid(act_in[n]);
                            end
                            3'd2: begin  //INPUT_GATE-sigmoid
                                for (n = 0; n < NUM_NEURONS; n = n + 1)
                                    i_out[n*DATA_WIDTH +: DATA_WIDTH] <= hard_sigmoid(act_in[n]);
                            end
                            3'd3: begin  //CELL_CANDIDATE-tanh
                                for (n = 0; n < NUM_NEURONS; n = n + 1)
                                    c_out[n*DATA_WIDTH +: DATA_WIDTH] <= hard_tanh(act_in[n]);
                            end
                            3'd4: begin  //OUTPUT_GATE-sigmoid
                                for (n = 0; n < NUM_NEURONS; n = n + 1)
                                    o_out[n*DATA_WIDTH +: DATA_WIDTH] <= hard_sigmoid(act_in[n]);
                            end
                            default: ;
                        endcase
                    end

                    default: mac_step <= 0;
                endcase
            end
        end
    end
endmodule