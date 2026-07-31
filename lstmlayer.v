`timescale 1ns / 1ps

module lstmlayer #(
    parameter DATA_WIDTH=16,
    parameter NUM_NEURONS=16,
    parameter NUM_INPUTS=4       
)(
    input wire clk,
    input wire rst,
    input wire start_calc,             
    input wire [(NUM_INPUTS*DATA_WIDTH)-1:0] x_in, 
    output reg [(NUM_NEURONS*DATA_WIDTH)-1:0] h_out, 
    output reg calc_done   
);

    reg signed [DATA_WIDTH-1:0] cell_state   [0:NUM_NEURONS-1];
    reg signed [DATA_WIDTH-1:0] hidden_state [0:NUM_NEURONS-1];
        
    localparam IDLE           = 3'd0;
    localparam FORGET_GATE    = 3'd1; 
    localparam INPUT_GATE     = 3'd2; 
    localparam CELL_CANDIDATE = 3'd3; 
    localparam OUTPUT_GATE    = 3'd4; 
    localparam UPDATE_CELL    = 3'd5; 
    localparam UPDATE_H       = 3'd6; 
    localparam DONE           = 3'd7; 
    
    reg [2:0] state;
    reg  mac_enable;
    wire mac_done;
        
    wire [(NUM_NEURONS*DATA_WIDTH)-1:0] f_t_bus; 
    wire [(NUM_NEURONS*DATA_WIDTH)-1:0] i_t_bus; 
    wire [(NUM_NEURONS*DATA_WIDTH)-1:0] c_t_bus;
    wire [(NUM_NEURONS*DATA_WIDTH)-1:0] o_t_bus; 
    
    wire signed [DATA_WIDTH-1:0] f_t     [0:NUM_NEURONS-1];
    wire signed [DATA_WIDTH-1:0] i_t     [0:NUM_NEURONS-1];
    wire signed [DATA_WIDTH-1:0] c_tilde [0:NUM_NEURONS-1];
    wire signed [DATA_WIDTH-1:0] o_t     [0:NUM_NEURONS-1];
    
    genvar g;
    generate
        for(g=0; g<NUM_NEURONS; g=g+1) begin : split_buses
            assign f_t[g]     = f_t_bus[g*DATA_WIDTH +: DATA_WIDTH];
            assign i_t[g]     = i_t_bus[g*DATA_WIDTH +: DATA_WIDTH];
            assign c_tilde[g] = c_t_bus[g*DATA_WIDTH +: DATA_WIDTH];
            assign o_t[g]     = o_t_bus[g*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    wire signed [DATA_WIDTH-1:0] tanh_cell [0:NUM_NEURONS-1];
    genvar gi;
    generate
        for (gi = 0; gi < NUM_NEURONS; gi = gi + 1) begin : tanh_cell_gen
            assign tanh_cell[gi] =
                ($signed(cell_state[gi]) >  16'sh0100) ?  16'sh0100 :
                ($signed(cell_state[gi]) < -16'sh0100) ? -16'sh0100 : 
                 cell_state[gi];
        end
    endgenerate

    integer i;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state      <= IDLE;
            calc_done  <= 1'b0;
            mac_enable <= 1'b0;
            h_out      <= 0;
            for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                cell_state[i]   <= 0;
                hidden_state[i] <= 0;
            end
        end else begin
            calc_done  <= 1'b0;
            mac_enable <= 1'b0;
                         
            case (state)
                IDLE: begin
                    if (start_calc) begin
                        state      <= FORGET_GATE;
                        mac_enable <= 1'b1;
                    end
                end
                FORGET_GATE: begin
                    if (mac_done) begin 
                        state      <= INPUT_GATE;
                        mac_enable <= 1'b1;
                    end
                end
                INPUT_GATE: begin
                    if (mac_done) begin
                        state      <= CELL_CANDIDATE;
                        mac_enable <= 1'b1;
                    end
                end
                CELL_CANDIDATE: begin
                    if (mac_done) begin
                        state      <= OUTPUT_GATE;
                        mac_enable <= 1'b1;
                    end
                end
                OUTPUT_GATE: begin
                    if (mac_done) begin
                        state <= UPDATE_CELL;
                    end
                end
                UPDATE_CELL: begin
                    for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                        cell_state[i] <=
                            (($signed({{16{f_t[i][15]}},    f_t[i]})    * $signed({{16{cell_state[i][15]}}, cell_state[i]})) >>> 8) +
                            (($signed({{16{i_t[i][15]}},    i_t[i]})    * $signed({{16{c_tilde[i][15]}}, c_tilde[i]})) >>> 8);
                    end
                    state <= UPDATE_H;
                end
                UPDATE_H: begin
                    for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                        hidden_state[i] <=
                            (($signed({{16{o_t[i][15]}},      o_t[i]}))    * ($signed({{16{tanh_cell[i][15]}}, tanh_cell[i]}))) >>> 8;
                    end
                    state <= DONE;
                end
                DONE: begin
                    // Pack hidden state into output bus (LSB = neuron 0)
                    h_out <= {hidden_state[15], hidden_state[14],
                              hidden_state[13], hidden_state[12],
                              hidden_state[11], hidden_state[10],
                              hidden_state[9],  hidden_state[8],
                              hidden_state[7],  hidden_state[6],
                              hidden_state[5],  hidden_state[4],
                              hidden_state[3],  hidden_state[2],
                              hidden_state[1],  hidden_state[0]};
                    calc_done <= 1'b1; 
                    state     <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

    macarray #(
        .DATA_WIDTH (DATA_WIDTH),
        .NUM_NEURONS(NUM_NEURONS),
        .NUM_INPUTS (NUM_INPUTS)
    ) dsp_multipliers (
        .clk     (clk),
        .rst     (rst),
        .enable  (mac_enable),
        .state   (state),   
        .x_in    (x_in),         
        .h_in    (h_out),
        .mac_done(mac_done),
        .f_out   (f_t_bus),
        .i_out   (i_t_bus),
        .c_out   (c_t_bus),
        .o_out   (o_t_bus)
    );

endmodule