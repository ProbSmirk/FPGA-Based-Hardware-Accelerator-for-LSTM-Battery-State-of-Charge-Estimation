`timescale 1ns / 1ps

module denselayer #(
    parameter DATA_WIDTH=16,
    parameter NUM_INPUTS=16,       
    parameter NUM_OUTPUTS=1        
)(
    input wire clk,
    input wire rst,
    input wire start_calc,           
    input wire [(NUM_INPUTS*DATA_WIDTH)-1:0] x_in, 
    output reg [DATA_WIDTH-1:0] y_out, //final SOC %
    output reg calc_done               
);

 (* ram_style = "block" *) reg signed [31:0] W_memory [0:15];
(* ram_style = "block" *) reg signed [31:0] b_memory [0:0];

    initial 
    begin
        $readmemh("dense_W.txt", W_memory);
        $readmemh("dense_b.txt", b_memory);
        
        // Let's prove the Dense memory is actually loading!
        $display("DENSE SANITY CHECK - First W_memory: %h, Bias: %h", W_memory[0], b_memory[0]);
    end

    reg [1:0] state;

   
    wire signed [47:0] next_accumulator;
    
    assign next_accumulator = 
               ($signed(x_in[15:0])   * $signed(W_memory[0]))  +
               ($signed(x_in[31:16])  * $signed(W_memory[1]))  +
               ($signed(x_in[47:32])  * $signed(W_memory[2]))  +
               ($signed(x_in[63:48])  * $signed(W_memory[3]))  +
               ($signed(x_in[79:64])  * $signed(W_memory[4]))  +
               ($signed(x_in[95:80])  * $signed(W_memory[5]))  +
               ($signed(x_in[111:96]) * $signed(W_memory[6]))  +
               ($signed(x_in[127:112])* $signed(W_memory[7]))  +
               ($signed(x_in[143:128])* $signed(W_memory[8]))  +
               ($signed(x_in[159:144])* $signed(W_memory[9]))  +
               ($signed(x_in[175:160])* $signed(W_memory[10])) +
               ($signed(x_in[191:176])* $signed(W_memory[11])) +
               ($signed(x_in[207:192])* $signed(W_memory[12])) +
               ($signed(x_in[223:208])* $signed(W_memory[13])) +
               ($signed(x_in[239:224])* $signed(W_memory[14])) +
               ($signed(x_in[255:240])* $signed(W_memory[15])) +
               ($signed(b_memory[0]) <<< 8);

    wire signed [DATA_WIDTH-1:0] act_in;
    assign act_in = next_accumulator[31:16]; //middle q8.8 used

    // Final Sigmoid Logic
    wire signed [DATA_WIDTH-1:0] final_soc;
    wire signed [DATA_WIDTH-1:0] sig_approx;
    
    assign sig_approx = (act_in >>> 2) + $signed(16'h0080);
    assign final_soc = (act_in > $signed(16'h0200)) ? 16'h0100 :   
                       (act_in < $signed(-16'h0200)) ? 16'h0000 :  
                       sig_approx;  

    always @(posedge clk or posedge rst) 
    begin
        if (rst) 
        begin
            state <= 0;
            calc_done <= 0;
            y_out <= 0;
        end 
        else 
        begin
            calc_done <= 0; 
            
            case (state)
                0: begin //IDLE
                    if (start_calc) 
                    begin
                        $display("DENSE MATH CHECK: LSTM_Neuron_0=%h | act_in=%h | final_soc=%h", x_in[15:0], act_in, final_soc);
                        y_out <= final_soc; 
                        calc_done <= 1'b1;  
                        state     <= 1;       

                    end
                end
                 1: begin 
        state <= 0;  
        end
            endcase
        end
    end

endmodule