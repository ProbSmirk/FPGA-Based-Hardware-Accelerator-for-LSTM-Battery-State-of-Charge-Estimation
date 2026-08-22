`timescale 1ns / 1ps

module spicontrol #(
    parameter CLK_DIV = 50,       //100MHz/50 to generate a 2MHz spi clock
    parameter DATA_WIDTH = 16    
)(
    input wire clk,               
    input wire rst,               
    // SPI Physical Wires to ADC 
    input wire sdin,              
    output reg cs_n,             
    output reg sclk,             
    output reg [DATA_WIDTH-1:0] v_out,
    output reg [DATA_WIDTH-1:0] i_out,
    output reg [DATA_WIDTH-1:0] t_out,
    output reg sensor_valid    //lstm starts
);
localparam IDLE= 3'd0;
localparam READ_VOLTAGE=3'd1;
localparam READ_CURRENT=3'd2;
localparam READ_TEMP=3'd3;
localparam DONE=3'd4;
reg [2:0] state;
    reg [7:0] clk_counter;
    reg [4:0] bit_counter;
    reg [DATA_WIDTH-1:0] shift_reg;
    reg sclk_en; 
//internal counters and registers
//generating spi cloc
  always @(posedge clk or posedge rst) 
  begin
        if (rst) 
        begin
            clk_counter<=0;
            sclk<=0;
        end 
        else if (sclk_en) 
        begin
            if (clk_counter==(CLK_DIV/2)-1) 
            begin
                sclk<=~sclk; 
                clk_counter<=0;
            end 
            else 
            begin
                clk_counter<=clk_counter+1;
            end
        end else begin
            sclk<=0;
            clk_counter<=0;
        end
    end
    //detecting positive edge
    reg sclk_prev;
    always @(posedge clk) 
    sclk_prev <= sclk;
    wire sclk_rising_edge=(sclk==1'b1 && sclk_prev==1'b0);


    always @(posedge clk or posedge rst) 
    begin
        if (rst) 
        begin
            state<=IDLE;
            cs_n<=1'b1;
            sclk_en<=1'b0;
            bit_counter<=DATA_WIDTH;
            shift_reg<=0;
            v_out<=0; 
            i_out<=0; 
            t_out<=0;
            sensor_valid <= 1'b0;
        end 
        else
        begin
            sensor_valid <= 1'b0;
            case (state)
                IDLE: begin
                    cs_n<=1'b1;
                    sclk_en<=1'b0;
                    bit_counter<=DATA_WIDTH;
                    state<=READ_VOLTAGE;//can introduce delay here or feed continous data
                end
                
                READ_VOLTAGE: begin
                    cs_n<=1'b0;      //ADC on
                    sclk_en<=1'b1;  
                    if (sclk_rising_edge) 
                    begin
                        shift_reg<={shift_reg[DATA_WIDTH-2:0], sdin};
                        bit_counter<=bit_counter-1;
                        if (bit_counter==1) 
                        begin
                            v_out<={shift_reg[DATA_WIDTH-2:0], sdin};//read voltage
                            bit_counter<=DATA_WIDTH;                  //reset
                            state<=READ_CURRENT;
                        end
                    end
                end
                
                READ_CURRENT: begin
                    if (sclk_rising_edge) 
                    begin
                        shift_reg<={shift_reg[DATA_WIDTH-2:0], sdin};
                        bit_counter<=bit_counter-1;
                        
                        if (bit_counter==1) 
                        begin
                            i_out<={shift_reg[DATA_WIDTH-2:0], sdin}; //read current
                            bit_counter<=DATA_WIDTH;
                            state<=READ_TEMP;
                        end
                    end
                end
                
                READ_TEMP: begin
                    if (sclk_rising_edge) 
                    begin
                        shift_reg<={shift_reg[DATA_WIDTH-2:0], sdin};
                        bit_counter<=bit_counter - 1;
                        
                        if (bit_counter==1) 
                        begin
                            t_out<={shift_reg[DATA_WIDTH-2:0], sdin}; //read temperature
                            state<=DONE;
                        end
                    end
                end

                DONE: begin
                    sclk_en<=1'b0;        //stop spi clock
                    cs_n<=1'b1;           
                    sensor_valid<=1'b1;   //goes to lstm 
                    state<=IDLE;
                end
                
                default: 
                state<=IDLE;
            endcase
        end
    end


endmodule