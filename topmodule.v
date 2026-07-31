

module topmodule #(
    parameter DATA_WIDTH = 16,

    // Replace 32'h00000000 with actual Q16.16 hex values printed by the script
    parameter signed [31:0] MM_SCALE_V   = 32'h00001CF2,
    parameter signed [31:0] MM_SCALE_I   = 32'h00003333,
    parameter signed [31:0] MM_SCALE_T   = 32'h00000124,
    parameter signed [31:0] MM_SCALE_AH  = 32'h00000025,
    parameter signed [31:0] MM_OFFSET_V  = 32'h000000C8,
    parameter signed [31:0] MM_OFFSET_I  = 32'h00000000,
    parameter signed [31:0] MM_OFFSET_T  = 32'h0000846F,
    parameter signed [31:0] MM_OFFSET_AH = 32'h00000000,

    parameter signed [31:0] BN_SCALE_V   = 32'h00073AA6,  // default = 1.0
    parameter signed [31:0] BN_SCALE_I   = 32'h0002D577,
    parameter signed [31:0] BN_SCALE_T   = 32'h00147F2C,
    parameter signed [31:0] BN_SCALE_AH  = 32'h0004D26D,
    parameter signed [31:0] BN_OFFSET_V  = 32'hFFFB5B83,  // default = 0.0
    parameter signed [31:0] BN_OFFSET_I  = 32'hFFFEFB22,
    parameter signed [31:0] BN_OFFSET_T  = 32'hFFF1F2A7,
    parameter signed [31:0] BN_OFFSET_AH = 32'hFFFEDB57,

    // Set to match your ADC polling rate
    parameter integer SAMPLE_PERIOD_S = 10
)(
    input  wire clk,
    input  wire rst,

    // SPI ADC interface (connect to ZedBoard JA/JB header)
    input  wire sdin,
    output wire cs_n,
    output wire sclk,

    // SoC output
    output wire [DATA_WIDTH-1:0] soc_out,
    output wire                  soc_valid
);

    //clock divider
    reg [1:0] div_cnt    = 0;
    reg       slow_clk_r = 0;
    wire      slow_clk;

    always @(posedge clk) begin
        div_cnt <= div_cnt + 1;
        if (div_cnt == 2'b11) begin   // toggle every 4 cycles
            slow_clk_r <= ~slow_clk_r;
            div_cnt    <= 0;
        end
    end

    BUFG clk_buf (.I(slow_clk_r), .O(slow_clk));

    //  SPI controller â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    wire [DATA_WIDTH-1:0] v_raw_adc, i_raw_adc, t_raw_adc;
    wire                  sensor_valid_spi;

    spicontrol #(
        .CLK_DIV   (50),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_spi (
        .clk         (slow_clk),
        .rst         (rst),
        .sdin        (sdin),
        .cs_n        (cs_n),
        .sclk        (sclk),
        .v_out       (v_raw_adc),
        .i_out       (i_raw_adc),
        .t_out       (t_raw_adc),
        .sensor_valid(sensor_valid_spi)
    );

    wire signed [DATA_WIDTH-1:0] ah_raw;

    coulomb_counter #(
        .DATA_WIDTH    (DATA_WIDTH),
        .SAMPLE_PERIOD_S(SAMPLE_PERIOD_S)
    ) u_cc (
        .clk         (slow_clk),
        .rst         (rst),
        .sample_valid(sensor_valid_spi),
        .i_in        ($signed(i_raw_adc)),
        .ah_out      (ah_raw)
    );

    wire signed [DATA_WIDTH-1:0] v_mm, i_mm, t_mm, ah_mm;
    wire                         mm_valid;

    minmax_preprocess #(
        .MM_SCALE_V  (MM_SCALE_V),
        .MM_SCALE_I  (MM_SCALE_I),
        .MM_SCALE_T  (MM_SCALE_T),
        .MM_SCALE_AH (MM_SCALE_AH),
        .MM_OFFSET_V (MM_OFFSET_V),
        .MM_OFFSET_I (MM_OFFSET_I),
        .MM_OFFSET_T (MM_OFFSET_T),
        .MM_OFFSET_AH(MM_OFFSET_AH),
        .DATA_WIDTH  (DATA_WIDTH)
    ) u_mm (
        .clk      (slow_clk),
        .rst      (rst),
        .valid_in (sensor_valid_spi),
        .v_raw    ($signed(v_raw_adc)),
        .i_raw    ($signed(i_raw_adc)),
        .t_raw    ($signed(t_raw_adc)),
        .ah_raw   (ah_raw),
        .v_norm   (v_mm),
        .i_norm   (i_mm),
        .t_norm   (t_mm),
        .ah_norm  (ah_mm),
        .valid_out(mm_valid)
    );

    wire signed [DATA_WIDTH-1:0] v_bn, i_bn, t_bn, ah_bn;
    wire                         bn_valid;

    bnorm_preprocess #(
        .BN_SCALE_V  (BN_SCALE_V),
        .BN_SCALE_I  (BN_SCALE_I),
        .BN_SCALE_T  (BN_SCALE_T),
        .BN_SCALE_AH (BN_SCALE_AH),
        .BN_OFFSET_V (BN_OFFSET_V),
        .BN_OFFSET_I (BN_OFFSET_I),
        .BN_OFFSET_T (BN_OFFSET_T),
        .BN_OFFSET_AH(BN_OFFSET_AH),
        .DATA_WIDTH  (DATA_WIDTH)
    ) u_bn (
        .clk      (slow_clk),
        .rst      (rst),
        .valid_in (mm_valid),
        .v_in     (v_mm),
        .i_in     (i_mm),
        .t_in     (t_mm),
        .ah_in    (ah_mm),
        .v_out    (v_bn),
        .i_out    (i_bn),
        .t_out    (t_bn),
        .ah_out   (ah_bn),
        .valid_out(bn_valid)
    );

    lstmtop #(
        .DATA_WIDTH (DATA_WIDTH),
        .NUM_NEURONS(16),
        .NUM_INPUTS (4)
    ) u_lstm (
        .clk         (slow_clk),
        .rst         (rst),
        .sensor_valid(bn_valid),
        .v_in        (v_bn),
        .i_in        (i_bn),
        .t_in        (t_bn),
        .ah_in       (ah_bn),
        .soc_out     (soc_out),
        .soc_valid   (soc_valid)
    );

endmodule
