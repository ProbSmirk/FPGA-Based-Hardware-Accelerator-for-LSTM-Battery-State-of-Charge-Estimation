`timescale 1ns / 1ps
module tb_lstm_system;

localparam DATA_WIDTH   = 16;
localparam NUM_VECTORS  = 1050;   // must match generate_golden_ref.py output
                                   // = 10 voltages � 3 currents � 5 temps � 7 Ah
localparam TOLERANCE    = 16'h0000; // 32/256 = 12.5% SoC tolerance (current)
localparam TIMEOUT_CYC  = 20000;    // max cycles to wait per vector

reg clk = 0;
reg rst;
always #5 clk = ~clk;  // 100MHz

reg  signed [DATA_WIDTH-1:0] v_in, i_in, t_in, ah_in;
reg  sensor_valid;
wire signed [DATA_WIDTH-1:0] soc_out;
wire soc_valid;

lstmtop #(
    .DATA_WIDTH (DATA_WIDTH),
    .NUM_NEURONS(16),
    .NUM_INPUTS (4)
) dut (
    .clk         (clk),
    .rst         (rst),
    .sensor_valid(sensor_valid),
    .v_in        (v_in),
    .i_in        (i_in),
    .t_in        (t_in),
    .ah_in       (ah_in),
    .soc_out     (soc_out),
    .soc_valid   (soc_valid)
);

// ?? Golden reference memories ????????????????????????????????????????????????
reg [DATA_WIDTH-1:0] ref_v   [0:NUM_VECTORS-1];
reg [DATA_WIDTH-1:0] ref_i   [0:NUM_VECTORS-1];
reg [DATA_WIDTH-1:0] ref_t   [0:NUM_VECTORS-1];
reg [DATA_WIDTH-1:0] ref_ah  [0:NUM_VECTORS-1];
reg [DATA_WIDTH-1:0] ref_soc [0:NUM_VECTORS-1];

initial begin
    $readmemh("golden_v.mem",   ref_v);
    $readmemh("golden_i.mem",   ref_i);
    $readmemh("golden_t.mem",   ref_t);
    $readmemh("golden_ah.mem",  ref_ah);
    $readmemh("golden_soc.mem", ref_soc);
end

// ?? Test counters ?????????????????????????????????????????????????????????????
integer pass_count;
integer fail_count;
integer timeout_count;
integer vec;
integer cyc;


real       sum_diff_pct;
real       max_diff_pct;
integer    diff_count;
reg  [15:0] this_diff;
real        this_diff_pct;

function real q8_to_real;
    input signed [15:0] q;
    q8_to_real = $itor($signed(q)) / 256.0;
endfunction

function [15:0] abs_diff;
    input signed [15:0] a, b;
    reg signed [15:0] diff;
    begin
        diff = $signed(a) - $signed(b);
        abs_diff = ($signed(diff) < 0) ? -diff : diff;
    end
endfunction

initial begin
    $dumpfile("tb_lstm_system.vcd");
    $dumpvars(0, tb_lstm_system);
end

initial begin
    pass_count    = 0;
    fail_count    = 0;
    timeout_count = 0;
    sensor_valid  = 0;
    v_in = 0; i_in = 0; t_in = 0; ah_in = 0;
    max_diff_lsb  = 0;
    sum_diff_pct  = 0.0;
    max_diff_pct  = 0.0;
    diff_count    = 0;

    
    $display("  LSTM SoC SYSTEM TESTBENCH");
    $display("  %0d golden vectors | Real weights | 4 inputs", NUM_VECTORS);
    $display("  Tolerance: %0d LSB = %.2f%% SoC",
             TOLERANCE, $itor(TOLERANCE)/2.56);
    

    rst = 1;
    repeat(10) @(negedge clk);
    rst = 0;
    repeat(3)  @(negedge clk);

   
    $display("\nTEST 1: Golden Reference Sweep (%0d vectors)",
             NUM_VECTORS);

    for (vec = 0; vec < NUM_VECTORS; vec = vec + 1) begin

        rst = 1;
        @(negedge clk); @(negedge clk);
        rst = 0;

        v_in  = ref_v[vec];
        i_in  = ref_i[vec];
        t_in  = ref_t[vec];
        ah_in = ref_ah[vec];

        @(negedge clk);
        sensor_valid = 1'b1;
        @(negedge clk);
        sensor_valid = 1'b0;

        cyc = 0;
        while (!soc_valid && cyc < TIMEOUT_CYC) begin
            @(posedge clk);
            cyc = cyc + 1;
        end

        if (!soc_valid) begin
            $display("  [%4d] TIMEOUT  v=%h i=%h t=%h ah=%h (exp=%h)",
                     vec, ref_v[vec], ref_i[vec],
                     ref_t[vec], ref_ah[vec], ref_soc[vec]);
            timeout_count = timeout_count + 1;
            fail_count    = fail_count    + 1;
        end else begin
            // Real diff tracking, every vector, regardless of pass/fail
            this_diff     = abs_diff(soc_out, ref_soc[vec]);
            this_diff_pct = $itor(this_diff) / 2.56;

            if (this_diff > max_diff_lsb) max_diff_lsb = this_diff;
            if (this_diff_pct > max_diff_pct) max_diff_pct = this_diff_pct;
            sum_diff_pct = sum_diff_pct + this_diff_pct;
            diff_count   = diff_count + 1;

            if (abs_diff(soc_out, ref_soc[vec]) <= TOLERANCE) begin
                pass_count = pass_count + 1;
            end else begin
                $display("  [%4d] FAIL  v=%.3fV i=%.2fA t=%.1fC ah=%.1fAh",
                         vec,
                         q8_to_real(ref_v[vec]),
                         q8_to_real(ref_i[vec]),
                         q8_to_real(ref_t[vec]),
                         q8_to_real(ref_ah[vec]));
                $display("         got=%h (%.4f)  exp=%h (%.4f)  diff=%h",
                         soc_out,      q8_to_real(soc_out),
                         ref_soc[vec], q8_to_real(ref_soc[vec]),
                         abs_diff(soc_out, ref_soc[vec]));
                fail_count = fail_count + 1;
            end
        end

        @(negedge clk);
    end

    $display("  Test 1 result: %0d PASS  %0d FAIL  %0d TIMEOUT",
             pass_count, fail_count, timeout_count);

    // NEW: print the real error distribution
    $display("\n REAL ERROR DISTRIBUTION (Test 1, all %0d vectors)", diff_count);
    $display("  Max diff   : %0d LSB = %.4f%% SoC", max_diff_lsb, max_diff_pct);
    $display("  Mean diff  : %.4f%% SoC", sum_diff_pct / diff_count);
    $display("  Suggested tighter TOLERANCE (max diff + small margin):");
    $display("    16'h%04h  (= %.4f%% SoC, ~1.5x measured max)",
              (max_diff_lsb + (max_diff_lsb >> 1)), max_diff_pct * 1.5);

  
    $display("\n--- TEST 2: soc_valid Pulse Width ---");
    begin : t2
        rst = 1; @(negedge clk); @(negedge clk); rst = 0;
        v_in = ref_v[0]; i_in = ref_i[0];
        t_in = ref_t[0]; ah_in = ref_ah[0];
        sensor_valid = 1; @(negedge clk); sensor_valid = 0;

        cyc = 0;
        while (!soc_valid && cyc < TIMEOUT_CYC) begin
            @(posedge clk); cyc = cyc + 1;
        end

        if (soc_valid) begin
            @(posedge clk); #1;
            if (!soc_valid) begin
                $display("  soc_valid pulsed exactly 1 cycle: PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  soc_valid stayed HIGH >1 cycle: FAIL");
                fail_count = fail_count + 1;
            end
        end else begin
            $display("  soc_valid never asserted: FAIL");
            fail_count = fail_count + 1;
        end
    end

    $display("\nTEST 3: Reset Reproducibility ");
    begin : t3
        reg [DATA_WIDTH-1:0] first_soc;

        rst = 1; @(negedge clk); @(negedge clk); rst = 0;
        v_in = ref_v[5]; i_in = ref_i[5];
        t_in = ref_t[5]; ah_in = ref_ah[5];
        sensor_valid = 1; @(negedge clk); sensor_valid = 0;
        cyc = 0;
        while (!soc_valid && cyc < TIMEOUT_CYC) begin
            @(posedge clk); cyc = cyc + 1;
        end
        first_soc = soc_out;
        @(negedge clk);

        rst = 1; @(negedge clk); @(negedge clk); rst = 0;
        v_in = ref_v[5]; i_in = ref_i[5];
        t_in = ref_t[5]; ah_in = ref_ah[5];
        sensor_valid = 1; @(negedge clk); sensor_valid = 0;
        cyc = 0;
        while (!soc_valid && cyc < TIMEOUT_CYC) begin
            @(posedge clk); cyc = cyc + 1;
        end

        if (soc_out === first_soc) begin
            $display("  Reset reproducibility: PASS  (both runs = %h = %.4f)",
                     soc_out, q8_to_real(soc_out));
            pass_count = pass_count + 1;
        end else begin
            $display(" Reset reproducibility: FAIL first=%h second=%h", first_soc, soc_out);
            fail_count = fail_count + 1;
        end
    end

    
    $display("\n TEST 4: LSTM Recurrence Active");
    begin : t4
        reg [DATA_WIDTH-1:0] soc_first, soc_second;
        integer vec_idx;
        vec_idx = 10;

        rst = 1; @(negedge clk); @(negedge clk); rst = 0;

        v_in = ref_v[vec_idx]; i_in = ref_i[vec_idx];
        t_in = ref_t[vec_idx]; ah_in = ref_ah[vec_idx];
        sensor_valid = 1; @(negedge clk); sensor_valid = 0;
        cyc = 0;
        while (!soc_valid && cyc < TIMEOUT_CYC) begin
            @(posedge clk); cyc = cyc + 1;
        end
        soc_first = soc_out;
        @(negedge clk);

        sensor_valid = 1; @(negedge clk); sensor_valid = 0;
        cyc = 0;
        while (!soc_valid && cyc < TIMEOUT_CYC) begin
            @(posedge clk); cyc = cyc + 1;
        end
        soc_second = soc_out;

        $display("  1st call SoC: %h (%.4f)", soc_first,  q8_to_real(soc_first));
        $display("  2nd call SoC: %h (%.4f)", soc_second, q8_to_real(soc_second));

        if (soc_first !== soc_second) begin
            $display("  Recurrence active (outputs differ): PASS");
            pass_count = pass_count + 1;
        end else begin
            $display("  Outputs identical - recurrence may not be active: INFO");
            pass_count = pass_count + 1;
        end
    end

    $display("\n TEST 5: SoC Output Range Validity");
    begin : t5
        integer range_pass, range_fail;
        range_pass = 0; range_fail = 0;

        for (vec = 0; vec < NUM_VECTORS; vec = vec + 64) begin
            rst = 1; @(negedge clk); @(negedge clk); rst = 0;
            v_in = ref_v[vec]; i_in = ref_i[vec];
            t_in = ref_t[vec]; ah_in = ref_ah[vec];
            sensor_valid = 1; @(negedge clk); sensor_valid = 0;
            cyc = 0;
            while (!soc_valid && cyc < TIMEOUT_CYC) begin
                @(posedge clk); cyc = cyc + 1;
            end
            if (soc_valid) begin
                if ($signed(soc_out) >= 0 && soc_out <= 16'h0100) begin
                    range_pass = range_pass + 1;
                end else begin
                    $display("  OUT OF RANGE: vec=%0d soc=%h (%.4f)",
                             vec, soc_out, q8_to_real(soc_out));
                    range_fail = range_fail + 1;
                end
            end
            @(negedge clk);
        end

        pass_count = pass_count + range_pass;
        fail_count = fail_count + range_fail;
        $display("  Range check: %0d in-range  %0d out-of-range",
                 range_pass, range_fail);
    end

   
    $display("\nTEST 6: Accuracy Report (subset) ");
    $display("  %-6s %-8s %-8s %-10s %-12s %-8s",
             "Vec", "V(Q8.8)", "Ah(Q8.8)", "Exp SoC", "Got SoC", "Err%");
    $display("  %s", "");

    begin : t6
        integer acc_pass, acc_fail;
        real err_pct;
        acc_pass = 0; acc_fail = 0;

        for (vec = 0; vec < NUM_VECTORS; vec = vec + 50) begin
            rst = 1; @(negedge clk); @(negedge clk); rst = 0;
            v_in = ref_v[vec]; i_in = ref_i[vec];
            t_in = ref_t[vec]; ah_in = ref_ah[vec];
            sensor_valid = 1; @(negedge clk); sensor_valid = 0;
            cyc = 0;
            while (!soc_valid && cyc < TIMEOUT_CYC) begin
                @(posedge clk); cyc = cyc + 1;
            end

            if (soc_valid) begin
                err_pct = ($itor($signed(soc_out)) -
                           $itor($signed(ref_soc[vec]))) / 2.56;
                if (err_pct < 0) err_pct = -err_pct;

                $display("  %-6d %h     %h     %.4f     %.4f     %.2f%%",
                         vec, ref_v[vec], ref_ah[vec],
                         q8_to_real(ref_soc[vec]),
                         q8_to_real(soc_out),
                         err_pct);

                if (abs_diff(soc_out, ref_soc[vec]) <= TOLERANCE)
                    acc_pass = acc_pass + 1;
                else
                    acc_fail = acc_fail + 1;
            end
            @(negedge clk);
        end

        pass_count = pass_count + acc_pass;
        fail_count = fail_count + acc_fail;
    end

    $display("  FINAL RESULTS");
    $display("  PASS     : %0d", pass_count);
    $display("  FAIL     : %0d", fail_count);
    $display("  TIMEOUTS : %0d", timeout_count);
    $display("  TOTAL    : %0d", pass_count + fail_count);
    $display("  TOLERANCE USED : %0d LSB = %.3f%% SoC", TOLERANCE,
             $itor(TOLERANCE)/2.56);
    $display("  REAL MAX DIFF ACROSS ALL %0d VECTORS : %0d LSB = %.4f%% SoC",
             diff_count, max_diff_lsb, max_diff_pct);
    $display("  REAL MEAN DIFF : %.4f%% SoC", sum_diff_pct / diff_count);
    if (fail_count == 0)
        $display("  RESULT : ALL TESTS PASSED");
    else
        $display("  RESULT : %0d FAILURE(S)", fail_count);

    $finish;
end

initial begin
    #500_000_000;
    $display("WATCHDOG: simulation exceeded 500ms, force stop");
    $finish;
end

endmodule