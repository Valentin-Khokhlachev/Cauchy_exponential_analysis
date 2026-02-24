clear; clc;

M = readmatrix("../results/interpolation/convergence.csv");

N = M(:, 1);
log10_abs_err_int = M(:, 2);
log10_abs_err_interp = M(:, 3);
log10_abs_err_der = M(:, 4);

f = figure;
hold on;
grid on;
plot(N, log10_abs_err_int,...
    "Marker",".",...
    "DisplayName","error Cauchy integration");
plot(N, log10_abs_err_interp,...
    "Marker",".",...
    "DisplayName","error Cauchy interpolation");
plot(N, log10_abs_err_der,...
    "Marker",".",...
    "DisplayName","error Cauchy differentiation");
legend("Location","best");
xlabel("M=N");
ylabel("log_{10}||error||_{L2}");
saveas(f, "graph/Cauchy_exp_convergence", 'fig');
saveas(f, "graph/Cauchy_exp_convergence", 'png');


