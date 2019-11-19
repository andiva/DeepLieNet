clc;
clear;

Req = 10;
defl = @(x)[x(2); -2*x(1) + x(1)^2/Req];

N=10000;

'Runge-Kutta 4th'
tic
x = RK4(defl, [0.1;0], N);
toc

hold on

'Taylor mapping'
tic
xm = map_defl_alpha([0.1;0], N);
toc

plot(xm(1,:), xm(2,:), 'ro', 'MarkerSize', 3)
plot(x(1,:), x(2,:), 'b*', 'MarkerSize', 2)