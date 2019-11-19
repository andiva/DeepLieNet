warning('off','all')

clo = [0    0.4470    0.7410;
       0.8500    0.3250    0.0980;
       0.9290    0.6940    0.1250;
       0.4940    0.1840    0.5560;
       0.4660    0.6740    0.1880;
       0.3010    0.7450    0.9330;
       0.6350    0.0780    0.1840;];


options = odeset('RelTol',1e-7);
tic
[t, X] = ode45(@(t,x)RayPle(t,x), [0 2], X0, options);
toc

plot(t, X(:, 1), '-', 'color', clo(1,:), 'MarkerSize', 2, 'LineWidth', 2);
grid on;
hold on;

warning('on','all')
