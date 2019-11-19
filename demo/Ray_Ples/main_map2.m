
'ode45'
main_ode;

'Taylor mapping'


c  ={%@(X, N)map7_4(X,N),
     %@(X, N)map7_4_05(X,N),
     @(X, N)map7_5(X,N),
     @(X, N)map7_5_05(X,N),
     @(X, N)map7_6(X,N),
     @(X, N)map7_6_05(X,N),
     @(X, N)map7_7(X,N),
     @(X, N)map7_7_05(X,N),
     @(X, N)map7_8(X,N),
     @(X, N)map7_8_05(X,N),
     @(X, N)map7_9(X,N),
     @(X, N)map7_9_05(X,N),
     @(X, N)map7_10(X,N),
     @(X, N)map7_10_05(X,N),
     @(X, N)map7_11(X,N),
     @(X, N)map7_11_05(X,N),
     @(X, N)map7_12(X,N),
     @(X, N)map7_12_05(X,N),
     @(X, N)map7_13(X,N),
     @(X, N)map7_13_05(X,N),
     @(X, N)map7_14(X,N),
     @(X, N)map7_14_05(X,N),
     @(X, N)map7_15(X,N),
     @(X, N)map7_15_05(X,N),
     @(X, N)map7_16(X,N),
     @(X, N)map7_16_05(X,N),
     @(X, N)map7_17(X,N),
     @(X, N)map7_17_05(X,N),
     @(X, N)map7_18(X,N),
     @(X, N)map7_18_05(X,N),
     @(X, N)map7_19(X,N),
     @(X, N)map7_19_05(X,N),
    };


%dt = [4:1:19];
dt = [%1e-4, 0.5e-4,
      1e-5, 0.5e-5, 1e-6, 0.5e-6, 1e-7, 0.5e-7, 1e-8, 0.5e-8,...
      1e-9, 0.5e-9, 1e-10, 0.5e-10, 1e-11, 0.5e-11, 1e-12, 0.5e-12, 1e-13, 0.5e-13,...
      1e-14, 0.5e-14, 1e-15, 0.5e-15, 1e-16, 0.5e-16, 1e-17, 0.5e-17, 1e-18, 0.5e-18, 1e-19, 0.5e-19,];
dt=dt*(1-2.78e-4);%1e-7
%dt=dt;%1e-6

X_last = X0;
buf_n=1000;
X_map = zeros(2, buf_n);
X_map(:,1) = X0;
t_map = zeros(2, buf_n);

cur_f=1;
tic
%for i=1:250
tol = 1e-7;
n=1;
x_tol = X(end,1);
while X_last(1)>x_tol%n_attemps<2   
    s = size(c);s=s(1);
    
    map_f = c{cur_f};
    X5 = map_f([X_last;1/X_last(1)], 1);
    er = abs(X5(1)-1/X5(3));


    if X5(1)<x_tol & cur_f<s
        cur_f = cur_f+1;
        continue;
    end
    
    
    X_last = X5(1:end-1);
    n=n+1;
    X_map(:, n) = X_last;
    t_map(n) = t_map(n-1)+dt(cur_f);
    

    if er > tol & cur_f<s
        cur_f = cur_f+1;
    end
end
toc
X_map =  X_map(:, 1:n-1);
t_map = t_map(1:n-1);


plot(t_map, X_map(1,:), '-o', 'color', clo(2,:), 'MarkerSize', 3)%,'MarkerFaceColor', clo(2,:))


%'relative time shift:'
%(t_map(end) - t(end))/t(end)

