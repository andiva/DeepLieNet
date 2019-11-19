function X = RK4(f, x0, N)
    %t0 = 0;
    n_steps = 30;
    h = pi/(4*n_steps);
    X = zeros(length(x0), N+1);
    X(:, 1) = x0;
    %t = t0;
    for i = 2:N+1
        X(:, i) = X(:, i-1);
        for j=1:n_steps
            k1 = f(X(:,i));
            k2 = f(X(:,i) + .5*k1*h);
            k3 = f(X(:,i) + .5*k2*h);
            k4 = f(X(:,i) + k3*h);

            X(:,i) = X(:,i) + h*(k1+2*k2+2*k3+k4)/6;
            %t = t+h;
        end
    end
    X = X(:, 2:N+1);
end