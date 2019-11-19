function dX = RayPle(t, X)
    x = X(1);
    y = X(2);
    
    rho = 1000;
    P = 2300;
    P0 = 10000;
    
    dX = [y; -1.5*y*y/x + (1.0/rho)*(P-P0)/x];
end