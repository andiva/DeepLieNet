%code generation
%mapping function
function X = map(X0, n)
%map generation:
    R0=[0;
        0;
        0;];
    R1=[1 5E-06 -9.625E-11;
        0 1 -3.85E-05;
        0 0 1;];
    R2=[0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;];
    R3=[0 0 0 0 0 0 0 -1.875E-11 6.41666666666667E-16 -6.170328828125E-21;
        0 0 0 0 0 0 0 -7.5E-06 3.85E-10 -4.94083333333333E-15;
        0 0 0 0 0 0 0 0 -5E-06 9.625E-11;];
    R4=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;];
    R5=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.25E-16 -6.40405390625E-21 1.16718077838542E-25 -7.45067924282605E-31;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7.5E-11 -5.13333333333333E-15 1.17293376119792E-19 -9.02451904972004E-25;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4.375E-11 -1.60416666666667E-15 1.5434391328125E-20;];
    R6=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;];
    R7=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.013384765625E-21 6.9317294921875E-26 -1.85735849931125E-30 2.28927484008901E-35 -1.08950246586377E-40;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -8.125E-16 6.97270893229166E-20 -2.25275798786567E-24 3.26029591450204E-29 -1.78794687720899E-34;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -4.375E-16 2.36399424479167E-20 -4.40545338272569E-25 2.82000882409858E-30;];

    X = zeros(length(X0), n+1);
    X(:,1) = X0;
    for i=1:n
        X(:, i+1) = mfcalc(X(:,i), R0, R1, R2, R3, R4, R5, R6, R7);
    end
X = X(:, 2:n+1);
end


%one-tune solution
function X = mfcalc(X0, R0, R1, R2, R3, R4, R5, R6, R7)
%initial state:
    x0=X0(1);
    x1=X0(2);
    x2=X0(3);
%calculation of kronecker pows:
    X1=[x0;x1;x2];
    X2=[x0*x0;x0*x1;x0*x2;x1*x1;x1*x2;x2*x2];
    X3=[x0*x0*x0;x0*x0*x1;x0*x0*x2;x0*x1*x1;x0*x1*x2;x0*x2*x2;x1*x1*x1;x1*x1*x2;x1*x2*x2;x2*x2*x2];
    X4=[x0*x0*x0*x0;x0*x0*x0*x1;x0*x0*x0*x2;x0*x0*x1*x1;x0*x0*x1*x2;x0*x0*x2*x2;x0*x1*x1*x1;x0*x1*x1*x2;x0*x1*x2*x2;x0*x2*x2*x2;x1*x1*x1*x1;x1*x1*x1*x2;x1*x1*x2*x2;x1*x2*x2*x2;x2*x2*x2*x2];
    X5=[x0*x0*x0*x0*x0;x0*x0*x0*x0*x1;x0*x0*x0*x0*x2;x0*x0*x0*x1*x1;x0*x0*x0*x1*x2;x0*x0*x0*x2*x2;x0*x0*x1*x1*x1;x0*x0*x1*x1*x2;x0*x0*x1*x2*x2;x0*x0*x2*x2*x2;x0*x1*x1*x1*x1;x0*x1*x1*x1*x2;x0*x1*x1*x2*x2;x0*x1*x2*x2*x2;x0*x2*x2*x2*x2;x1*x1*x1*x1*x1;x1*x1*x1*x1*x2;x1*x1*x1*x2*x2;x1*x1*x2*x2*x2;x1*x2*x2*x2*x2;x2*x2*x2*x2*x2];
    X6=[x0*x0*x0*x0*x0*x0;x0*x0*x0*x0*x0*x1;x0*x0*x0*x0*x0*x2;x0*x0*x0*x0*x1*x1;x0*x0*x0*x0*x1*x2;x0*x0*x0*x0*x2*x2;x0*x0*x0*x1*x1*x1;x0*x0*x0*x1*x1*x2;x0*x0*x0*x1*x2*x2;x0*x0*x0*x2*x2*x2;x0*x0*x1*x1*x1*x1;x0*x0*x1*x1*x1*x2;x0*x0*x1*x1*x2*x2;x0*x0*x1*x2*x2*x2;x0*x0*x2*x2*x2*x2;x0*x1*x1*x1*x1*x1;x0*x1*x1*x1*x1*x2;x0*x1*x1*x1*x2*x2;x0*x1*x1*x2*x2*x2;x0*x1*x2*x2*x2*x2;x0*x2*x2*x2*x2*x2;x1*x1*x1*x1*x1*x1;x1*x1*x1*x1*x1*x2;x1*x1*x1*x1*x2*x2;x1*x1*x1*x2*x2*x2;x1*x1*x2*x2*x2*x2;x1*x2*x2*x2*x2*x2;x2*x2*x2*x2*x2*x2];
    X7=[x0*x0*x0*x0*x0*x0*x0;x0*x0*x0*x0*x0*x0*x1;x0*x0*x0*x0*x0*x0*x2;x0*x0*x0*x0*x0*x1*x1;x0*x0*x0*x0*x0*x1*x2;x0*x0*x0*x0*x0*x2*x2;x0*x0*x0*x0*x1*x1*x1;x0*x0*x0*x0*x1*x1*x2;x0*x0*x0*x0*x1*x2*x2;x0*x0*x0*x0*x2*x2*x2;x0*x0*x0*x1*x1*x1*x1;x0*x0*x0*x1*x1*x1*x2;x0*x0*x0*x1*x1*x2*x2;x0*x0*x0*x1*x2*x2*x2;x0*x0*x0*x2*x2*x2*x2;x0*x0*x1*x1*x1*x1*x1;x0*x0*x1*x1*x1*x1*x2;x0*x0*x1*x1*x1*x2*x2;x0*x0*x1*x1*x2*x2*x2;x0*x0*x1*x2*x2*x2*x2;x0*x0*x2*x2*x2*x2*x2;x0*x1*x1*x1*x1*x1*x1;x0*x1*x1*x1*x1*x1*x2;x0*x1*x1*x1*x1*x2*x2;x0*x1*x1*x1*x2*x2*x2;x0*x1*x1*x2*x2*x2*x2;x0*x1*x2*x2*x2*x2*x2;x0*x2*x2*x2*x2*x2*x2;x1*x1*x1*x1*x1*x1*x1;x1*x1*x1*x1*x1*x1*x2;x1*x1*x1*x1*x1*x2*x2;x1*x1*x1*x1*x2*x2*x2;x1*x1*x1*x2*x2*x2*x2;x1*x1*x2*x2*x2*x2*x2;x1*x2*x2*x2*x2*x2*x2;x2*x2*x2*x2*x2*x2*x2];
%solution:
    X=R0+R1*X1+R2*X2+R3*X3+R4*X4+R5*X5+R6*X6+R7*X7;
end
