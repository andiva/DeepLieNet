%code generation
%mapping function
function X = map(X0, n)
%map generation:
    R0=[0;
        0;
        0;];
    R1=[1 1E-11 -3.85E-22;
        0 1 -7.7E-11;
        0 0 1;];
    R2=[0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;];
    R3=[0 0 0 0 0 0 0 -7.5E-23 4.9896E-33 -9.241087625E-44;
        0 0 0 0 0 0 0 -1.5E-11 1.54E-21 -3.924998E-32;
        0 0 0 0 0 0 0 0 -1E-11 3.85E-22;];
    R4=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;];
    R5=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9.72E-34 -9.45646625E-44 3.26340024159722E-54 -3.90166766086319E-65;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3E-22 -4.04558E-32 1.81516335E-42 -2.72283858379653E-53;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.75E-22 -1.26177333333333E-32 2.3759726375E-43;];
    R6=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;];
    R7=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.491478125E-44 1.91402937152778E-54 -9.55410407447222E-65 2.17472922589719E-75 -1.89348664612991E-86;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -6.3915E-33 1.070186425E-42 -6.71171765553819E-53 1.87414164118324E-63 -1.96956042178101E-74;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -3.43933333333333E-33 3.612503125E-43 -1.30146103801389E-53 1.59985768002691E-64;];

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
