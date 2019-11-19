%code generation
%mapping function
function X = map(X0, n)
%map generation:
    R0=[0;
        0;];
    R1=[0.44401584034725 0.633581065658028;
        -1.26716213131606 0.44401584034725;];
    R2=[0.0226472348838309 0.0117420345441513 0.00257598654793418;
        0.0398740374708035 0.0349905235759251 0.0117420345441514;];
    R3=[0.000207520496740676 0.000174502169909385 4.6571306739075E-05 5.64626978806772E-06;
        0.00082519911514193 0.000951473572798119 0.000315126721090362 4.6571306739074E-05;];

    X = zeros(length(X0), n+1);
    X(:,1) = X0;
    for i=1:n
        X(:, i+1) = mfcalc(X(:,i), R0, R1, R2, R3);
    end
X = X(:, 2:n+1);
end


%one-tune solution
function X = mfcalc(X0, R0, R1, R2, R3)
%initial state:
    x0=X0(1);
    x1=X0(2);
%calculation of kronecker pows:
    X1=[x0;x1];
    X2=[x0*x0;x0*x1;x1*x1];
    X3=[x0*x0*x0;x0*x0*x1;x0*x1*x1;x1*x1*x1];
%solution:
    X=R0+R1*X1+R2*X2+R3*X3;
end
