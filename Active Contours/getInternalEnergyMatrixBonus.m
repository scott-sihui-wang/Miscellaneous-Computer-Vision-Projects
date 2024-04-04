function [Ainv] = getInternalEnergyMatrix(nPoints, alpha, beta, gamma)
    k=1;
    for l=1:nPoints
        i(k)=l;j(k)=l;v(k)=2*alpha+6*beta;k=k+1;
        i(k)=l;j(k)=mod(l,nPoints)+1;v(k)=-alpha-4*beta;k=k+1;
        i(k)=l;j(k)=mod(mod(l,nPoints)+1,nPoints)+1;v(k)=beta;k=k+1;
        i(k)=mod(l,nPoints)+1;j(k)=l;v(k)=-alpha-4*beta;k=k+1;
        i(k)=mod(mod(l,nPoints)+1,nPoints)+1;j(k)=l;v(k)=beta;k=k+1;
    end
    A=sparse(i,j,v);
    Ainv=inv(A+gamma*eye(nPoints));
end

