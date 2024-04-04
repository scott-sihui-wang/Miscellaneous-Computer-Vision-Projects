function Eext = getExternalEnergy(I,Wline,Wedge,Wterm)

% Eline

Eline=I;

% Eedge

Eedge=imgradient(I); %Magnitude
Eedge=Eedge.*Eedge;
Eedge=(-1)*Eedge;

% Eterm
D_x=[0 0 0; 0 -1 1; 0 0 0];
D_y=[0 0 0; 0 -1 0; 0 1 0];
C_x=conv2(I,D_x,'same');
C_y=conv2(I,D_y,'same');
C_xx=conv2(C_x,D_x,'same');
C_xy=conv2(C_x,D_y,'same');
C_yy=conv2(C_y,D_y,'same');
Eterm=(C_yy.*C_x.*C_x-2*C_xy.*C_x.*C_y+C_xx.*C_y.*C_y)./((C_x.*C_x+C_y.*C_y+1).^1.5);

% Eext
Eext=Wline*Eline+Wedge*Eedge+Wterm*Eterm;

end

