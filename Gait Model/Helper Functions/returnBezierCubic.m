function bezierCoeffs = returnBezierCubic(t)

    bezierCoeffs = [(1-t).^3,...
     3*(1-t).^2.*t,...
     3*(1-t).*(t).^2,...
     (t).^3];
end