function bezierCoeffs = returnBezierQuadratic(t)
    bezierCoeffs = [(1-t).^2,...
     2*(1-t).*t,...
     (t).^2];
end