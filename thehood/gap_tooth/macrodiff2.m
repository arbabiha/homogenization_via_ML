function ut = macrodiff2(t,u)
  global macros
  uu=reshape(u,round(sqrt(length(u))),[]);
  ut = macros.A*diff(diff(uu([end, 1:end, 1],:),1),1)/macros.dx^2 ...
      + macros.B*diff(diff(uu(:,[end, 1:end, 1]),1,2),1,2)/macros.dy^2 ...
      + macros.C*diff(diff(uu([1:end, 1],[1:end, 1]),1),1,2)/macros.dy/macros.dx; 
  ut=ut(:);
end% function